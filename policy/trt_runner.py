from threading import RLock

import torch

from policy.rgb_feature_cache import RGBFeatureCache
from policy.engine_metadata import read_metadata, validate_runtime_config


class _TensorRTRunner:
    """Execute a named component on the current Torch CUDA stream."""

    def __init__(self, engine_path, input_names, output_names, device="cuda"):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("TensorRT inference requires CUDA.")
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError("TensorRT Python package is required to load .engine files.") from exc
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with torch.cuda.device(self.device):
            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
            self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT context: {engine_path}")
        actual_inputs, actual_outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            target = actual_inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else actual_outputs
            target.append(name)
        if set(actual_inputs) != set(input_names) or set(actual_outputs) != set(output_names):
            raise RuntimeError(
                f"Unexpected TensorRT bindings in {engine_path}: "
                f"inputs={actual_inputs}, outputs={actual_outputs}; "
                f"expected inputs={list(input_names)}, outputs={list(output_names)}. "
                "Dual-score engines must be copied together with their .engine.json files."
            )
        self.input_names, self.output_names = tuple(input_names), tuple(output_names)
        self._lock = RLock()
        self._last_execution = None

    @torch.inference_mode()
    def run(self, inputs):
        with self._lock, torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            # A context may be used from successive ROS callback streams, but
            # must not execute overlapping requests on different streams.
            if self._last_execution is not None:
                stream.wait_event(self._last_execution)
            bound_inputs = {}
            for name in self.input_names:
                source = inputs[name]
                if not source.is_cuda or source.device != stream.device:
                    raise ValueError(f"TensorRT input {name} must be on {stream.device}")
                tensor = source.to(dtype=self._torch_dtype(self.engine.get_tensor_dtype(name))).contiguous()
                if not self.context.set_input_shape(name, tuple(tensor.shape)):
                    raise ValueError(f"TensorRT input {name} has incompatible shape {tuple(tensor.shape)}")
                self.context.set_tensor_address(name, tensor.data_ptr())
                tensor.record_stream(stream)
                bound_inputs[name] = tensor
            outputs = []
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                if any(dim < 0 for dim in shape):
                    raise RuntimeError(f"Unresolved TensorRT output shape for {name}: {shape}")
                # Fresh storage matters: cached encoder outputs must survive
                # subsequent calls without being overwritten by a shared buffer.
                tensor = torch.empty(shape, dtype=self._torch_dtype(self.engine.get_tensor_dtype(name)), device=stream.device)
                self.context.set_tensor_address(name, tensor.data_ptr())
                outputs.append(tensor)
            if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 failed.")
            self._last_execution = torch.cuda.Event()
            self._last_execution.record(stream)
            return tuple(outputs)

    def _torch_dtype(self, trt_dtype):
        for name, dtype in (("float32", torch.float32), ("float16", torch.float16),
                            ("int32", torch.int32), ("int8", torch.int8), ("bool", torch.bool)):
            if trt_dtype == getattr(self.trt, name, None):
                return dtype
        raise TypeError(f"Unsupported TensorRT dtype: {trt_dtype}")


class TensorRTPolicy(_TensorRTRunner):
    """Existing complete two-frame engine interface."""

    def __init__(self, engine_path, device="cuda"):
        from config.config import cfg
        metadata = read_metadata(engine_path)
        validate_runtime_config(metadata, cfg)
        if metadata is not None and metadata['role'] != 'full':
            raise ValueError('Expected a full engine, not a split component')
        self.score_mode = metadata['score_mode'] if metadata else 'selection'
        self.candidate_topk = metadata['candidate_topk'] if metadata else 8
        names = metadata['output_names'] if metadata else ('endstate', 'score')
        super().__init__(engine_path, ("rgb", "obs", "pose"), names, device)

    def __call__(self, rgb, obs, pose):
        return self.run({"rgb": rgb, "obs": obs, "pose": pose})


class CachedTensorRTPolicy:
    """Single-frame encoder + pose/temporal planner, with an explicit frame cache."""

    def __init__(self, encoder_engine, planner_engine, device="cuda", max_frames=2):
        from config.config import cfg
        encoder_meta, planner_meta = read_metadata(encoder_engine), read_metadata(planner_engine)
        if (encoder_meta is None) != (planner_meta is None):
            raise ValueError('Copy both split engines together with both .engine.json files')
        if encoder_meta is not None:
            if encoder_meta['role'] != 'encoder' or planner_meta['role'] != 'planner':
                raise ValueError('Encoder/planner engine roles are reversed or invalid')
            if encoder_meta['deployment_id'] != planner_meta['deployment_id']:
                raise ValueError('Encoder and planner engines come from different checkpoints/configurations')
            validate_runtime_config(encoder_meta, cfg)
            validate_runtime_config(planner_meta, cfg)
        self.score_mode = planner_meta['score_mode'] if planner_meta else 'selection'
        self.candidate_topk = planner_meta['candidate_topk'] if planner_meta else 8
        names = planner_meta['output_names'] if planner_meta else ('endstate', 'score')
        self.encoder = _TensorRTRunner(encoder_engine, ("rgb",), ("features",), device)
        self.planner = _TensorRTRunner(planner_engine, ("features", "obs", "pose"), names, device)
        self.feature_cache = RGBFeatureCache(self.encode_rgb_frame, max_frames=max_frames)

    def encode_rgb_frame(self, rgb):
        return self.encoder.run({"rgb": rgb})[0]

    def clear_cache(self):
        self.feature_cache.clear()

    def __call__(self, rgb, obs, pose, frame_ids=None):
        features = self.feature_cache.encode_sequence(rgb, frame_ids)
        return self.planner.run({"features": features, "obs": obs, "pose": pose})


def planning_outputs_for_ros(outputs):
    """Match the PyTorch ROS tuple; collision cost is slot 5, not depth slot 2."""
    if len(outputs) not in (2, 3):
        raise ValueError('Expected endstate, score and optional safety_score')
    return outputs[0], outputs[1], None, None, None, outputs[2] if len(outputs) == 3 else None
