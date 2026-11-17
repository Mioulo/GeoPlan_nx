"""
导出 YOPO 到 ONNX，并可选择构建 TensorRT engine。
以下常用命令按特征缓存部署：使用 --split_encoder 导出两个 engine。

部署说明：
    1. TensorRT engine 和硬件、TensorRT 版本、CUDA/cuDNN 环境强相关。
       可以在 PC 上导出 ONNX，但最终部署用的 .engine 应该在部署设备
       上构建和测试，例如 Orin/NX。
    2. 特征缓存模式将同一份 checkpoint 拆为两个 engine：
           yopo_encoder.engine:
               输入 rgb: [1, 3, image_height, image_width]
               输出 features: [1, 512, 17, 30]（当前 RGB 旁路配置）
           yopo_planner.engine:
               输入 features: [1, 2, 512, 17, 30]
               输入 obs: [1, 9]
               输入 pose: [1, 7]，相对位姿 [qx, qy, qz, qw, tx, ty, tz]
               输出 endstate、score；topk-collision 权重另有 safety_score
       推理程序缓存位姿融合前的历史帧特征，每次只编码新帧；位姿嵌入、
       空间注意力、时序融合和规划每次重新计算。无需重新训练，但两个
       engine 必须来自同一份新结构权重和配置，并重新导出、构建。
       上述 512 通道由 256 通道深度特征和 256 通道原始 RGB 特征组成；
       仅最后一帧的 RGB 特征参与规划。旧 256 通道 engine 不兼容。
    3. Jetson TensorRT 8.x 建议使用 opset 16。opset 17 可能导出
       LayerNormalization，TensorRT 8.5 不能直接解析。

常用命令（在 YOPO 目录执行，均使用特征缓存）：
    从 checkpoint 导出、构建两个 FP16 engine，并验证误差和延迟：
        python yopo_trt_transfer.py --weight saved/YOPO_NEW/epoch140.pth \
            --split_encoder --fp16 1 --benchmark 1000

    只导出两个 FP32 ONNX（无 CUDA 时也可在 CPU 上执行）：
        python yopo_trt_transfer.py --weight saved/YOPO_NEW/epoch140.pth \
            --split_encoder --opset 16 --no_engine
        默认生成 yopo_encoder.onnx、yopo_planner.onnx。
        YOPO_NEW 为示例目录，使用时替换成实际训练生成的 checkpoint。

    使用已有两个 ONNX 构建 FP16 engine，并与 checkpoint 对比：
        python yopo_trt_transfer.py --weight saved/YOPO_NEW/epoch140.pth \
            --split_encoder --fp16 1 --build_only --verify_after_build \
            --benchmark 1000

    验证已有两个 engine 的误差和连续推理延迟：
        python yopo_trt_transfer.py --weight saved/YOPO_NEW/epoch140.pth \
            --split_encoder --verify_only --benchmark 1000

    ROS 推理启用特征缓存：
        python test_yopo_ros.py --use_tensorrt 1 --use_feature_cache 1 \
            --encoder_engine yopo_encoder.engine \
            --planner_engine yopo_planner.engine \
            --publish_r2d_depth 0 --publish_depth_expert_traj 0
        tracker 使用相同参数，将脚本替换为 test_yopo_ros_tracker.py。

    拆分验证使用非零图像及变化位姿；测速期间每次恰好编码一张新图，
    包含两个 engine、缓存及特征拼接开销。首次输入两张不同图片时都
    需要编码，因此首次耗时与连续运行平均耗时不同。

单 engine 用法（不使用特征缓存）：
    不传 --split_encoder 时仍走原单 engine 路径，输入完整的两帧 RGB。
    --onnx、--engine、--fp16_onnx 参数只用于这个路径。

    从 checkpoint 导出并构建单 FP16 engine：
        python yopo_trt_transfer.py --weight saved/YOPO_NEW/epoch140.pth \
            --onnx yopo.onnx --engine yopo.engine --fp16 1 --benchmark 1000

    从已有 ONNX 构建单 INT8 engine，允许回退到 FP16：
        python yopo_trt_transfer.py --onnx yopo.onnx --engine yopo_int8.engine \
            --fp16 1 --int8 1 --calib_npz calib/calib_raw_abs.npz \
            --calib_pose_format absolute --calib_cache calib/yopo_int8.cache \
            --calib_batches 256 --workspace 8589934592 --build_only

TensorRT 参数说明：
    --split_encoder:
        导出、构建或验证单帧 RGB 编码器和特征输入规划器。拆分模式
        支持 FP16/FP32，暂不支持现有的 INT8 校准流程。

    --encoder_onnx、--planner_onnx:
        拆分 ONNX 路径，默认为 yopo_encoder.onnx、yopo_planner.onnx。

    --encoder_engine、--planner_engine:
        拆分 engine 路径，默认为 yopo_encoder.engine、yopo_planner.engine。
        ROS 推理时也要传入对应的两个路径，并启用 --use_feature_cache 1。

    --fp16:
        默认 1。支持 BuilderFlag.FP16 的 TensorRT 开启 FP16 tactic；
        TensorRT 11 等没有该标志的版本使用 FP16 ONNX 构建。
        拆分模式自动生成对应的 *_fp16.onnx，保留 FP32 输入输出接口。
        --build_only 会复用已有的 *_fp16.onnx；不存在时转换 FP32 ONNX，
        需要 ModelOpt 或 onnxconverter-common。更换权重或配置后应重新
        导出全部 ONNX，再构建两个 engine，避免复用旧的 FP16 文件。
        单 engine 的 INT8 部署也建议保持 --fp16 1，允许部分层回退到 FP16。

    --int8:
        仅用于单 engine，开启 INT8 后训练量化校准。TensorRT 8.x 使用
        IInt8EntropyCalibrator2。PC 上的 TensorRT 11 strong-typed 环境可能
        没有 BuilderFlag.INT8，这种情况不能用本脚本的 calibrator 流程，
        需要在 Jetson/Orin 上构建，或者另走 TensorRT 11 的 QDQ/ModelOpt
        量化流程。

    --workspace:
        TensorRT 构建 engine 时允许使用的临时 workspace 上限，单位是字节。
        workspace 越大，TensorRT build 时可尝试的 tactic 越多，可能得到
        更快的 engine。它不是最终推理时一定占用的显存。常用取值：
            2147483648  = 2GB
            4294967296  = 4GB
            8589934592  = 8GB

    --calib_npz:
        推荐的 INT8 校准输入。npz 可以包含：
            rgb:  [N, T, 3, H, W] 或 [N, T, H, W, 3]，float32 或 uint8
            obs:  [N, 9]，可选；缺失时使用 0
            pose: [N, 7] 相对位姿，或 [N, 2, 7] 两帧绝对位姿

    --calib_image_dir 和 --calib_state_npz:
        图像和状态分开保存时使用。calib_image_dir 提供 RGB 图像序列；
        calib_state_npz 提供 obs 和 pose。如果不提供 calib_state_npz，
        obs 会使用 0，pose 会使用单位相对位姿，这只适合测试能否 build，
        不适合作为高质量 INT8 校准。

    --calib_pose_format:
        "auto" 会把 [N, 2, 7] 判断为两帧绝对位姿，把 [N, 7] 判断为
        相对位姿。如果保存的是原始 prev/current 绝对位姿，建议显式传
        --calib_pose_format absolute。

    --calib_cache:
        INT8 校准缓存文件。更换 ONNX、权重、输入格式或校准数据后，应该
        删除旧 cache，避免 TensorRT 复用过期校准表。

    --build_only:
        只从已有 ONNX 构建 engine。拆分模式构建两个 engine，默认不加载
        PyTorch 权重；额外传 --verify_after_build 和 --weight 可在构建后
        立刻与 PyTorch 对比。

    --verify_only:
        加载已有 engine，并和 PyTorch 输出做误差/延迟对比。特征缓存
        模式需同时传 --split_encoder，加载两个 engine 和匹配的 .pth 权重。
"""

import argparse
import glob
import os
import time
import copy

import cv2
import numpy as np
import torch
from torch import nn

from config.config import cfg
from policy.primitive import LatticePrimitive
from policy.yopo_network import YopoNetwork
from policy.trt_runner import TensorRTPolicy, CachedTensorRTPolicy
from policy.engine_metadata import (make_metadata, read_metadata, write_metadata,
                                    file_sha256, planning_output_names)


class YopoONNXWrapper(nn.Module):
    def __init__(self, policy: YopoNetwork):
        super().__init__()
        self.policy = policy

    def forward(self, rgb, obs, pose):
        return self.policy.forward_trt(rgb, obs, pose)


class YopoONNXFP16Wrapper(nn.Module):
    def __init__(self, policy: YopoNetwork):
        super().__init__()
        self.policy = policy.half()

    def forward(self, rgb, obs, pose):
        outputs = self.policy.forward_trt(
            rgb.half(),
            obs.half(),
            pose.half(),
        )
        return tuple(value.float() for value in outputs)


class YopoRGBEncoderONNXWrapper(nn.Module):
    def __init__(self, policy, fp16=False):
        super().__init__()
        self.policy = policy
        self.fp16 = fp16

    def forward(self, rgb):
        features = self.policy.encode_rgb_frame(rgb.half() if self.fp16 else rgb)
        return features.float() if self.fp16 else features


class YopoFeaturePlannerONNXWrapper(nn.Module):
    def __init__(self, policy, fp16=False):
        super().__init__()
        self.policy = policy
        self.fp16 = fp16

    def forward(self, features, obs, pose):
        if self.fp16:
            features, obs, pose = features.half(), obs.half(), pose.half()
        outputs = self.policy.forward_trt_from_features(features, obs, pose)
        return tuple(value.float() for value in outputs) if self.fp16 else outputs


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=31, help="trial number")
    parser.add_argument("--epoch", type=int, default=140, help="epoch number")
    parser.add_argument("--weight", type=str, default="", help="explicit PyTorch checkpoint path")
    parser.add_argument("--training_config", type=str, default="", help="saved ablation_config.json used to construct deployment primitives")
    parser.add_argument("--onnx", type=str, default="yopo.onnx", help="output ONNX path")
    parser.add_argument("--fp16_onnx", type=str, default="", help="output mixed-precision FP16 ONNX path")
    parser.add_argument("--engine", type=str, default="yopo.engine", help="output TensorRT engine path")
    parser.add_argument("--split_encoder", action="store_true", help="export a cached single-frame encoder and feature-input planner")
    parser.add_argument("--encoder_onnx", default="yopo_encoder.onnx", help="single-frame encoder ONNX path")
    parser.add_argument("--planner_onnx", default="yopo_planner.onnx", help="feature-input planner ONNX path")
    parser.add_argument("--encoder_engine", default="yopo_encoder.engine", help="single-frame encoder TensorRT path")
    parser.add_argument("--planner_engine", default="yopo_planner.engine", help="feature-input planner TensorRT path")
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset version")
    parser.add_argument("--fp16", type=int, default=1, help="enable TensorRT fp16")
    parser.add_argument("--int8", type=int, default=0, help="enable TensorRT int8 calibration")
    parser.add_argument("--calib_npz", type=str, default="", help="INT8 calibration npz with rgb, obs, pose arrays")
    parser.add_argument("--calib_image_dir", type=str, default="", help="INT8 calibration image directory; obs/pose use defaults")
    parser.add_argument("--calib_state_npz", type=str, default="", help="optional npz with obs and pose for --calib_image_dir")
    parser.add_argument("--calib_pose_format", type=str, default="auto", choices=["auto", "relative", "absolute"], help="pose format in calibration npz")
    parser.add_argument("--calib_cache", type=str, default="yopo_int8.cache", help="INT8 calibration cache path")
    parser.add_argument("--calib_batches", type=int, default=256, help="maximum INT8 calibration batches")
    parser.add_argument("--workspace", type=int, default=2 << 30, help="TensorRT workspace bytes")
    parser.add_argument("--no_engine", action="store_true", help="only export ONNX")
    parser.add_argument("--build_only", action="store_true", help="build TensorRT engine from an existing ONNX file")
    parser.add_argument("--verify_only", action="store_true", help="compare an existing TensorRT engine with PyTorch")
    parser.add_argument("--verify_after_build", action="store_true", help="run PyTorch-vs-TensorRT check after --build_only")
    parser.add_argument("--benchmark", type=int, default=100, help="latency benchmark iterations")
    return parser


def load_policy(weight, device):
    state_dict = torch.load(weight, map_location="cpu", weights_only=True)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if YopoNetwork.checkpoint_single_frame(state_dict) or YopoNetwork.checkpoint_cnn_early_fusion(state_dict):
        raise ValueError('This exporter targets the proposed two-frame Transformer policy, not single-RGB/CNN ablations')
    if YopoNetwork.checkpoint_temporal_no_pose(state_dict):
        raise ValueError("TensorRT export for temporal-no-pose checkpoints is not supported; use PyTorch inference")
    policy = YopoNetwork(rgb_pretrained=False, single_frame_no_pose=False, cnn_early_fusion=False,
                         temporal_no_pose=False,
                         disable_auxiliary_depth=YopoNetwork.checkpoint_auxiliary_depth_disabled(state_dict),
                         rgb_encoder=YopoNetwork.checkpoint_rgb_encoder(state_dict),
                         score_mode=YopoNetwork.checkpoint_score_mode(state_dict),
                         candidate_topk=YopoNetwork.checkpoint_candidate_topk(state_dict))
    load_result = policy.load_test_state_dict(state_dict)
    if load_result.unexpected_keys:
        print("Unexpected keys when loading test weights:", load_result.unexpected_keys)
    policy.deployment_weight_sha256 = file_sha256(weight)
    return policy.to(device).eval()


def example_inputs(device):
    rgb = torch.zeros(
        (1, cfg["data_timesteps"], 3, cfg["image_height"], cfg["image_width"]),
        dtype=torch.float32,
        device=device,
    )
    obs = torch.zeros((1, 9), dtype=torch.float32, device=device)
    pose = torch.zeros((1, 7), dtype=torch.float32, device=device)
    pose[:, 3] = 1.0
    return rgb, obs, pose


def export_onnx(model, inputs, onnx_path, opset,
                input_names=("rgb", "obs", "pose"), output_names=None,
                metadata=None):
    if output_names is None:
        output_names = planning_output_names(model.policy.score_mode)
    print(f"Exporting ONNX to {onnx_path}...")
    # Disable native MHA during export, restoring all state afterwards so
    # subsequent verification still measures the ordinary inference path.
    mha_backend = getattr(torch.backends, "mha", None)
    fastpath = mha_backend.get_fastpath_enabled() if mha_backend is not None else None
    mha_modules = [(m, m.training) for m in model.modules() if isinstance(m, nn.MultiheadAttention)]
    try:
        if mha_backend is not None:
            mha_backend.set_fastpath_enabled(False)
        for module, _ in mha_modules:
            module.train()  # dropout=0 in this model; avoids native fused MHA op
        torch.onnx.export(
            model, inputs, onnx_path,
            input_names=list(input_names), output_names=list(output_names),
            opset_version=opset, do_constant_folding=True,
            training=torch.onnx.TrainingMode.PRESERVE,
        )
    finally:
        for module, training in mha_modules:
            module.train(training)
        if mha_backend is not None:
            mha_backend.set_fastpath_enabled(fastpath)
    print(f"Saved ONNX: {onnx_path}")
    if metadata is not None:
        import onnx
        onnx.checker.check_model(onnx_path)
        write_metadata(onnx_path, metadata, 'onnx')


def export_fp16_onnx(policy, inputs, onnx_path, opset):
    print(f"Exporting FP16 ONNX to {onnx_path}...")
    fp16_policy = copy.deepcopy(policy)
    fp16_wrapper = YopoONNXFP16Wrapper(fp16_policy).to("cuda").eval()
    export_onnx(fp16_wrapper, inputs, onnx_path, opset,
                metadata=make_metadata(policy, 'full', cfg))
    return onnx_path


def convert_onnx_to_fp16(onnx_path, fp16_onnx_path):
    try:
        import onnx
        import modelopt.onnx.autocast as autocast
    except ImportError as exc:
        print(f"Warning: ModelOpt AutoCast is unavailable ({exc}). Falling back to onnxconverter-common FP16 conversion.")
        try:
            import onnx
            from onnxconverter_common import float16
        except ImportError as fallback_exc:
            raise ImportError(
                "FP16 ONNX conversion requires either NVIDIA ModelOpt or onnxconverter-common. Install fallback with:\n"
                "  pip install onnxconverter-common"
            ) from fallback_exc

        print(f"Converting ONNX to FP16 with onnxconverter-common: {fp16_onnx_path}...")
        model = onnx.load(onnx_path)
        converted_model = float16.convert_float_to_float16(
            model,
            keep_io_types=True,
            op_block_list=["Clip"],
        )
        onnx.save(converted_model, fp16_onnx_path)
        metadata = read_metadata(onnx_path, 'onnx')
        if metadata is not None:
            write_metadata(fp16_onnx_path, metadata, 'onnx')
        print(f"Saved FP16 ONNX: {fp16_onnx_path}")
        return fp16_onnx_path

    print(f"Converting ONNX to mixed-precision FP16: {fp16_onnx_path}...")
    converted_model = autocast.convert_to_mixed_precision(
        onnx_path=onnx_path,
        low_precision_type="fp16",
        keep_io_types=True,
    )
    onnx.save(converted_model, fp16_onnx_path)
    metadata = read_metadata(onnx_path, 'onnx')
    if metadata is not None:
        write_metadata(fp16_onnx_path, metadata, 'onnx')
    print(f"Saved FP16 ONNX: {fp16_onnx_path}")
    return fp16_onnx_path


def _prepare_rgb_array(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim == 4:
        rgb = rgb[None, ...]
    if rgb.shape[-1] == 3:
        rgb = np.transpose(rgb, (0, 1, 4, 2, 3))
    if rgb.max() > 2.0:
        rgb = rgb / 255.0
    return np.ascontiguousarray(rgb)


def _get_npz_array(data, names, required=True):
    for name in names:
        if name in data.files:
            return data[name]
    if required:
        raise ValueError(f"npz must contain one of: {', '.join(names)}")
    return None


def _normalize_calibration_pose(pose, pose_format="auto"):
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim == 1:
        pose = pose[None, :]

    if pose_format == "auto":
        pose_format = "absolute" if pose.ndim == 3 else "relative"

    if pose_format == "absolute":
        if pose.ndim != 3 or pose.shape[1:] != (2, 7):
            raise ValueError(f"absolute calibration pose must have shape [N, 2, 7], got {pose.shape}")
        with torch.no_grad():
            pose = YopoNetwork.absolute_pose_to_relative(torch.from_numpy(pose)).cpu().numpy()
    elif pose_format == "relative":
        if pose.ndim != 2 or pose.shape[1] != 7:
            raise ValueError(f"relative calibration pose must have shape [N, 7], got {pose.shape}")
    else:
        raise ValueError(f"Unknown calibration pose format: {pose_format}")

    return np.ascontiguousarray(pose.astype(np.float32, copy=False))


def _normalize_calibration_obs(obs, count):
    if obs is None:
        return np.zeros((count, 9), dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[None, :]
    if obs.ndim != 2 or obs.shape[1] != 9:
        raise ValueError(f"calibration obs must have shape [N, 9], got {obs.shape}")
    return np.ascontiguousarray(obs)


def load_calibration_samples(calib_npz="", calib_image_dir="", calib_state_npz="", pose_format="auto", max_batches=256):
    samples = []
    if calib_npz:
        data = np.load(calib_npz)
        rgb = _prepare_rgb_array(_get_npz_array(data, ["rgb", "images", "image"]))
        pose = _normalize_calibration_pose(_get_npz_array(data, ["pose", "poses", "absolute_pose", "absolute_poses"]), pose_format)
        obs = _normalize_calibration_obs(_get_npz_array(data, ["obs", "observation", "observations"], required=False), rgb.shape[0])
        count = min(rgb.shape[0], obs.shape[0], pose.shape[0], int(max_batches))
        for i in range(count):
            samples.append({
                "rgb": np.ascontiguousarray(rgb[i:i + 1]),
                "obs": np.ascontiguousarray(obs[i:i + 1]),
                "pose": np.ascontiguousarray(pose[i:i + 1]),
            })
        return samples

    if calib_image_dir:
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        image_paths = []
        for pattern in patterns:
            image_paths.extend(glob.glob(os.path.join(calib_image_dir, pattern)))
        image_paths = sorted(image_paths)
        timesteps = int(cfg["data_timesteps"])
        if len(image_paths) < timesteps:
            raise ValueError(f"calib_image_dir needs at least {timesteps} images: {calib_image_dir}")

        state_obs = None
        state_pose = None
        if calib_state_npz:
            state_data = np.load(calib_state_npz)
            state_pose = _normalize_calibration_pose(
                _get_npz_array(state_data, ["pose", "poses", "absolute_pose", "absolute_poses"]),
                pose_format,
            )
            state_obs = _normalize_calibration_obs(
                _get_npz_array(state_data, ["obs", "observation", "observations"], required=False),
                state_pose.shape[0],
            )

        max_start = min(len(image_paths) - timesteps + 1, int(max_batches))
        if state_pose is not None:
            max_start = min(max_start, state_pose.shape[0], state_obs.shape[0])
        for start in range(max_start):
            frames = []
            for path in image_paths[start:start + timesteps]:
                image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise FileNotFoundError(f"Failed to read calibration image: {path}")
                if image.ndim == 2:
                    image = np.repeat(image[:, :, None], 3, axis=2)
                elif image.shape[2] == 4:
                    image = image[:, :, :3]
                if image.shape[0] != cfg["image_height"] or image.shape[1] != cfg["image_width"]:
                    image = cv2.resize(
                        image,
                        (cfg["image_width"], cfg["image_height"]),
                        interpolation=cv2.INTER_AREA,
                    )
                frames.append(image.astype(np.float32) / 255.0)
            rgb = np.stack(frames, axis=0)[None, ...]
            rgb = _prepare_rgb_array(rgb)
            if state_pose is None:
                obs = np.zeros((1, 9), dtype=np.float32)
                pose = np.zeros((1, 7), dtype=np.float32)
                pose[:, 3] = 1.0
            else:
                obs = state_obs[start:start + 1]
                pose = state_pose[start:start + 1]
            samples.append({"rgb": rgb, "obs": obs, "pose": pose})
        return samples

    raise ValueError("INT8 build requires --calib_npz or --calib_image_dir")


def create_int8_calibrator(trt, samples, cache_path):
    class YopoInt8Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, calibration_samples, calibration_cache):
            super().__init__()
            self.samples = calibration_samples
            self.cache_path = calibration_cache
            self.index = 0
            self.device_tensors = {}

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.index >= len(self.samples):
                return None
            sample = self.samples[self.index]
            self.index += 1
            ptrs = []
            for name in names:
                if name not in sample:
                    raise KeyError(f"Calibration sample is missing input: {name}")
                tensor = torch.as_tensor(sample[name], dtype=torch.float32, device="cuda").contiguous()
                self.device_tensors[name] = tensor
                ptrs.append(int(tensor.data_ptr()))
            return ptrs

        def read_calibration_cache(self):
            if self.cache_path and os.path.exists(self.cache_path):
                with open(self.cache_path, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            if self.cache_path:
                with open(self.cache_path, "wb") as f:
                    f.write(cache)

    return YopoInt8Calibrator(samples, cache_path)


def build_engine(
    onnx_path,
    engine_path,
    fp16=True,
    int8=False,
    workspace=2 << 30,
    fp16_onnx_path="",
    calib_npz="",
    calib_image_dir="",
    calib_state_npz="",
    calib_pose_format="auto",
    calib_cache="yopo_int8.cache",
    calib_batches=256,
):
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ImportError("TensorRT Python package is required to build .engine files.") from exc

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    network_flags = 0 if explicit_batch is None else 1 << int(explicit_batch)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
    int8_flag = getattr(trt.BuilderFlag, "INT8", None)
    build_onnx_path = onnx_path
    if fp16 and fp16_flag is None:
        build_onnx_path = fp16_onnx_path or onnx_path

    with open(build_onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"TensorRT ONNX parse failed:\n{errors}")

    metadata = read_metadata(build_onnx_path, 'onnx')
    actual_inputs = [network.get_input(index).name for index in range(network.num_inputs)]
    actual_outputs = [network.get_output(index).name for index in range(network.num_outputs)]
    if metadata is not None:
        if set(actual_inputs) != set(metadata['input_names']) or set(actual_outputs) != set(metadata['output_names']):
            raise ValueError('ONNX bindings differ from deployment metadata; re-export the checkpoint')
    elif 'safety_score' in actual_outputs:
        raise ValueError('Dual-score ONNX requires its matching .onnx.json file')

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace))
    if fp16:
        if fp16_flag is None:
            print("TensorRT BuilderFlag.FP16 is unavailable; using FP16 ONNX for strong typing.")
        else:
            config.set_flag(fp16_flag)
    if int8:
        if int8_flag is None:
            raise RuntimeError("TensorRT BuilderFlag.INT8 is unavailable in this environment.")
        samples = load_calibration_samples(
            calib_npz=calib_npz,
            calib_image_dir=calib_image_dir,
            calib_state_npz=calib_state_npz,
            pose_format=calib_pose_format,
            max_batches=calib_batches,
        )
        if not samples:
            raise RuntimeError("No INT8 calibration samples were loaded.")
        print(f"Loaded {len(samples)} INT8 calibration batches.")
        config.set_flag(int8_flag)
        config.int8_calibrator = create_int8_calibrator(trt, samples, calib_cache)

    print(f"Building TensorRT engine: {engine_path}...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed.")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    if metadata is not None:
        write_metadata(engine_path, metadata, 'engine')
    print(f"Saved TensorRT engine: {engine_path}")


def compare_planning_outputs(expected, actual, names):
    if len(expected) != len(actual) or len(actual) != len(names):
        raise RuntimeError('TensorRT planning outputs differ from the checkpoint')
    differences = []
    for name, reference, result in zip(names, expected, actual):
        torch.testing.assert_close(result.float(), reference.float(), atol=.05, rtol=.02,
                                   msg=lambda message: name + ': ' + message)
        differences.append((reference.float() - result.float()).abs())
    return differences


def benchmark(wrapper, engine_path, inputs, iterations):
    trt_policy = TensorRTPolicy(engine_path)
    names = planning_output_names(wrapper.policy.score_mode)
    means, maxima = [], []
    with torch.inference_mode():
        for index in range(6):
            rgb, obs, pose = inputs
            current_pose = pose.clone()
            current_pose[:, 4] += .1 * (index + 1)
            current_pose[:, 5] -= .05 * index
            sample = (torch.rand_like(rgb), torch.randn_like(obs) * .2, current_pose)
            differences = compare_planning_outputs(wrapper(*sample), trt_policy(*sample), names)
            means.append([value.mean().item() for value in differences])
            maxima.append([value.max().item() for value in differences])
        print('Transfer mean/max error (6 nonzero inputs): ' + ', '.join(
            f'{name}={mean:.6f}/{maximum:.6f}' for name, mean, maximum in
            zip(names, np.mean(means, axis=0), np.max(maxima, axis=0))))
        if iterations <= 0:
            return
        for _ in range(10):
            torch_outputs = wrapper(*inputs)
            trt_outputs = trt_policy(*inputs)
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(iterations):
            torch_outputs = wrapper(*inputs)
        torch.cuda.synchronize()
        t1 = time.time()

        t2 = time.time()
        for _ in range(iterations):
            trt_outputs = trt_policy(*inputs)
        torch.cuda.synchronize()
        t3 = time.time()

    print(f"Torch Latency: {1000 * (t1 - t0) / iterations:.3f} ms")
    print(f"TensorRT Latency: {1000 * (t3 - t2) / iterations:.3f} ms")


def verify_engine(weight, engine_path, device, iterations):
    print("Loading Network for verification...")
    policy = load_policy(weight, device)
    wrapper = YopoONNXWrapper(policy).to(device).eval()
    inputs = example_inputs(device)
    benchmark(wrapper, engine_path, inputs, iterations)


def _fp16_path(path):
    base, ext = os.path.splitext(path)
    return f"{base}_fp16{ext or '.onnx'}"


def export_split_onnx(policy, inputs, encoder_path, planner_path, opset=16, fp16=False):
    if policy.single_frame_no_pose or inputs[0].shape[1] != 2:
        raise ValueError("Cached split export requires the two-frame temporal RGB model")
    export_policy = copy.deepcopy(policy).half().eval() if fp16 else policy
    encoder = YopoRGBEncoderONNXWrapper(export_policy, fp16=fp16).eval()
    planner = YopoFeaturePlannerONNXWrapper(export_policy, fp16=fp16).eval()
    rgb, obs, pose = inputs
    with torch.inference_mode():
        frame_features = encoder(rgb[:, 0].contiguous())
        features = torch.stack((frame_features, frame_features), dim=1)
    export_onnx(encoder, (rgb[:, 0].contiguous(),), encoder_path, opset,
                input_names=("rgb",), output_names=("features",),
                metadata=make_metadata(policy, 'encoder', cfg) if hasattr(policy, 'deployment_weight_sha256') else None)
    export_onnx(planner, (features, obs, pose), planner_path, opset,
                input_names=("features", "obs", "pose"),
                metadata=make_metadata(policy, 'planner', cfg) if hasattr(policy, 'deployment_weight_sha256') else None)


def benchmark_split(policy, encoder_engine, planner_engine, device, iterations):
    """Check moving frame pairs and time a stream with one new frame per call."""
    trt_policy = CachedTensorRTPolicy(encoder_engine, planner_engine, device=device)
    names = planning_output_names(policy.score_mode)
    rgb, obs, pose = example_inputs(device)
    # Nonzero images and changing poses catch stale features and frame-order
    # mistakes that an all-zero input would hide.
    generator = torch.Generator(device=device).manual_seed(42)
    frames = torch.rand((3, *rgb.shape[2:]), generator=generator, device=device)
    pairs = [torch.stack((frames[i], frames[(i + 1) % 3]), dim=0)[None] for i in range(3)]
    obs = torch.randn(obs.shape, generator=generator, device=device) * 0.2
    means, maxima = [], []
    with torch.inference_mode():
        for i in range(6):
            current_pose = pose.clone()
            current_pose[:, 4] = 0.1 * (i + 1)
            current_pose[:, 5] = -0.05 * i
            expected = policy.forward_trt(pairs[i % 3], obs, current_pose)
            actual = trt_policy(pairs[i % 3], obs, current_pose, frame_ids=(i, i + 1))
            torch.cuda.synchronize()
            differences = compare_planning_outputs(expected, actual, names)
            means.append([d.mean().item() for d in differences])
            maxima.append([d.max().item() for d in differences])
        print('Transfer mean/max error (6 moving pairs): ' + ', '.join(
            f'{name}={mean:.6f}/{maximum:.6f}' for name, mean, maximum in
            zip(names, np.mean(means, axis=0), np.max(maxima, axis=0))))
        if iterations <= 0:
            return
        trt_policy.clear_cache()
        for i in range(10):
            policy.forward_trt(pairs[i % 3], obs, pose)
            trt_policy(pairs[i % 3], obs, pose, frame_ids=(i, i + 1))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(iterations):
            policy.forward_trt(pairs[(i + 10) % 3], obs, pose)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        misses_before = trt_policy.feature_cache.misses
        t2 = time.perf_counter()
        for i in range(10, 10 + iterations):
            trt_policy(pairs[i % 3], obs, pose, frame_ids=(i, i + 1))
        torch.cuda.synchronize()
        t3 = time.perf_counter()
    fresh_frames = trt_policy.feature_cache.misses - misses_before
    if fresh_frames != iterations:
        raise RuntimeError(f"Streaming benchmark encoded {fresh_frames} frames, expected {iterations}")
    print(f"Torch full two-frame latency: {1000 * (t1 - t0) / iterations:.3f} ms")
    print(f"Cached TensorRT latency (one new frame): {1000 * (t3 - t2) / iterations:.3f} ms")
    print(f"Cache benchmark: {fresh_frames} new encodings / {iterations} planning calls")


def run_split(args, weight, device):
    if args.int8:
        raise ValueError("Split-engine calibration is not implemented; use --int8 0 with FP16/FP32")
    if args.verify_only:
        policy = load_policy(weight, device)
        benchmark_split(policy, args.encoder_engine, args.planner_engine, device, args.benchmark)
        return
    paths = ((args.encoder_onnx, args.encoder_engine), (args.planner_onnx, args.planner_engine))
    policy = None
    if not args.build_only:
        print("Loading Network for split export...")
        policy = load_policy(weight, device)
        inputs = example_inputs(device)
        export_split_onnx(policy, inputs, args.encoder_onnx, args.planner_onnx, args.opset)
    if args.no_engine:
        return
    import tensorrt as trt
    strong_fp16 = bool(args.fp16) and getattr(trt.BuilderFlag, "FP16", None) is None
    if strong_fp16 and (not args.build_only or os.path.isfile(weight)):
        # Export from the half-precision Torch model, including geometry
        # constants/casts. Generic FP16 conversion can mix Float constants and
        # Half Gemm weights in the planning adapter on strongly typed TRT 11.
        if policy is None:
            policy = load_policy(weight, device)
            inputs = example_inputs(device)
        export_split_onnx(policy, inputs, _fp16_path(args.encoder_onnx),
                          _fp16_path(args.planner_onnx), args.opset, fp16=True)
    for onnx_path, engine_path in paths:
        fp16_path = _fp16_path(onnx_path) if strong_fp16 else ""
        if args.build_only and strong_fp16 and not os.path.exists(fp16_path):
            convert_onnx_to_fp16(onnx_path, fp16_path)
        build_engine(onnx_path, engine_path, fp16=bool(args.fp16),
                     workspace=args.workspace, fp16_onnx_path=fp16_path)
    if not args.build_only or args.verify_after_build:
        if policy is None:
            policy = load_policy(weight, device)
        benchmark_split(policy.eval(), args.encoder_engine, args.planner_engine, device, args.benchmark)


if __name__ == "__main__":
    args = parser().parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weight = args.weight or os.path.join(base_dir, "saved", f"YOPO_{args.trial}", f"epoch{args.epoch}.pth")
    if args.training_config:
        import json
        with open(args.training_config) as stream:
            cfg._data.update(json.load(stream))
        cfg['goal_length'] = 2.0 * cfg['radio_range']
        cfg['sgm_time'] = 2.0 * cfg['radio_range'] / cfg['vel_max_train']
        cfg['traj_num'] = cfg['horizon_num'] * cfg['vertical_num'] * cfg['radio_num']
    cuda_available = torch.cuda.is_available()
    if not cuda_available and (not args.no_engine or args.build_only or args.verify_only):
        raise RuntimeError("TensorRT build and verification require CUDA; use --no_engine for CPU ONNX export.")

    cfg["train"] = False
    LatticePrimitive._instance = None
    device = "cuda" if cuda_available else "cpu"

    if args.split_encoder:
        run_split(args, weight, device)
        raise SystemExit(0)

    if args.verify_only:
        verify_engine(weight, args.engine, device, args.benchmark)
        raise SystemExit(0)

    if args.build_only:
        build_engine(
            args.onnx,
            args.engine,
            fp16=bool(args.fp16),
            int8=bool(args.int8),
            workspace=args.workspace,
            fp16_onnx_path=args.fp16_onnx,
            calib_npz=args.calib_npz,
            calib_image_dir=args.calib_image_dir,
            calib_state_npz=args.calib_state_npz,
            calib_pose_format=args.calib_pose_format,
            calib_cache=args.calib_cache,
            calib_batches=args.calib_batches,
        )
        if args.verify_after_build:
            verify_engine(weight, args.engine, device, args.benchmark)
        raise SystemExit(0)

    print("Loading Network...")
    policy = load_policy(weight, device)
    wrapper = YopoONNXWrapper(policy).to(device).eval()
    inputs = example_inputs(device)

    export_onnx(wrapper, inputs, args.onnx, args.opset,
                metadata=make_metadata(policy, 'full', cfg))
    if not args.no_engine:
        engine_onnx = args.onnx
        fp16_flag_available = False
        if bool(args.fp16):
            try:
                import tensorrt as trt
                fp16_flag_available = getattr(trt.BuilderFlag, "FP16", None) is not None
            except ImportError:
                fp16_flag_available = False
            if not fp16_flag_available:
                fp16_onnx_path = args.fp16_onnx
                if not fp16_onnx_path:
                    base, ext = os.path.splitext(args.onnx)
                    fp16_onnx_path = f"{base}_fp16{ext or '.onnx'}"
                engine_onnx = export_fp16_onnx(policy, inputs, fp16_onnx_path, args.opset)
                args.fp16_onnx = engine_onnx
        build_engine(
            engine_onnx,
            args.engine,
            fp16=bool(args.fp16),
            int8=bool(args.int8),
            workspace=args.workspace,
            fp16_onnx_path=args.fp16_onnx,
            calib_npz=args.calib_npz,
            calib_image_dir=args.calib_image_dir,
            calib_state_npz=args.calib_state_npz,
            calib_pose_format=args.calib_pose_format,
            calib_cache=args.calib_cache,
            calib_batches=args.calib_batches,
        )
        benchmark(wrapper, args.engine, inputs, args.benchmark)
