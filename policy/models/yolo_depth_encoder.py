"""Complete YOLODepth with adapters, followed by a small depth feature extractor.

The pretrained decoder produces calibrated native P2 depth. Narrow convolutions
connect the original depth output to the existing pose/temporal
fusion; auxiliary depth supervision remains after that fusion.
"""

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .resnet import BasicBlock


DEFAULT_ADAPTER_LAYERS = (4, 6, 8, 10, 13, 16, 19, 22)


class ResidualConvAdapter(nn.Module):
    """Adapt a feature map without changing its initial pretrained output."""

    def __init__(self, channels, bottleneck=32, residual_scale=1.0):
        super().__init__()
        if bottleneck <= 0:
            raise ValueError("Adapter bottleneck must be positive")
        residual_scale = float(residual_scale)
        if not math.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError("Adapter residual_scale must be finite and nonnegative")
        # Save the fixed scale with the weights so inference/export restore the
        # same function even if the YAML default changes later.
        self.register_buffer("residual_scale", torch.tensor(residual_scale))
        self.down = nn.Conv2d(channels, bottleneck, 1, bias=False)
        self.spatial = nn.Conv2d(
            bottleneck, bottleneck, 3, padding=1, groups=bottleneck, bias=False
        )
        self.up = nn.Conv2d(bottleneck, channels, 1, bias=False)
        self.activation = nn.SiLU()
        nn.init.zeros_(self.up.weight)

    def forward(self, feature):
        correction = self.activation(self.down(feature))
        correction = self.activation(self.spatial(correction))
        return feature + self.residual_scale * self.up(correction)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        scale_key = prefix + "residual_scale"
        if scale_key not in state_dict:
            # Checkpoints predating this field used x + delta, i.e. scale 1.
            state_dict[scale_key] = torch.ones_like(self.residual_scale)
        value = state_dict[scale_key]
        if value.numel() != 1 or not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            error_msgs.append(f"Invalid adapter residual scale in checkpoint: {scale_key}")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )


class DepthFeatureExtractor(nn.Module):
    """Learn progressive depth downsampling before the final channel projection."""

    def __init__(self, output_dim=128, channels=32, output_stride=8):
        super().__init__()
        if channels <= 0 or output_dim <= 0:
            raise ValueError("Depth feature channel counts must be positive")
        if output_stride not in (8, 16):
            raise ValueError("Depth feature output_stride must be 8 or 16")
        stem_channels = max(1, channels // 2)
        # Lift depth into channels at native resolution before discarding any
        # spatial samples. Stem width is half the first residual-stage width.
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(),
        )
        shortcut = nn.Sequential(
            nn.Conv2d(stem_channels, channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(channels),
        )
        # First residual stage: native P2 -> P3 (72x128 -> 36x64).
        self.refine = BasicBlock(
            stem_channels, channels, stride=2, downsample=shortcut
        )
        if output_stride == 16:
            coarse_channels = 2 * channels
            coarse_shortcut = nn.Sequential(
                nn.Conv2d(channels, coarse_channels, 1, stride=2, bias=False),
                nn.BatchNorm2d(coarse_channels),
            )
            # Second residual stage learns the P3 -> P4 reduction at low width.
            # Delay the 256-channel projection until the smaller 18x32 grid.
            self.coarse_refine = BasicBlock(
                channels, coarse_channels, stride=2, downsample=coarse_shortcut
            )
        else:
            # Preserve the original stride-8 extractor/checkpoint layout.
            coarse_channels = channels
            self.coarse_refine = nn.Identity()
        self.projection = nn.Conv2d(coarse_channels, output_dim, 1, bias=False)

    def forward(self, depth):
        feature = self.stem(depth)
        feature = self.refine(feature)
        feature = self.coarse_refine(feature)
        return self.projection(feature)


class YoloDepthEncoder(nn.Module):
    def __init__(self, output_dim=128, weights="yolo26m-depth.pt",
                 architecture="yolo26m-depth.yaml", input_size=(288, 512),
                 freeze_backbone=True, freeze_neck=True, pretrained=True,
                 adapter_enabled=True, adapter_bottleneck=32,
                 adapter_layers=DEFAULT_ADAPTER_LAYERS, freeze_depth_head=True,
                 depth_feature_channels=32, depth_normalization_m=20.0, output_stride=8, adapter_scale=1.0):
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YOLO26-Depth requires a depth-capable Ultralytics release. "
                "Install YOPO/requirements-yolo-depth.txt in your training environment."
            ) from exc

        self.output_stride = int(output_stride)
        if self.output_stride not in (8, 16):
            raise ValueError("YOLO depth feature output_stride must be 8 or 16")
        self.input_size = tuple(int(value) for value in input_size)
        if len(self.input_size) != 2 or any(
                value <= 0 or value % 32 for value in self.input_size):
            raise ValueError("YOLO input height/width must be positive multiples of 32")
        if not math.isfinite(depth_normalization_m) or depth_normalization_m <= 0:
            raise ValueError("Depth normalization in meters must be finite and positive")
        # Resolve project-relative weights independently of the caller's cwd.
        weights_path = Path(weights).expanduser()
        project_path = Path(__file__).resolve().parents[2] / weights_path
        if not weights_path.is_absolute() and project_path.is_file():
            weights_path = project_path
        source = str(weights_path) if pretrained else str(architecture)
        if pretrained and weights_path.suffix != ".pt":
            raise ValueError("Pretrained YOLO depth weights must be a .pt checkpoint")
        model = YOLO(source, task="depth").model.float()
        layers = list(model.model)
        depth_head = layers[-1]
        if type(depth_head).__name__ != "Depth" or not all(
                hasattr(depth_head, name) for name in ("proj", "f", "cal_a", "cal_b")):
            raise ValueError(
                "Expected YOLO26-Depth weights with a multi-scale Depth head; "
                "detection weights and incompatible depth releases are unsupported."
            )
        backbone_count = len(model.yaml["backbone"])
        self.feature_indices = tuple(depth_head.f)
        if len(self.feature_indices) != 3 or any(
                index < 0 or index >= len(layers) - 1
                for index in self.feature_indices):
            raise ValueError("Expected three P3/P4/P5 feature inputs before the depth head")
        self.backbone = nn.ModuleList(layers[:backbone_count])
        self.neck = nn.ModuleList(layers[backbone_count:-1])
        self.depth_head = depth_head
        # Ultralytics' export flag adds a final 4x resize. Our ONNX encoder uses
        # native P2 depth and exports the small extractor together with the head.
        self.depth_head.export = False
        self.adapters = nn.ModuleDict()
        if adapter_enabled:
            indices = tuple(int(index) for index in adapter_layers)
            if len(indices) != len(set(indices)):
                raise ValueError("YOLO adapter layer indices must be unique")
            for index in indices:
                if index < 0 or index >= len(layers) - 1:
                    raise ValueError(f"Invalid YOLO adapter layer index: {index}")
                layer = layers[index]
                if type(layer).__name__ not in ("C3k2", "C2PSA"):
                    raise ValueError(
                        f"YOLO adapter layer {index} must be C3k2 or C2PSA, "
                        f"got {type(layer).__name__}"
                    )
                self.adapters[str(index)] = ResidualConvAdapter(
                    layer.cv2.conv.out_channels, int(adapter_bottleneck),
                    residual_scale=adapter_scale,
                )
        self.depth_features = DepthFeatureExtractor(
            output_dim=int(output_dim), channels=int(depth_feature_channels),
            output_stride=self.output_stride,
        )
        # Retain this unused buffer for compatibility with existing checkpoints.
        # Native depth is passed to the feature extractor without normalization.
        self.register_buffer(
            "depth_normalization_m", torch.tensor(float(depth_normalization_m))
        )
        self.backbone_frozen = bool(freeze_backbone)
        self.neck_frozen = bool(freeze_neck)
        self.depth_head_frozen = bool(freeze_depth_head)
        self.requires_grad_(True)
        self._apply_backbone_freeze()

    def _apply_backbone_freeze(self):
        for module, frozen in ((self.backbone, self.backbone_frozen),
                               (self.neck, self.neck_frozen),
                               (self.depth_head, self.depth_head_frozen)):
            if frozen:
                module.eval()
                module.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        # Parent/stage train() must not change pretrained BatchNorm statistics.
        self._apply_backbone_freeze()
        return self

    def _run_layer(self, layer, current, outputs):
        source = layer.f
        if source != -1:
            current = (
                outputs[source] if isinstance(source, int)
                else [current if index == -1 else outputs[index] for index in source]
            )
        current = layer(current)
        adapter_key = str(layer.i)
        if adapter_key in self.adapters:
            current = self.adapters[adapter_key](current)
        # All later sequential and skip/concat routes use the adapted output.
        outputs.append(current)
        return current

    def forward_native_depth(self, images):
        """Return calibrated depth in meters at the head's native H/4 x W/4."""
        # Dataset/ROS inputs are BGR floats in [0,1]; YOLO expects RGB.
        current = F.interpolate(
            images[:, [2, 1, 0]], size=self.input_size,
            mode="bilinear", align_corners=False,
        )
        outputs = []
        # Frozen weights naturally avoid building a graph before the first
        # trainable adapter. Later frozen layers must propagate adapter gradients.
        for layer in self.backbone:
            current = self._run_layer(layer, current, outputs)
        for layer in self.neck:
            current = self._run_layer(layer, current, outputs)

        # No no_grad()/detach(): the frozen decoder must pass gradients back to
        # the adapters. Its original calibration buffers are retained as well.
        depth = self.depth_head([outputs[index] for index in self.feature_indices])
        if isinstance(depth, dict):
            # An optionally unfrozen head returns uncalibrated depth in train
            # mode. Apply the same calibration used by its eval forward path.
            depth = depth["depth"].pow(self.depth_head.cal_a) * self.depth_head.cal_b.exp()
        return depth

    def forward(self, images, target_depth=None, target_indices=None):
        # Align the lightweight depth features to the requested effective token
        # stride with only a small final resize (18x32 -> 17x30 by default).
        # Keep the pretrained YOLO decoder and calibration unchanged.
        stride = self.output_stride
        output_size = (
            (images.shape[-2] + stride - 1) // stride,
            (images.shape[-1] + stride - 1) // stride,
        )
        depth = self.forward_native_depth(images)
        if target_depth is not None:
            # Both modalities use metric depth and exactly the same extractor.
            # Concatenate before BatchNorm so its statistics see both domains.
            if target_depth.ndim != 4 or target_depth.shape[1] != 1:
                raise ValueError('Expected metric target depth [B,1,H,W]')
            if target_indices is None or target_indices.numel() != target_depth.shape[0]:
                raise ValueError('Provide one current-frame index per target depth')
            target = F.interpolate(target_depth, size=depth.shape[-2:], mode='nearest')
            all_features = self.depth_features(torch.cat((depth, target), dim=0))
            all_features = F.interpolate(all_features, size=output_size, mode='bilinear', align_corners=False)
            features, target_features = all_features.split((depth.shape[0], target.shape[0]))
            return features, features[target_indices], target_features
        features = self.depth_features(depth)
        return F.interpolate(
            features, size=output_size, mode="bilinear", align_corners=False
        )
