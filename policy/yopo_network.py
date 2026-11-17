"""
YOPO Network
forward, prediction, pre-processing, post-processing
"""

import copy
import warnings

import torch
from torch import nn
import numpy as np
from config.config import cfg
from policy.models.network_hub import YopoBackbone ,CustomTransformerDecoderLayer,CustomTransformerDecoder,PrimitiveEncoder, ResNet_depth , ResNet_rgb , PositionalEncoding ,CustomTransformerEncoderLayer,CustomTransformerEncoder, TrajectoryAlignedPlanningAdapter, TwoScaleDPTDepthHead
from policy.models.head import YopoHead
from policy.models.yolo_depth_encoder import YoloDepthEncoder
from policy.models.early_fusion_cnn import EarlyFusionCNN
from policy.state_transform import *


class YopoNetwork(nn.Module):
    TEST_STATE_PREFIXES = (
        "rgb_architecture",
        "rgb_use_relative_pose",
        "rgb_auxiliary_depth_enabled",
        "planning_score_mode",
        "planning_candidate_topk",
        "rgb_backbone.",
        "visual_rgb_backbone.",
        "pose_embed.",
        "positional_encoding.",
        "rgb_positional_encoding.",
        "frame_encoder.",
        "fusion_encoder.",
        "fusion_head.",
        "cnn_fusion.",
        "r2d_plan_conv.",
        "state_backbone.",
        "planning_adapter.",
        "plan_header.",
        "traj_header.",
    )
    DEPTH_EXPERT_STATE_PREFIXES = (
        "planning_score_mode",
        "planning_candidate_topk",
        "depth_backbone.",
        "depth_plan_conv.",
        "positional_encoding.",
        "state_backbone.",
        "planning_adapter.",
        "plan_header.",
        "traj_header.",
    )

    def __init__(
            self,
            observation_dim=9,  # 9: v_xyz, a_xyz, goal_xyz
            output_dim=None,  # 11 for topk-collision; 10 for single-score modes
            hidden_state=128,
            rgb_pretrained=True,
            temporal_no_pose=None,
            disable_auxiliary_depth=None,
            score_mode=None,
            candidate_topk=None,
            single_frame_no_pose=None,
            rgb_encoder=None,
            cnn_early_fusion=None,
    ):
        super(YopoNetwork, self).__init__()
        self.score_mode = str(cfg._data.get('score_mode', 'selection') if score_mode is None else score_mode)
        modes = ('selection', 'collision', 'topk-collision')
        if self.score_mode not in modes:
            raise ValueError('Unknown score_mode')
        expected_dim = 11 if self.score_mode == 'topk-collision' else 10
        if output_dim is not None and output_dim != expected_dim:
            raise ValueError('output_dim does not match score_mode')
        self.output_dim = expected_dim
        self.candidate_topk = int(cfg._data.get('selection_candidate_topk', 8) if candidate_topk is None else candidate_topk)
        if not 1 <= self.candidate_topk <= int(cfg['traj_num']):
            raise ValueError('candidate_topk must be between 1 and traj_num')
        self.register_buffer('planning_score_mode', torch.tensor(modes.index(self.score_mode), dtype=torch.int64))
        self.register_buffer('planning_candidate_topk', torch.tensor(self.candidate_topk, dtype=torch.int64))
        self.hidden_state = hidden_state
        self.observation_dim = observation_dim
        self.state_transform = StateTransform()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.traj_num = cfg['traj_num']
        if int(cfg["radio_num"]) != 1:
            raise ValueError("Trajectory-aligned visual attention currently requires radio_num=1")
        self.single_frame_no_pose = bool(cfg._data.get("single_frame_no_pose", False) if single_frame_no_pose is None else single_frame_no_pose)
        self.temporal_no_pose = bool(
            cfg._data.get("temporal_no_pose", False) if temporal_no_pose is None else temporal_no_pose
        )
        self.cnn_early_fusion = bool(
            cfg._data.get('cnn_early_fusion', False) if cnn_early_fusion is None else cnn_early_fusion
        )
        if self.cnn_early_fusion and (self.single_frame_no_pose or self.temporal_no_pose):
            raise ValueError('cnn-early-fusion requires two frames and relative-pose channels')
        if self.temporal_no_pose and self.single_frame_no_pose:
            raise ValueError("temporal_no_pose requires two-frame input")
        self.auxiliary_depth_enabled = not bool(
            cfg._data.get("disable_auxiliary_depth", False)
            if disable_auxiliary_depth is None else disable_auxiliary_depth
        )
        self.register_buffer("rgb_auxiliary_depth_enabled", torch.tensor(
            int(not self.single_frame_no_pose and self.auxiliary_depth_enabled), dtype=torch.int64
        ))
        self.register_buffer("rgb_use_relative_pose", torch.tensor(
            int(not self.single_frame_no_pose and not self.temporal_no_pose), dtype=torch.int64
        ))
        self.single_rgb_feature_distill = bool(
            cfg._data.get("single_rgb_feature_distill", False)
        )
        self.pure_single_rgb = (
            self.single_frame_no_pose and not self.single_rgb_feature_distill
        )
        self.use_depth_expert = not self.pure_single_rgb
        if self.single_frame_no_pose:
            training_mode = (
                "feature distillation" if self.single_rgb_feature_distill
                else "20 contrastive + 20 frozen-encoder planning epochs" if cfg._data.get('single_rgb_contrastive', False)
                else "end-to-end planning"
            )
            print(f"Ablation enabled: single-frame RGB ({training_mode})")
        if self.single_frame_no_pose and self.single_rgb_feature_distill:
            if float(cfg['wfeature']) <= 0.0:
                raise ValueError(
                    "single_rgb_feature_distill=true requires wfeature > 0"
                )


        #深度特征提取分支
        if self.use_depth_expert:
            self.depth_backbone = ResNet_depth(hidden_state)
            # Preserve the native planning grid (9x15 for a 270x480 input).
            self.depth_plan_conv = nn.Sequential(
                nn.Conv2d(
                    hidden_state, hidden_state, kernel_size=3,
                    stride=1, padding=1, bias=False
                ),
            )
        else:
            self.depth_backbone = None
            self.depth_plan_conv = None


        #RGB特征提取分支
        self.rgb_encoder_type = str(cfg._data.get("rgb_encoder", "resnet") if rgb_encoder is None else rgb_encoder)
        self.rgb_visual_fusion = (
            bool(cfg._data.get("rgb_visual_fusion", False))
            and self.rgb_encoder_type == "yolo26_depth" and not self.single_frame_no_pose
        )
        # Keep teacher/planner width independent of the wider RGB representation.
        # Selecting resnet reconstructs the original checkpoint architecture.
        self.rgb_feature_channels = (
            int(cfg._data.get("rgb_feature_channels", hidden_state))
            if self.rgb_encoder_type == "yolo26_depth" else hidden_state
        )
        if self.rgb_feature_channels <= 0 or self.rgb_feature_channels % 4:
            raise ValueError("rgb_feature_channels must be a positive multiple of 4")
        self.rgb_feature_stride = (
            int(cfg._data.get("rgb_feature_stride", 8))
            if self.rgb_encoder_type == "yolo26_depth" else 8
        )
        if self.rgb_feature_stride not in (8, 16):
            raise ValueError("rgb_feature_stride must be 8 or 16")
        self.rgb_frame_layers = (
            int(cfg._data.get("rgb_frame_layers", 2))
            if self.rgb_encoder_type == "yolo26_depth" else 2
        )
        self.rgb_fusion_layers = (
            int(cfg._data.get("rgb_fusion_layers", 2))
            if self.rgb_encoder_type == "yolo26_depth" else 2
        )
        if min(self.rgb_frame_layers, self.rgb_fusion_layers) <= 0:
            raise ValueError("RGB Transformer layer counts must be positive")
        # Stride is not encoded in convolution weight shapes. Persist it with
        # layer counts so resume/export cannot silently use a different layout.
        self.register_buffer("rgb_architecture", torch.tensor([
            self.rgb_feature_stride,
            0 if self.single_frame_no_pose or self.cnn_early_fusion else self.rgb_frame_layers,
            0 if self.single_frame_no_pose or self.cnn_early_fusion else self.rgb_fusion_layers,
        ], dtype=torch.int64))
        if self.rgb_encoder_type == "yolo26_depth":
            self.rgb_backbone = YoloDepthEncoder(
                output_dim=self.rgb_feature_channels,
                output_stride=self.rgb_feature_stride,
                weights=str(cfg._data.get("yolo_depth_weights", "yolo26m-depth.pt")),
                architecture=str(cfg._data.get("yolo_depth_architecture", "yolo26m-depth.yaml")),
                input_size=(
                    int(cfg._data.get("yolo_depth_input_height", 288)),
                    int(cfg._data.get("yolo_depth_input_width", 512)),
                ),
                freeze_backbone=bool(cfg._data.get("yolo_depth_freeze_backbone", True)),
                freeze_neck=bool(cfg._data.get("yolo_depth_freeze_neck", True)),
                freeze_depth_head=bool(cfg._data.get("yolo_depth_freeze_head", True)),
                depth_feature_channels=int(cfg._data.get("yolo_depth_feature_channels", 32)),
                pretrained=rgb_pretrained,
                adapter_enabled=bool(cfg._data.get("yolo_depth_adapter_enabled", True)),
                adapter_bottleneck=int(cfg._data.get("yolo_depth_adapter_bottleneck", 32)),
                adapter_scale=float(cfg._data.get("yolo_depth_adapter_scale", 1.0)),
                adapter_layers=cfg._data.get(
                    "yolo_depth_adapter_layers", (4, 6, 8, 10, 13, 16, 19, 22)
                ),
            )
        elif self.rgb_encoder_type == "resnet":
            self.rgb_backbone = ResNet_rgb(hidden_state)
        else:
            raise ValueError(f"Unknown rgb_encoder: {self.rgb_encoder_type}")
        self.single_rgb_contrastive = bool(
            self.pure_single_rgb and cfg._data.get('single_rgb_contrastive', False)
        )
        self.depth_feature_contrastive = bool(cfg._data.get('depth_feature_contrastive', False))
        self.depth_contrastive_projection = None
        if self.depth_feature_contrastive:
            if self.rgb_encoder_type != 'yolo26_depth' or self.single_frame_no_pose:
                raise ValueError('Depth-feature contrastive supervision requires temporal YOLODepth')
            self.depth_contrastive_projection = nn.Sequential(
                nn.Linear(self.rgb_feature_channels, self.rgb_feature_channels),
                nn.ReLU(), nn.Linear(self.rgb_feature_channels, 128),
            )
        self.contrastive_projection = None
        if self.single_rgb_contrastive:
            if self.rgb_encoder_type != 'resnet':
                raise ValueError('single-rgb-contrastive requires the existing single-RGB ResNet encoder')
            # Training-only head; the planner keeps using the spatial feature map.
            self.contrastive_projection = nn.Sequential(
                nn.Linear(self.rgb_feature_channels, self.rgb_feature_channels),
                nn.ReLU(),
                nn.Linear(self.rgb_feature_channels, 128),
            )
            self.register_buffer('contrastive_schedule', torch.tensor([
                cfg['contrastive_pretrain_num'], cfg['pure_rgb_train_num']
            ], dtype=torch.int64))
        self.positional_encoding = PositionalEncoding(hidden_state)
        # The original buffer belongs to existing depth-expert checkpoints.
        # Preserve it at planner width and give wider RGB tokens their own buffer.
        self.rgb_positional_encoding = (
            PositionalEncoding(self.rgb_feature_channels)
            if self.rgb_feature_channels != hidden_state else None
        )
        #transformer的位置编码（代替cnn）
        self.cnn_fusion = EarlyFusionCNN(self.rgb_feature_channels) if self.cnn_early_fusion else None
        if self.single_frame_no_pose or self.cnn_early_fusion:
            # Single-frame RGB and CNN fusion do not construct frame/cross-frame
            # Transformers. CNN receives relative pose as seven input channels.
            self.pose_embed = None
            self.frame_encoder = None
            self.fusion_encoder = None
            self.fusion_head = None
        else:
            # 四元数(4) + 平移(3) = 7维输入，映射到d_model维度
            self.pose_embed = None if self.temporal_no_pose else nn.Linear(7, self.rgb_feature_channels)
            frame_encoder_layer = CustomTransformerEncoderLayer(
                self.rgb_feature_channels, 4, 2*self.rgb_feature_channels
            )
            self.frame_encoder = CustomTransformerEncoder(
                frame_encoder_layer, num_layers=self.rgb_frame_layers
            )
            fusion_encoder_layer = CustomTransformerEncoderLayer(
                2*self.rgb_feature_channels, 4, 2*self.rgb_feature_channels
            )
            self.fusion_encoder = CustomTransformerEncoder(
                fusion_encoder_layer, num_layers=self.rgb_fusion_layers
            )
            # Keep the full RGB width throughout temporal fusion.
            self.fusion_head = nn.Conv1d(
                in_channels=2*self.rgb_feature_channels,
                out_channels=self.rgb_feature_channels,
                kernel_size=1,
            )
        # Preserve ceil(input / 32) and the existing depth expert interface.
        second_plan_stride = 32 // (2 * self.rgb_feature_stride)
        self.visual_rgb_backbone = ResNet_rgb(self.rgb_feature_channels) if self.rgb_visual_fusion else None
        self.r2d_plan_conv = nn.Sequential(
            nn.Conv2d(self.rgb_feature_channels * (2 if self.rgb_visual_fusion else 1), self.rgb_feature_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.rgb_feature_channels, self.rgb_feature_channels, kernel_size=3, stride=second_plan_stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.rgb_feature_channels, hidden_state, kernel_size=3, stride=1, padding=1, bias=False),
        )

        # With visual fusion, derive both auxiliary scales from geometry alone.
        if self.single_frame_no_pose or not self.auxiliary_depth_enabled:
            self.auxiliary_plan_conv = None
            self.auxiliary_depth_head = None
        else:
            self.auxiliary_plan_conv = nn.Sequential(
                nn.Conv2d(self.rgb_feature_channels, self.rgb_feature_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.rgb_feature_channels, self.rgb_feature_channels, kernel_size=3, stride=second_plan_stride, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.rgb_feature_channels, hidden_state, kernel_size=3, stride=1, padding=1, bias=False),
            ) if self.rgb_visual_fusion else None
            self.auxiliary_depth_head = TwoScaleDPTDepthHead(
                fine_channels=self.rgb_feature_channels,
                planning_channels=self.hidden_state,
                features=int(cfg._data.get("r2d_dpt_features", 32)),
            )
        self.state_backbone = PrimitiveEncoder(self.hidden_state, self.traj_num, primitive_dim=observation_dim, state_dim=observation_dim)
        self.planning_adapter = TrajectoryAlignedPlanningAdapter(
            d_model=self.hidden_state,
            nhead=4,
            vertical_num=cfg["vertical_num"],
            horizon_num=cfg["horizon_num"],
            vertical_fov_deg=cfg["vertical_camera_fov"],
            horizon_fov_deg=cfg["horizon_camera_fov"],
            local_window=int(cfg._data.get("planning_local_window", 5)),
            global_size=(
                int(cfg._data.get("planning_global_height", 2)),
                int(cfg._data.get("planning_global_width", 3)),
            ),
        )


        #共用规划头
        plan_decoder_layer = CustomTransformerDecoderLayer(self.hidden_state, 4, 2*self.hidden_state)
        self.plan_header = CustomTransformerDecoder(plan_decoder_layer,num_layers=2)
        self.traj_header = nn.Sequential(nn.Linear(hidden_state, self.output_dim))

    def forward(self,depth: torch.Tensor = None, rgb:torch.Tensor = None, obs:torch.Tensor = None, pose:torch.Tensor = None , mode="test"):
        """
        Stage 1: Depth Teacher Pretraining
        We first train a depth-based planner using ground-truth depth.

        Stage 2: RGB Perception Distillation
        Freeze the depth expert and planning modules; train perception using
        feature, auxiliary depth and clearance supervision. The YOLO backbone
        and neck and its original depth decoder stay frozen by default, while
        gradients pass through them to the adapters. The small depth feature
        extractor and temporal/auxiliary modules use the configured RGB width;
        the shared depth expert and planner remain 128-wide.

        Stage 3: RGB Planner Fine-tuning
        Freeze perception and fine-tune planning without auxiliary depth loss.
        """
        if mode == "depth_pretrain":
            return self.forward_depth_pretrain(depth, obs)
        if mode == "rgb_percepetion_distill":#这个阶段会固定depth_pretrain阶段的规划头
            return self.forward_rgb_distill(depth, rgb, obs, pose)
        if mode == "rgb_percepetion_distill_train":
            return self.forward_rgb_distill(
                depth, rgb, obs, pose, compute_planning=False
            )
        if mode == "rgb_planner_finetune":
            return self.forward_rgb_finetune(rgb, obs, pose)
        if mode == "test":
            return self.forward_test(rgb, obs, pose)
        if mode == "test_r2d":
            return self.forward_test(rgb, obs, pose, return_depth=True)
        if mode == "r2d":
            return self.forward_r2d(rgb, pose)
        raise ValueError(f"Unknown forward mode: {mode}")

    def _pose_to_embedding(self, pose: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        pose = pose.to(dtype=dtype)
        if pose.dim() == 3:
            pose = self.absolute_pose_to_relative(pose)
        return self.pose_embed(pose)

    def forward_contrastive(self, rgb, depth):
        """Encode paired current-frame images with shared, jointly trained weights.

        RGB is BGR in [0,1], matching the existing single-RGB dataset. Depth is
        fixed-scale [0,1] (1 == 20 m), repeated into three channels.
        No state, pose, planning feature adapter or teacher enters this path.
        """
        if self.contrastive_projection is None:
            raise RuntimeError('Contrastive pretraining requires single-rgb-contrastive')
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        if rgb.ndim != 4 or rgb.shape[1] != 3 or depth.shape != rgb[:, :1].shape:
            raise ValueError('Expected paired RGB [B,3,H,W] and depth [B,1,H,W]')
        depth_image = depth.expand(-1, 3, -1, -1)
        features = self.rgb_backbone(torch.cat((rgb, depth_image), dim=0))
        embedding = self.contrastive_projection(features.mean(dim=(-2, -1)))
        return embedding.chunk(2, dim=0)

    def enforce_rgb_backbone_freeze(self):
        """Reapply YOLO backbone/neck/depth-head freezing after stage toggles."""
        if isinstance(self.rgb_backbone, YoloDepthEncoder):
            self.rgb_backbone._apply_backbone_freeze()

    def _encode_rgb(self, rgb: torch.Tensor, pose: torch.Tensor = None,
                    predict_depth: bool = False, contrastive_depth=None):
        """Encode RGB observations into the shared planning-token interface."""
        target_height, target_width = rgb.shape[-2:]
        if self.single_frame_no_pose:
            # Accept a sequence at external call sites for compatibility, but pass
            # only its current frame through the actual single-frame network.
            if rgb.dim() == 5:
                rgb = rgb[:, -1]
            if rgb.dim() != 4:
                raise ValueError("single-frame RGB must have shape [B, C, H, W]")
            features = self.encode_rgb_frame(rgb)
        else:
            if rgb.dim() != 5 or rgb.shape[1] != 2:
                raise ValueError("temporal RGB must have shape [B, 2, C, H, W]")
            if pose is None and not self.temporal_no_pose:
                raise ValueError("relative pose is required by the temporal RGB branch")

            B, T, C, H, W = rgb.shape
            current_indices = torch.arange(B, device=rgb.device) * T + T - 1
            encoded = self.encode_rgb_frame(
                rgb.reshape(B*T, C, H, W), contrastive_depth, current_indices
            ) if contrastive_depth is not None else self.encode_rgb_frame(rgb.reshape(B*T, C, H, W))
            if contrastive_depth is not None:
                rgb_feature, embeddings = encoded
            else:
                rgb_feature = encoded
            features = rgb_feature.reshape(
                B, T, rgb_feature.shape[1], *rgb_feature.shape[-2:]
            )
        outputs = self._encode_rgb_features(
            features, pose, predict_depth=predict_depth,
            output_size=(target_height, target_width),
        )
        return (*outputs, embeddings) if contrastive_depth is not None else outputs

    def encode_rgb_frame(self, rgb: torch.Tensor, contrastive_depth=None, current_indices=None):
        """Encode BGR frames before any pose-dependent computation (cache boundary)."""
        if rgb.dim() != 4:
            raise ValueError("RGB encoder input must have shape [B,C,H,W]")
        if contrastive_depth is not None:
            if self.depth_contrastive_projection is None:
                raise RuntimeError('Depth-feature contrastive supervision is disabled')
            geometry, predicted_feature, target_feature = self.rgb_backbone(
                rgb, target_depth=contrastive_depth, target_indices=current_indices
            )
            embeddings = tuple(self.depth_contrastive_projection(f.mean(dim=(-2, -1)))
                               for f in (predicted_feature, target_feature))
        else:
            geometry = self.rgb_backbone(rgb)
        if self.visual_rgb_backbone is None:
            return (geometry, embeddings) if contrastive_depth is not None else geometry
        visual = self.visual_rgb_backbone(rgb)
        visual = torch.nn.functional.interpolate(
            visual, size=geometry.shape[-2:], mode="bilinear", align_corners=False
        )
        # Pack both per-frame features so cached and split-ONNX inference use the
        # same interface; only the current frame's visual features are fused.
        packed = torch.cat((geometry, visual), dim=1)
        return (packed, embeddings) if contrastive_depth is not None else packed

    def _encode_rgb_features(self, features: torch.Tensor, pose: torch.Tensor = None,
                             predict_depth: bool = False, output_size=None):
        """Run the unchanged pose/temporal bridge on pre-encoded RGB features."""
        visual_current = None
        if self.rgb_visual_fusion:
            if features.dim() != 5 or features.shape[2] != 2 * self.rgb_feature_channels:
                raise ValueError("RGB fusion requires packed geometry and visual features for two frames")
            visual_current = features[:, -1, self.rgb_feature_channels:]
            features = features[:, :, :self.rgb_feature_channels]
        if self.cnn_early_fusion:
            relative_pose = self.absolute_pose_to_relative(pose) if pose is not None and pose.ndim == 3 else pose
            planning_feature_map = self.cnn_fusion(features, relative_pose)
        elif self.single_frame_no_pose:
            planning_feature_map = features[:, -1] if features.dim() == 5 else features
        else:
            if features.dim() != 5 or features.shape[1] != 2:
                raise ValueError("Temporal features must have shape [B,2,C,H,W]")
            if pose is None and not self.temporal_no_pose:
                raise ValueError("Relative pose is required by the temporal RGB branch")
            B, T, channels, H_feature, W_feature = features.shape
            if channels != self.rgb_feature_channels:
                raise ValueError("RGB feature channels do not match the configured encoder")
            N = H_feature * W_feature
            # Encode independent frames in one B*T batch, just like the RGB
            # backbone. Spatial attention remains within each individual frame.
            tokens = features.flatten(3).permute(0, 1, 3, 2)
            if not self.temporal_no_pose:
                pose_embedding = self._pose_to_embedding(pose, features.dtype)
                tokens = tokens + pose_embedding[:, None, None, :]
            tokens = tokens.reshape(B*T, N, self.rgb_feature_channels)
            position_encoder = (
                self.rgb_positional_encoding
                if self.rgb_positional_encoding is not None
                else self.positional_encoding
            )
            frame_features = self.frame_encoder(position_encoder(tokens))
            # Preserve the original per-location channel order: [frame0, frame1].
            fusion_input = frame_features.reshape(
                B, T, N, self.rgb_feature_channels
            ).permute(0, 2, 1, 3).reshape(B, N, T*self.rgb_feature_channels)
            fused_tokens = self.fusion_encoder(fusion_input)
            planning_feature_map = self.fusion_head(
                fused_tokens.permute(0, 2, 1)
            ).reshape(B, self.rgb_feature_channels, H_feature, W_feature)

        planning_input = (torch.cat((planning_feature_map, visual_current), dim=1)
                          if visual_current is not None else planning_feature_map)
        bridge_feature_map = self.r2d_plan_conv(planning_input)
        bridge_feature = bridge_feature_map.flatten(2).permute(0, 2, 1)

        depth_est = None
        clearance_est = None
        if predict_depth and self.auxiliary_depth_head is not None:
            if output_size is None:
                output_size = (int(cfg["image_height"]), int(cfg["image_width"]))
            auxiliary_coarse = (self.auxiliary_plan_conv(planning_feature_map)
                                if self.auxiliary_plan_conv is not None else bridge_feature_map)
            depth_est, clearance_est = self.auxiliary_depth_head(
                planning_feature_map,
                auxiliary_coarse,
                output_size=output_size,
            )
        return bridge_feature, bridge_feature_map, depth_est, clearance_est

    def _decode_planning(self, detail_map: torch.Tensor, obs: torch.Tensor):
        trajectory_queries = self.state_backbone(obs)
        trajectory_queries, global_tokens, detail_tokens = self.planning_adapter(
            detail_map, trajectory_queries
        )
        output = self.plan_header(trajectory_queries, global_tokens)
        output = self.traj_header(output).permute(0, 2, 1)
        output = output.reshape(
            obs.shape[0], self.output_dim,
            cfg["vertical_num"], cfg["horizon_num"]
        )
        output[:, 0:9, :, :] = output[:, 0:9, :, :].tanh()
        return output, detail_tokens

    @staticmethod
    def absolute_pose_to_relative(pose: torch.Tensor) -> torch.Tensor:
        q0 = pose[:, 0, 0:4]
        q1 = pose[:, 1, 0:4]
        p0 = pose[:, 0, 4:7]
        p1 = pose[:, 1, 4:7]
        q0 = q0 / q0.norm(dim=1, keepdim=True).clamp_min(1e-8)
        q1 = q1 / q1.norm(dim=1, keepdim=True).clamp_min(1e-8)
        q1_inv = torch.cat([-q1[:, 0:3], q1[:, 3:4]], dim=1)
        q_rel_0 = torch.stack((
            q1_inv[:, 3] * q0[:, 0] + q1_inv[:, 0] * q0[:, 3] + q1_inv[:, 1] * q0[:, 2] - q1_inv[:, 2] * q0[:, 1],
            q1_inv[:, 3] * q0[:, 1] - q1_inv[:, 0] * q0[:, 2] + q1_inv[:, 1] * q0[:, 3] + q1_inv[:, 2] * q0[:, 0],
            q1_inv[:, 3] * q0[:, 2] + q1_inv[:, 0] * q0[:, 1] - q1_inv[:, 1] * q0[:, 0] + q1_inv[:, 2] * q0[:, 3],
            q1_inv[:, 3] * q0[:, 3] - q1_inv[:, 0] * q0[:, 0] - q1_inv[:, 1] * q0[:, 1] - q1_inv[:, 2] * q0[:, 2],
        ), dim=1)
        dp_0 = p0 - p1
        qv = q1_inv[:, 0:3]
        qw = q1_inv[:, 3:4]
        t_rel_0 = dp_0 + 2.0 * (qw * torch.cross(qv, dp_0, dim=1) + torch.cross(qv, torch.cross(qv, dp_0, dim=1), dim=1))
        return torch.cat([q_rel_0, t_rel_0], dim=1)
    
    def forward_depth_pretrain(self,depth:torch.Tensor,obs:torch.Tensor):
        if self.depth_backbone is None:
            raise RuntimeError(
                "Pure single-frame RGB mode has no depth pretraining branch"
            )
        #基于深度的教师，和基于transformer的方法一致）
        depth_cur = depth[:, -1:].contiguous()
        depth_feature_map = self.depth_backbone(depth_cur)
        depth_feature_map = self.depth_plan_conv(depth_feature_map)
        output, depth_feature = self._decode_planning(depth_feature_map, obs)
        endstate, score, safety_score = self._split_traj_output(output)
        return {
            "endstate": endstate,
            "score": score,
            "safety_score": safety_score,
            "depth_feature": depth_feature,
        }
    
    def forward_rgb_distill(
            self, depth: torch.Tensor, rgb: torch.Tensor,
            obs: torch.Tensor, pose: torch.Tensor,
            compute_planning: bool = True,
            contrastive_depth=None,
    ):
        if self.depth_backbone is None:
            raise RuntimeError(
                "Pure single-frame RGB mode does not use feature distillation"
            )
        #todo：理论上第二阶段可以不走到规划层，第三阶段再用规划loss的梯度（之前测试时不是这样，可以之后尝试）
        #基于深度的教师，和基于transformer的方法一致）
        depth_cur = depth[:, -1:].contiguous()
        depth_feature_map = self.depth_backbone(depth_cur)
        depth_feature_map = self.depth_plan_conv(depth_feature_map)
        depth_feature = depth_feature_map.flatten(2).permute(0, 2, 1)

        encoded = self._encode_rgb(rgb, pose, predict_depth=True, contrastive_depth=contrastive_depth)
        bridge_feature, bridge_feature_map, depth_est, clearance_est = encoded[:4]

        outputs = {
            "depth_feature": depth_feature,
            "bridge_feature": bridge_feature,
            # Legacy alias for existing ROS visualization and analysis scripts.
            "r2d_feature": bridge_feature,
            "depth_est": depth_est,
            "clearance_est": clearance_est,
        }
        if contrastive_depth is not None:
            outputs['depth_contrastive_embeddings'] = encoded[4]
        if compute_planning:
            output, _ = self._decode_planning(bridge_feature_map, obs)
            endstate, score, safety_score = self._split_traj_output(output)
            outputs.update({
                "endstate": endstate,
                "score": score,
                "safety_score": safety_score,
            })
        return outputs
    
    def forward_rgb_finetune(self,rgb:torch.Tensor,obs:torch.Tensor,pose:torch.Tensor):
        _, bridge_feature_map, _, _ = self._encode_rgb(rgb, pose)

        #规划损失部分
        output, _ = self._decode_planning(bridge_feature_map, obs)
        endstate, score, safety_score = self._split_traj_output(output)

        return {
            "endstate": endstate,
            "score": score,
            "safety_score": safety_score,
        }

    def forward_test(self,rgb:torch.Tensor,obs:torch.Tensor,pose:torch.Tensor,return_depth: bool = False):
        bridge_feature, bridge_feature_map, depth_est, clearance_est = self._encode_rgb(
            rgb, pose, predict_depth=return_depth
        )

        return self._planning_outputs(
            bridge_feature, bridge_feature_map, obs, depth_est, clearance_est, return_depth
        )

    def forward_test_from_features(self, features, obs, pose, return_depth=False):
        bridge_feature, bridge_feature_map, depth_est, clearance_est = self._encode_rgb_features(
            features, pose, predict_depth=return_depth
        )
        return self._planning_outputs(
            bridge_feature, bridge_feature_map, obs, depth_est, clearance_est, return_depth
        )

    def _planning_outputs(self, bridge_feature, bridge_feature_map, obs,
                          depth_est=None, clearance_est=None, return_depth=False):
        output, _ = self._decode_planning(bridge_feature_map, obs)
        endstate, score, safety_score = self._split_traj_output(output)

        outputs = {
            "endstate": endstate,
            "score": score,
            "safety_score": safety_score,
            "bridge_feature": bridge_feature,
            # Legacy alias for existing ROS visualization and analysis scripts.
            "r2d_feature": bridge_feature,
        }
        if return_depth:
            outputs["depth_est"] = depth_est
            outputs["clearance_est"] = clearance_est
        return outputs

    @torch.inference_mode()
    def inference_from_rgb_features(self, features, obs, pose, return_depth=False):
        if self.training:
            raise RuntimeError("Cached RGB feature inference requires policy.eval()")
        outputs = self.forward_test_from_features(
            features, self.state_transform.normalize_obs(obs.clone()), pose, return_depth
        )
        outputs["endstate"] = self.state_transform.pred_to_endstate(outputs["endstate"])
        return outputs
    
    def forward_r2d(self,rgb:torch.Tensor,pose:torch.Tensor):
        _, _, depth_est, clearance_est = self._encode_rgb(
            rgb, pose, predict_depth=True
        )

        return {
            "depth_est": depth_est,
            "clearance_est": clearance_est,
        }

    def inference(self,depth: torch.Tensor = None,rgb: torch.Tensor = None,obs: torch.Tensor = None,pose: torch.Tensor = None,mode: str = "test",) -> dict:
        """
            Forward wrapper for inference/training stages.
            Returns the same output dictionary as forward(), with "endstate"
            converted to the body frame when possible.
        """
        if obs is not None:
            obs = self.state_transform.normalize_obs(obs)
            #obs = self.state_transform.prepare_input(obs) 使用位置编码的话，不用再这样准备输入了，只需要归一化

        outputs = self.forward(depth=depth, rgb=rgb, obs=obs, pose=pose, mode=mode)
        #output可以根据字典索引 
        if outputs.get("endstate") is not None:
            outputs["endstate"] = self.state_transform.pred_to_endstate(outputs["endstate"])
        return outputs

    def forward_trt(self, rgb: torch.Tensor, obs: torch.Tensor, pose: torch.Tensor):
        """
            TensorRT export/inference path.
            Keep inputs and outputs as tensors only so ONNX/TensorRT export has
            a stable deployment boundary.
        """
        obs = obs.clone()
        obs = self.state_transform.normalize_obs(obs)
        outputs = self.forward_test(rgb, obs, pose)
        endstate = self.state_transform.pred_to_endstate(outputs["endstate"])
        values = (endstate, outputs["score"])
        return values + (outputs['safety_score'],) if self.score_mode == 'topk-collision' else values

    def forward_trt_from_features(self, features, obs, pose):
        outputs = self.forward_test_from_features(
            features, self.state_transform.normalize_obs(obs.clone()), pose
        )
        values = (self.state_transform.pred_to_endstate(outputs["endstate"]), outputs["score"])
        return values + (outputs['safety_score'],) if self.score_mode == 'topk-collision' else values

    def _split_traj_output(self, output):
        endstate = output[:, 0:9, :, :]
        score = output[:, 9, :, :]
        safety_score = output[:, 10, :, :] if self.score_mode == 'topk-collision' else None
        return endstate, score, safety_score

    def test_state_dict(self):
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if key.startswith(self.TEST_STATE_PREFIXES)
        }

    def depth_expert_state_dict(self):
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if key.startswith(self.DEPTH_EXPERT_STATE_PREFIXES)
        }

    @staticmethod
    def checkpoint_score_mode(state_dict):
        marker = state_dict.get('planning_score_mode')
        weight = state_dict.get('traj_header.0.weight')
        value = int(marker.item()) if marker is not None else (2 if weight is not None and weight.shape[0] == 11 else 0)
        if value not in (0, 1, 2):
            raise ValueError('Unknown checkpoint planning_score_mode')
        return ('selection', 'collision', 'topk-collision')[value]

    @staticmethod
    def checkpoint_candidate_topk(state_dict):
        value = state_dict.get('planning_candidate_topk')
        return int(value.item()) if value is not None else int(cfg._data.get('selection_candidate_topk', 8))

    @staticmethod
    def checkpoint_single_frame(state_dict):
        return (any(key.startswith('rgb_backbone.') for key in state_dict)
                and not any(key.startswith(('frame_encoder.', 'cnn_fusion.')) for key in state_dict))

    @staticmethod
    def checkpoint_cnn_early_fusion(state_dict):
        return any(key.startswith('cnn_fusion.') for key in state_dict)

    @staticmethod
    def checkpoint_rgb_encoder(state_dict):
        return 'resnet' if any(key.startswith('rgb_backbone.cnn.') for key in state_dict) else 'yolo26_depth'

    @staticmethod
    def checkpoint_temporal_no_pose(state_dict):
        """Infer the temporal-pose ablation from a saved RGB checkpoint."""
        has_temporal = any(k.startswith("frame_encoder.") for k in state_dict)
        if not has_temporal:
            return False
        marker = state_dict.get("rgb_use_relative_pose")
        return not bool(marker.item()) if marker is not None else not any(
            k.startswith("pose_embed.") for k in state_dict
        )

    @staticmethod
    def checkpoint_auxiliary_depth_disabled(state_dict):
        """Full checkpoints record this; inference-only checkpoints omit it."""
        marker = state_dict.get("rgb_auxiliary_depth_enabled")
        return marker is not None and not bool(marker.item())


    @staticmethod
    def _drop_legacy_safety_output(state_dict):
        """Convert an old 11-output checkpoint to the 10-output planning head."""
        compatible = dict(state_dict)
        for key in ("traj_header.0.weight", "traj_header.0.bias"):
            tensor = compatible.get(key)
            if torch.is_tensor(tensor) and tensor.ndim >= 1 and tensor.shape[0] == 11:
                compatible[key] = tensor[:10].clone()
        return compatible

    def load_compatible_state_dict(self, state_dict):
        if self.checkpoint_cnn_early_fusion(state_dict) != self.cnn_early_fusion:
            raise ValueError('Checkpoint CNN/Transformer fusion differs from the configured model; use a matching model or --depth_expert')
        if self.single_rgb_contrastive:
            schedule = state_dict.get('contrastive_schedule')
            if (schedule is None or not torch.equal(schedule.cpu(), self.contrastive_schedule.cpu())
                    or not any(key.startswith('contrastive_projection.') for key in state_dict)):
                raise ValueError('Contrastive resume requires a matching full checkpoint with projection head and 20+20 schedule')
        source_mode = self.checkpoint_score_mode(state_dict)
        if source_mode != self.score_mode:
            raise ValueError('Checkpoint score semantics do not match the model; train a new expert/policy with matching score_mode')
        if self.score_mode == 'topk-collision' and self.checkpoint_candidate_topk(state_dict) != self.candidate_topk:
            raise ValueError('Checkpoint candidate top-k differs from the configured model')
        auxiliary_marker = state_dict.get("rgb_auxiliary_depth_enabled")
        if auxiliary_marker is not None and int(auxiliary_marker.item()) != int(self.rgb_auxiliary_depth_enabled.item()):
            raise ValueError("Checkpoint auxiliary-depth mode does not match the configured model")
        has_temporal = any(k.startswith("frame_encoder.") for k in state_dict)
        if has_temporal and self.checkpoint_temporal_no_pose(state_dict) != self.temporal_no_pose:
            raise ValueError("Checkpoint relative-pose input does not match the configured temporal model")
        checkpoint_has_visual = any(key.startswith("visual_rgb_backbone.") for key in state_dict)
        if checkpoint_has_visual != self.rgb_visual_fusion:
            raise ValueError("Checkpoint RGB visual fusion does not match rgb_visual_fusion; "
                             "use a matching checkpoint or load --depth_expert to restart Stage II.")
        pose_weight = state_dict.get("pose_embed.weight")
        if pose_weight is not None and pose_weight.shape[0] != self.rgb_feature_channels:
            raise ValueError(
                f"Checkpoint RGB width is {pose_weight.shape[0]}, but current RGB width "
                f"is {self.rgb_feature_channels}. Match rgb_feature_channels when "
                "resuming/evaluating; load --depth_expert to restart Stage II at a new width."
            )
        checkpoint_encoder = (
            "resnet" if any(key.startswith("rgb_backbone.cnn.") for key in state_dict)
            else "yolo26_depth" if any(key.startswith("rgb_backbone.backbone.") for key in state_dict)
            else None
        )
        if checkpoint_encoder is not None and checkpoint_encoder != self.rgb_encoder_type:
            raise ValueError(
                f"Checkpoint uses {checkpoint_encoder}, but rgb_encoder={self.rgb_encoder_type}. "
                "Match the encoder configuration when resuming/evaluating; use "
                "--depth_expert to initialize the new encoder from a depth expert instead."
            )
        if checkpoint_encoder == "yolo26_depth":
            head = self.rgb_backbone.depth_head
            # Some released three-scale pretrained heads retain an unused third
            # refine block; YAML construction creates only the two used blocks.
            # Preserve the checkpoint layout instead of dropping parameters.
            if (getattr(head, "nl", None) == 3 and len(head.refine) == 2
                    and "rgb_backbone.depth_head.refine.2.0.conv.weight" in state_dict):
                head.refine.append(copy.deepcopy(head.refine[0]))
        compatible = dict(state_dict) if self.score_mode == 'topk-collision' else self._drop_legacy_safety_output(state_dict)
        compatible['planning_score_mode'] = self.planning_score_mode.detach().clone()
        compatible['planning_candidate_topk'] = self.planning_candidate_topk.detach().clone()
        if "rgb_use_relative_pose" not in compatible:
            compatible["rgb_use_relative_pose"] = self.rgb_use_relative_pose.detach().clone()
        if "rgb_auxiliary_depth_enabled" not in compatible:
            compatible["rgb_auxiliary_depth_enabled"] = self.rgb_auxiliary_depth_enabled.detach().clone()
        architecture = compatible.get("rgb_architecture")
        if architecture is None:
            # Legacy RGB checkpoints used F8 and two layers in each encoder.
            temporal_layers = 2 if "pose_embed.weight" in compatible else 0
            architecture = torch.tensor([8, temporal_layers, temporal_layers])
        if not torch.equal(architecture.detach().cpu(), self.rgb_architecture.cpu()):
            raise ValueError(
                f"Checkpoint RGB layout {architecture.tolist()} does not match "
                f"current layout {self.rgb_architecture.tolist()} "
                "[feature_stride, frame_layers, fusion_layers]. Match the RGB "
                "configuration or load --depth_expert to restart Stage II."
            )
        compatible["rgb_architecture"] = self.rgb_architecture.detach().clone()
        auxiliary_keys = {
            key: value for key, value in compatible.items()
            if key.startswith("auxiliary_depth_head.")
        }
        if auxiliary_keys:
            expected_auxiliary = {
                "auxiliary_depth_head." + key: value
                for key, value in self.auxiliary_depth_head.state_dict().items()
            } if self.auxiliary_depth_head is not None else {}
            if auxiliary_keys.keys() != expected_auxiliary.keys() or any(
                auxiliary_keys[key].shape != value.shape
                for key, value in expected_auxiliary.items()
            ):
                raise ValueError(
                    "Checkpoint auxiliary depth architecture does not match the "
                    "current two-scale decoder. Use a matching checkpoint "
                    "or load --depth_expert to restart Stage II."
                )
        if checkpoint_encoder == "yolo26_depth":
            if any(key.startswith(("rgb_backbone.projections.", "rgb_backbone.scale_logits"))
                   for key in compatible) or "rgb_backbone.depth_features.stem.0.weight" not in compatible:
                raise ValueError(
                    "Checkpoint uses a previous YOLO encoder, not the native-depth pipeline. "
                    "Load --depth_expert to restart Stage II with the complete pretrained "
                    "YOLODepth head and lightweight depth convolutions."
                )
            # Do not silently initialize a missing decoder/calibration or a
            # different small extractor when resuming a full RGB checkpoint.
            encoder_prefixes = ("rgb_backbone.depth_head.", "rgb_backbone.depth_features.")
            normalization_key = "rgb_backbone.depth_normalization_m"
            expected_depth = {
                key: value for key, value in self.state_dict().items()
                if key.startswith(encoder_prefixes) or key == normalization_key
            }
            actual_depth = {
                key: value for key, value in compatible.items()
                if key.startswith(encoder_prefixes) or key == normalization_key
            }
            if actual_depth.keys() != expected_depth.keys() or any(
                    actual_depth[key].shape != value.shape
                    for key, value in expected_depth.items()
            ):
                raise ValueError(
                    "Checkpoint YOLO depth head, calibration or depth feature extractor "
                    "does not match the current model. Match the YOLO architecture and "
                    "yolo_depth_feature_channels; use --depth_expert for a new RGB pipeline."
                )
            prefix = "rgb_backbone.adapters."
            expected = {
                prefix + key: value
                for key, value in self.rgb_backbone.adapters.state_dict().items()
            }
            actual = {key: value for key, value in compatible.items() if key.startswith(prefix)}
            if expected and actual and not any(key.endswith(".residual_scale") for key in actual):
                # Legacy trained adapters must keep their original scale 1.0;
                # applying the new YAML default would change predictions.
                for key, value in expected.items():
                    if key.endswith(".residual_scale"):
                        compatible[key] = actual[key] = torch.ones_like(value)
                warnings.warn(
                    "Checkpoint has no adapter residual-scale metadata; restoring "
                    "legacy scale 1.0 to preserve predictions.",
                    UserWarning, stacklevel=2,
                )
            if expected and not actual:
                # A native-depth checkpoint without adapters remains an exact warm start.
                # Reset the residual output even when loading into a used model.
                for key, value in expected.items():
                    compatible[key] = (
                        torch.zeros_like(value) if key.endswith(".up.weight")
                        else value.detach().clone()
                    )
                warnings.warn(
                    "Checkpoint has no YOLO adapters; initializing zero-output "
                    "residual adapters and preserving its original predictions.",
                    UserWarning, stacklevel=2,
                )
            elif actual.keys() != expected.keys() or any(
                    actual[key].shape != value.shape for key, value in expected.items()
            ):
                raise ValueError(
                    "Checkpoint YOLO adapter configuration does not match the current "
                    "model. Match yolo_depth_adapter_enabled, "
                    "yolo_depth_adapter_bottleneck and yolo_depth_adapter_layers."
                )
        return self.load_state_dict(compatible, strict=False)

    def load_depth_expert_state_dict(self, state_dict):
        """只载入深度专家相关权重(深度骨干 + 共享规划头)，RGB 分支保留构造时初始化。
        YOLO 分支保留完整预训练 backbone/neck/depth head 和零输出 adapter。
        兼容 *_depth_expert.pth 快照和完整 checkpoint(自动按 DEPTH_EXPERT_STATE_PREFIXES 过滤)。"""
        if self.checkpoint_score_mode(state_dict) != self.score_mode:
            raise ValueError('Depth expert score semantics do not match; retrain the expert for collision-cost regression')
        state_dict = dict(state_dict) if self.score_mode == 'topk-collision' else self._drop_legacy_safety_output(state_dict)
        if 'planning_score_mode' not in state_dict:
            state_dict['planning_score_mode'] = self.planning_score_mode.detach().clone()
        state_dict['planning_candidate_topk'] = self.planning_candidate_topk.detach().clone()
        depth_expert_sd = {
            key: value for key, value in state_dict.items()
            if key.startswith(self.DEPTH_EXPERT_STATE_PREFIXES)
        }
        if not depth_expert_sd:
            raise RuntimeError(
                f"Checkpoint contains no depth-expert weights "
                f"(expected prefixes {self.DEPTH_EXPERT_STATE_PREFIXES})."
            )
        load_result = self.load_state_dict(depth_expert_sd, strict=False)
        missing_depth_keys = [
            key for key in load_result.missing_keys
            if key.startswith(self.DEPTH_EXPERT_STATE_PREFIXES)
        ]
        if missing_depth_keys:
            raise RuntimeError(f"Missing depth expert weights: {missing_depth_keys}")
        return load_result

    def load_test_state_dict(self, state_dict):
        load_result = self.load_compatible_state_dict(state_dict)
        missing_test_keys = [
            key for key in load_result.missing_keys
            if key.startswith(self.TEST_STATE_PREFIXES)
        ]
        if missing_test_keys:
            raise RuntimeError(f"Missing test inference weights: {missing_test_keys}")
        return load_result

    def print_grad(self, grad):
        print("grad of hook: ", grad)
