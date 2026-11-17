import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=3000):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(max_len, d_model)  # shape: (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 计算偶数维度的 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 计算奇数维度的 cos
        pe = pe.unsqueeze(0)  # 增加 batch 维度，shape: (1, max_len, d_model)
        self.register_buffer('positional_encoding', pe)

    def forward(self, x):
        # x 的形状应该是 (batch_size, seq_len, d_model)
        pe = self.positional_encoding.repeat(x.size(0), 1, 1)
        return x + pe[:, 0:x.shape[1], :]


class PrimitiveEncoder(nn.Module):
    def __init__(self, d_model, primitive_num, primitive_dim=9, state_dim=9):
        super(PrimitiveEncoder, self).__init__()
        self.primitive_num = primitive_num
        self.primitive_token = nn.Parameter(torch.rand(1, primitive_num, primitive_dim))
        self.linear1 = nn.Linear(primitive_dim + state_dim, d_model)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)

    def forward(self, state):
        # state: [batch, featrue_dim]
        state = state.unsqueeze(1).repeat(1, self.primitive_num, 1)
        primitive_token = self.primitive_token.repeat(state.size(0), 1, 1)
        x = torch.cat([primitive_token, state], dim=2)
        primitive_embedding = self.linear2(self.relu(self.linear1(x)))
        return primitive_embedding


class TrajectoryAlignedPlanningAdapter(nn.Module):
    """Fuse dense visual detail into sparse trajectory queries.

    Each trajectory query attends to a small angularly aligned window on the
    native feature map.  A compact pooled memory then supplies scene-wide
    context to the shared planning decoder.
    """

    def __init__(
            self,
            d_model,
            nhead,
            vertical_num,
            horizon_num,
            vertical_fov_deg,
            horizon_fov_deg,
            local_window=5,
            global_size=(2, 3),
    ):
        super().__init__()
        if local_window <= 0 or local_window % 2 == 0:
            raise ValueError("planning_local_window must be a positive odd number")
        if min(global_size) <= 0:
            raise ValueError("planning global grid dimensions must be positive")

        self.vertical_num = int(vertical_num)
        self.horizon_num = int(horizon_num)
        self.query_num = self.vertical_num * self.horizon_num
        self.vertical_fov = math.radians(float(vertical_fov_deg))
        self.horizon_fov = math.radians(float(horizon_fov_deg))
        self.local_window = int(local_window)
        self.global_size = tuple(int(value) for value in global_size)

        self.angular_embedding = nn.Sequential(
            nn.Linear(4, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.local_attention = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )
        self.local_norm = nn.LayerNorm(d_model)
        self.global_fusion = nn.Conv2d(
            2 * d_model, d_model, kernel_size=1, bias=False
        )
        self.global_norm = nn.LayerNorm(d_model)

        # Network outputs use image-grid order (top-left to bottom-right).
        # StateTransform maps that grid to LatticePrimitive with ``flip(0)``,
        # so query angles must run positive-to-negative for both axes.  Using
        # the raw lattice order here would align output query 0 (top-left)
        # with the bottom-right visual window.
        pitch = self._axis_centers(self.vertical_num, self.vertical_fov, True)
        yaw = self._axis_centers(self.horizon_num, self.horizon_fov, True)
        query_pitch, query_yaw = torch.meshgrid(pitch, yaw, indexing="ij")
        self.register_buffer(
            "query_angles",
            torch.stack((query_yaw.reshape(-1), query_pitch.reshape(-1)), dim=-1),
            persistent=False,
        )

    @staticmethod
    def _axis_centers(count, field_of_view, reverse):
        centers = (
            torch.arange(count, dtype=torch.float32) - (count - 1) / 2.0
        ) * (field_of_view / count)
        return centers.flip(0) if reverse else centers

    @staticmethod
    def _angle_features(angles):
        yaw, pitch = angles.unbind(dim=-1)
        return torch.stack(
            (torch.sin(yaw), torch.cos(yaw), torch.sin(pitch), torch.cos(pitch)),
            dim=-1,
        )

    def _grid_angles(self, height, width, device):
        # Image rows/columns run top-to-bottom and left-to-right, the reverse
        # of positive body-frame pitch/yaw.
        pitch = self._axis_centers(height, self.vertical_fov, True).to(device)
        yaw = self._axis_centers(width, self.horizon_fov, True).to(device)
        grid_pitch, grid_yaw = torch.meshgrid(pitch, yaw, indexing="ij")
        return torch.stack(
            (grid_yaw.reshape(-1), grid_pitch.reshape(-1)), dim=-1
        )

    def _query_center_indices(self, height, width, device):
        angles = self.query_angles.to(device)
        yaw, pitch = angles.unbind(dim=-1)
        columns = torch.round(
            (0.5 - yaw / self.horizon_fov) * width - 0.5
        ).long().clamp(0, width - 1)
        rows = torch.round(
            (0.5 - pitch / self.vertical_fov) * height - 0.5
        ).long().clamp(0, height - 1)
        return rows * width + columns

    def _local_indices(self, height, width, device):
        centers = self._query_center_indices(height, width, device)
        center_rows = centers // width
        center_columns = centers % width
        offsets = torch.arange(
            -(self.local_window // 2),
            self.local_window // 2 + 1,
            device=device,
        )
        row_offsets, column_offsets = torch.meshgrid(
            offsets, offsets, indexing="ij"
        )
        rows = (
            center_rows[:, None] + row_offsets.reshape(1, -1)
        ).clamp(0, height - 1)
        columns = (
            center_columns[:, None] + column_offsets.reshape(1, -1)
        ).clamp(0, width - 1)
        return rows * width + columns

    def _position_embedding(self, angles, dtype):
        features = self._angle_features(angles).to(dtype=dtype)
        return self.angular_embedding(features)

    def forward(self, detail_map, trajectory_queries):
        if detail_map.dim() != 4:
            raise ValueError("detail_map must have shape [B, C, H, W]")
        if trajectory_queries.dim() != 3:
            raise ValueError("trajectory_queries must have shape [B, N, C]")

        batch_size, channels, height, width = detail_map.shape
        if trajectory_queries.shape[:2] != (batch_size, self.query_num):
            raise ValueError(
                f"Expected {self.query_num} trajectory queries, got "
                f"{trajectory_queries.shape[1]}"
            )

        detail_tokens = detail_map.flatten(2).permute(0, 2, 1)
        grid_angles = self._grid_angles(height, width, detail_map.device)
        grid_position = self._position_embedding(
            grid_angles, detail_map.dtype
        ).transpose(0, 1).reshape(1, channels, height, width)
        positioned_tokens = (detail_map + grid_position).flatten(2).permute(0, 2, 1)
        local_indices = self._local_indices(
            height, width, detail_map.device
        ).reshape(-1)
        local_memory = positioned_tokens.index_select(1, local_indices).reshape(
            batch_size * self.query_num,
            self.local_window * self.local_window,
            channels,
        )

        query_position = self._position_embedding(
            self.query_angles.to(detail_map.device), trajectory_queries.dtype
        ).unsqueeze(0)
        positioned_queries = trajectory_queries + query_position
        local_query = positioned_queries.reshape(
            batch_size * self.query_num, 1, channels
        )
        local_output, _ = self.local_attention(
            local_query, local_memory, local_memory, need_weights=False
        )
        aligned_queries = self.local_norm(
            positioned_queries
            + local_output.reshape(batch_size, self.query_num, channels)
        )

        # Bilinear resize is ONNX/TensorRT friendly even when the native grid
        # (9x15) is not divisible by the requested 2x3 global grid.  Repeating
        # the scene mean adds an explicit whole-frame summary to every token.
        global_coarse = F.interpolate(
            detail_map, size=self.global_size, mode="bilinear", align_corners=False
        )
        global_mean = detail_map.mean(dim=(2, 3), keepdim=True).expand(
            -1, -1, self.global_size[0], self.global_size[1]
        )
        global_map = self.global_fusion(
            torch.cat((global_coarse, global_mean), dim=1)
        )
        global_tokens = global_map.flatten(2).permute(0, 2, 1)
        global_angles = self._grid_angles(
            self.global_size[0], self.global_size[1], detail_map.device
        )
        global_tokens = self.global_norm(
            global_tokens
            + self._position_embedding(global_angles, global_tokens.dtype).unsqueeze(0)
        )

        return aligned_queries, global_tokens, detail_tokens


class DPTResidualConvUnit(nn.Module):
    """Pre-activation residual unit used by the lightweight DPT decoder."""

    def __init__(self, features):
        super().__init__()
        self.activation = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

    def forward(self, feature):
        residual = self.activation(feature)
        residual = self.conv1(residual)
        residual = self.activation(residual)
        residual = self.conv2(residual)
        return feature + residual


class TwoScaleDPTDepthHead(nn.Module):
    """Decode metric depth from the two feature maps also used by planning.

    ``fine_feature`` is the final fused map (default F16: 17x30), and
    ``planning_feature`` is the 1/32 map (9x15) consumed by the planning
    adapter.  The decoder intentionally has no shallower RGB skip: auxiliary
    geometry supervision therefore has to pass through planning-relevant
    features instead of being solved from low-level texture alone.

    The clearance output is an auxiliary normalized nearest-depth regression
    at the native planning resolution.  It is not fused back into planning.
    """

    def __init__(self, fine_channels, planning_channels, features=32):
        super().__init__()
        self.fine_projection = nn.Conv2d(
            fine_channels, features, kernel_size=1, bias=False
        )
        self.planning_projection = nn.Conv2d(
            planning_channels, features, kernel_size=1, bias=False
        )
        self.coarse_refine = DPTResidualConvUnit(features)
        self.intermediate_refine = DPTResidualConvUnit(features)
        self.fine_skip_refine = DPTResidualConvUnit(features)
        self.fused_refine = DPTResidualConvUnit(features)

        # Predict normalized metric clearance (1.0 == the configured maximum
        # depth) directly from P32.  This tiny branch is training/debug-only.
        self.clearance_head = nn.Conv2d(
            planning_channels, 1, kernel_size=1, stride=1, padding=0
        )

        # Refine progressively at 1/4 and 1/2 resolution. Explicit target
        # sizes in forward handle the odd 270-pixel input height without crops.
        self.quarter_refine = nn.Sequential(
            nn.Conv2d(features, 16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.half_refine = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.output_conv = nn.Conv2d(8, 1, kernel_size=3, padding=1)

    @staticmethod
    def _resize(feature, size):
        return F.interpolate(
            feature, size=size, mode="bilinear", align_corners=False
        )

    def forward(self, fine_feature, planning_feature, output_size):
        if fine_feature.dim() != 4 or planning_feature.dim() != 4:
            raise ValueError("DPT depth features must have shape [B,C,H,W]")
        if fine_feature.shape[0] != planning_feature.shape[0]:
            raise ValueError("DPT depth features must have the same batch size")

        fine_height, fine_width = fine_feature.shape[-2:]
        coarse = self.coarse_refine(
            self.planning_projection(planning_feature)
        )

        # Keep the original refinement blocks. F16 uses a P32 intermediate
        # grid; the same decoder also supports legacy F8 features.
        intermediate_size = (
            max(1, (fine_height + 1) // 2),
            max(1, (fine_width + 1) // 2),
        )
        coarse = self._resize(coarse, intermediate_size)
        coarse = self.intermediate_refine(coarse)
        coarse = self._resize(coarse, (fine_height, fine_width))

        fine = self.fine_skip_refine(self.fine_projection(fine_feature))
        decoded = self.fused_refine(coarse + fine)

        output_height, output_width = (int(value) for value in output_size)
        quarter_size = (
            max(1, (output_height + 3) // 4),
            max(1, (output_width + 3) // 4),
        )
        half_size = (
            max(1, (output_height + 1) // 2),
            max(1, (output_width + 1) // 2),
        )
        decoded = self._resize(decoded, quarter_size)
        decoded = self.quarter_refine(decoded)
        decoded = self._resize(decoded, half_size)
        decoded = self.half_refine(decoded)
        decoded = self._resize(decoded, (output_height, output_width))

        depth_est = self.output_conv(decoded).squeeze(1).sigmoid()
        clearance_est = self.clearance_head(planning_feature).sigmoid()
        return depth_est, clearance_est


class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super(CustomTransformerDecoderLayer, self).__init__()

        self.self_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # 前馈网络部分
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 层归一化
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory):
        # tgt: [batch_size, seq_len, d_model]
        # memory: [batch_size, seq_len, d_model]

        self_attn_output, _ = self.self_attention(tgt, tgt, tgt, need_weights=False)
        tgt = self.norm0(tgt + self_attn_output)

        # 交叉注意力（target-to-memory attention）
        cross_attn_output, _ = self.cross_attention(tgt, memory, memory, need_weights=False)
        tgt = self.norm1(tgt + cross_attn_output)  # 残差连接和归一化

        # 前馈网络
        ff_output = self.linear2(self.relu(self.linear1(tgt)))
        tgt = self.norm2(tgt + ff_output)  # 残差连接和归一化

        return tgt


class CustomTransformerDecoder(nn.Module):
    def __init__(self, layer, num_layers):
        super(CustomTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, tgt, memory):
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return tgt


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super(CustomTransformerEncoderLayer, self).__init__()

        self.self_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # 前馈网络部分
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 层归一化
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, src):
        # src: [batch_size, seq_len, d_model]

        # 自注意力
        self_attn_output, _ = self.self_attention(src, src, src, need_weights=False)
        src = self.norm0(src + self_attn_output)

        # 前馈网络
        ff_output = self.linear2(self.relu(self.linear1(src)))
        src = self.norm1(src + ff_output)  # 残差连接和归一化

        return src


class CustomTransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src


# [32, 64] 重塑为4x8的patch: [batch, 8x8, 4x8]
def patchify(images, n_patches_h, n_patches_w):
    n, c, h, w = images.shape

    patch_size_h = h // n_patches_h
    patch_size_w = w // n_patches_w

    patches = torch.zeros(n, n_patches_h * n_patches_w, patch_size_h * patch_size_w * c)

    for idx, image in enumerate(images):
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                patch = image[:, i * patch_size_h: (i + 1) * patch_size_h, j * patch_size_w: (j + 1) * patch_size_w]
                patches[idx, i * n_patches_w + j] = patch.flatten()

    return patches


class MLPBackbone(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=128, output_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)



# input: [3, 160, 640]
class ResNet18(torch.nn.Module):
    def __init__(self, output_dim: int):
        super(ResNet18, self).__init__()
        self.cnn = resnet18(pretrained=False)
        self.cnn.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.avgpool = torch.nn.Sequential()
        self.cnn.output_layer = torch.nn.Conv2d(512, output_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.cnn(depth)


# Faster and smaller (input: [3, 160, 640])
class ResNet14(torch.nn.Module):
    def __init__(self, output_dim: int):
        super(ResNet14, self).__init__()
        self.cnn = resnet18(pretrained=False)
        self.cnn.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.layer4 = torch.nn.Sequential()
        self.cnn.avgpool = torch.nn.Sequential()
        self.cnn.output_layer = torch.nn.Conv2d(256, output_dim, kernel_size=1, stride=1, padding=0, bias=False)

class ResNet_rgb(torch.nn.Module):
    def __init__(self, output_dim: int):
        super(ResNet_rgb, self).__init__()
        self.cnn = resnet18(pretrained=False)
        self.cnn.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.layer3 = torch.nn.Sequential()
        self.cnn.layer4 = torch.nn.Sequential()
        self.cnn.avgpool = torch.nn.Sequential()
        self.cnn.output_layer = torch.nn.Conv2d(128, output_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.cnn(depth)
    
class ResNet_depth(torch.nn.Module):
    #240 640的输出是8 20
    def __init__(self, output_dim: int):
        super(ResNet_depth, self).__init__()
        self.cnn = resnet18(pretrained=False)
        self.cnn.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cnn.output_layer = torch.nn.Conv2d(512, output_dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.cnn(depth)


def YopoBackbone(output_dim):
    return ResNet14(output_dim)


if __name__ == '__main__':
    net = ResNet_depth(128)
    input_ = torch.zeros((1, 1 ,240, 640))
    output = net(input_)
    print(output.shape)
