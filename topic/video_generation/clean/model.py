"""因子化时空注意力的视频流匹配器。"""

import torch
from torch import Tensor, nn

from other import sinusoidal_embedding


class FactorizedBlock(nn.Module):
    """先做空间注意力，再做时间注意力。"""

    def __init__(self, dim: int, heads: int):
        """输入：特征维度、头数；返回：时空块。"""
        super().__init__()
        self.spatial_norm = nn.LayerNorm(dim)
        self.temporal_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, tokens: Tensor, condition: Tensor) -> Tensor:
        """输入：[B,T,S,D]词元、条件；返回：更新词元。"""
        batch, frames, sites, dim = tokens.shape
        context = condition[:, None, None, :]

        # 每帧独立混合空间词元
        spatial = self.spatial_norm(tokens + context).reshape(batch * frames, sites, dim)
        spatial = self.spatial_attn(spatial, spatial, spatial, need_weights=False)[0]
        tokens = tokens + spatial.view(batch, frames, sites, dim)

        # 每个空间点混合时间词元
        temporal = self.temporal_norm(tokens + context).permute(0, 2, 1, 3)
        temporal = temporal.reshape(batch * sites, frames, dim)
        temporal = self.temporal_attn(temporal, temporal, temporal, need_weights=False)[0]
        temporal = temporal.view(batch, sites, frames, dim).permute(0, 2, 1, 3)
        tokens = tokens + temporal

        # 逐词元补充非线性变换
        return tokens + self.mlp(self.mlp_norm(tokens + context))


class VideoFlowTransformer(nn.Module):
    """用因子化 Transformer 预测视频速度。"""

    def __init__(self, dim: int = 64, depth: int = 2, heads: int = 4, patch: int = 4):
        """输入：维度、层数、头数、补丁；返回：视频模型。"""
        super().__init__()
        self.dim = dim
        self.patch_size = patch
        self.patch_embed = nn.Conv3d(1, dim, (1, patch, patch), stride=(1, patch, patch))
        self.patch_decode = nn.ConvTranspose3d(dim, 1, (1, patch, patch), stride=(1, patch, patch))
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.direction_embed = nn.Embedding(4, dim)
        self.blocks = nn.ModuleList(FactorizedBlock(dim, heads) for _ in range(depth))
        self.final_norm = nn.LayerNorm(dim)

    def _position(self, frames: int, sites: int, device: torch.device) -> Tensor:
        """输入：帧数、空间数、设备；返回：[1,T,S,D]位置编码。"""
        frame_ids = torch.arange(frames, device=device)
        site_ids = torch.arange(sites, device=device)
        frame_pos = sinusoidal_embedding(frame_ids, self.dim).view(1, frames, 1, self.dim)
        site_pos = sinusoidal_embedding(site_ids, self.dim).view(1, 1, sites, self.dim)
        return frame_pos + site_pos

    def forward(self, video: Tensor, time: Tensor, direction: Tensor) -> Tensor:
        """输入：视频、时间、方向；返回：同形状速度预测。"""
        feature = self.patch_embed(video)
        batch, dim, frames, height, width = feature.shape
        tokens = feature.permute(0, 2, 3, 4, 1).reshape(batch, frames, height * width, dim)
        condition = self.time_mlp(sinusoidal_embedding(time, dim)) + self.direction_embed(direction)

        # 加位置后重复时空混合
        tokens = tokens + self._position(frames, height * width, video.device)
        for block in self.blocks:
            tokens = block(tokens, condition)

        # 还原视频像素网格
        tokens = self.final_norm(tokens).view(batch, frames, height, width, dim)
        feature = tokens.permute(0, 4, 1, 2, 3).contiguous()
        return self.patch_decode(feature)
