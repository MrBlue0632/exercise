"""输入: 光线数量和采样数量。
返回: 合成射线与目标颜色。"""

import torch
from torch import Tensor


def make_rays(count: int = 32) -> tuple[Tensor, Tensor, Tensor]:
    """输入: 需要生成的射线数。
    返回: 原点、方向和目标颜色。"""
    origins = torch.zeros(count, 3)
    directions = torch.randn(count, 3)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    targets = (directions + 1) / 2
    return origins, directions, targets


def make_samples(origins: Tensor, directions: Tensor, steps: int = 8) -> tuple[Tensor, Tensor]:
    """输入: 光线原点与单位方向。
    返回: 每条光线的位置和间隔。"""
    distances = torch.linspace(0.1, 1.0, steps, device=origins.device)
    positions = origins[:, None] + directions[:, None] * distances[None, :, None]
    deltas = distances.diff(prepend=distances[:1])
    return positions, deltas
