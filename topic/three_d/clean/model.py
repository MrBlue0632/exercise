"""输入: 三维位置与视线方向。
返回: 密度和颜色预测。"""

import torch
from torch import Tensor, nn


class PositionalEncoder(nn.Module):
    """输入: 三维连续坐标。
    返回: 正余弦位置特征。"""

    def __init__(self, bands: int = 4) -> None:
        """输入：频带数量；返回：初始化编码器。"""
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(bands))

    def forward(self, values: Tensor) -> Tensor:
        """输入: [...,3] 位置张量。
        返回: [...,3+6L] 编码。"""
        angles = values[..., None] * self.frequencies
        features = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        return torch.cat((values, features), dim=-1)


class TinyNeRF(nn.Module):
    """输入: 编码后的三维位置。
    返回: 颜色与密度预测。"""

    def __init__(self, bands: int = 4, width: int = 48) -> None:
        """输入：频带和宽度；返回：初始化辐射场。"""
        super().__init__()
        self.encoder = PositionalEncoder(bands)
        encoded = 3 + 6 * bands
        self.network = nn.Sequential(
            nn.Linear(encoded, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU()
        )
        self.density = nn.Linear(width, 1)
        self.color = nn.Linear(width, 3)

    def forward(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """输入: [...,3] 采样位置。
        返回: RGB颜色和非负密度。"""
        hidden = self.network(self.encoder(positions))
        return self.color(hidden).sigmoid(), self.density(hidden).relu()


class GaussianCloud(nn.Module):
    """输入: 高斯数量与初始尺度。
    返回: 可优化三维高斯集合。"""

    def __init__(self, count: int = 32) -> None:
        """输入：高斯数量；返回：初始化点云。"""
        super().__init__()
        self.means = nn.Parameter(torch.randn(count, 3) * 0.2)
        self.colors = nn.Parameter(torch.rand(count, 3))
        self.log_scales = nn.Parameter(torch.zeros(count, 1) - 2)

    def forward(self) -> tuple[Tensor, Tensor, Tensor]:
        """输入: 无额外输入。
        返回: 均值、颜色和标准差。"""
        return self.means, self.colors.sigmoid(), self.log_scales.exp()
