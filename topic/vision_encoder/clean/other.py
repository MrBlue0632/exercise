"""输入: 补丁位置参数与新网格。
返回: 插值后的视觉位置编码。"""

import torch
from torch import Tensor
from torch.nn import functional as F


def interpolate_positions(positions: Tensor, target_side: int) -> Tensor:
    """输入: [1,N+1,D] 位置编码。
    返回: 新网格的位置编码。"""
    cls, patches = positions[:, :1], positions[:, 1:]
    side = int(patches.size(1) ** 0.5)
    patches = patches.transpose(1, 2).reshape(1, -1, side, side)
    # 适配不同图像分辨率
    patches = F.interpolate(patches, size=target_side, mode="bicubic", align_corners=False)
    patches = patches.flatten(2).transpose(1, 2)
    return torch.cat((cls, patches), dim=1)
