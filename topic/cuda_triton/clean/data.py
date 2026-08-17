"""输入: 批大小与特征维度。
返回: 算子测试输入与目标。"""

import torch
from torch import Tensor


def make_batch(batch_size: int = 32, features: int = 32, outputs: int = 16) -> tuple[Tensor, Tensor]:
    """输入: 批次和张量维度。
    返回: 特征与回归目标。"""
    values = torch.randn(batch_size, features)
    target = torch.randn(batch_size, outputs)
    return values, target


def make_matrices(size: int = 64) -> tuple[Tensor, Tensor]:
    """输入: 方阵单边长度。
    返回: 两个矩阵乘法输入。"""
    return torch.randn(size, size), torch.randn(size, size)
