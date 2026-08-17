"""训练公共工具。"""

from __future__ import annotations

import random

import torch
from torch import nn


def set_seed(seed: int = 42) -> None:
    """输入随机种子，返回无；固定 Python 与 PyTorch 随机性。"""
    random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """输入为空，返回优先使用 CUDA 的计算设备。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parameter_count(model: nn.Module) -> int:
    """输入模型，返回可训练参数数量。"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    print(get_device())
