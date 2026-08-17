"""DDP 演示模型。"""

from __future__ import annotations

import torch
from torch import nn


class Classifier(nn.Module):
    """对二维点做二分类的小型 MLP。"""

    def __init__(self, hidden_dim: int = 16) -> None:
        """输入隐藏维度，返回初始化分类器。"""
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 [batch, 2] 特征，返回 [batch, 2] logits。"""
        return self.net(x)


if __name__ == "__main__":
    print(Classifier()(torch.zeros(2, 2)).shape)
