"""最小回归模型。"""

from __future__ import annotations

import torch
from torch import nn


class RegressionModel(nn.Module):
    """用两层 MLP 拟合一维非线性函数。"""

    def __init__(self, hidden_dim: int = 32) -> None:
        """输入隐藏维度，返回初始化后的模型。"""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入形状为 [batch, 1]，返回同形状预测。"""
        return self.net(x)


if __name__ == "__main__":
    print(RegressionModel()(torch.zeros(2, 1)).shape)
