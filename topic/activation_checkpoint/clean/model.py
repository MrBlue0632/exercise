"""带激活重计算的最小分类模型。"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class FeedForwardBlock(nn.Module):
    """一个可被 checkpoint 包裹的前馈块。"""

    def __init__(self, dim: int) -> None:
        """输入特征维度，返回残差前馈块。"""
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 [batch, dim]，返回同形状残差结果。"""
        return x + self.net(x)


class CheckpointedClassifier(nn.Module):
    """训练时可选择重算中间激活的分类器。"""

    def __init__(self, dim: int = 32, depth: int = 4, use_checkpoint: bool = True) -> None:
        """输入维度、层数和开关，返回初始化模型。"""
        super().__init__()
        self.input = nn.Linear(8, dim)
        self.blocks = nn.ModuleList(FeedForwardBlock(dim) for _ in range(depth))
        self.head = nn.Linear(dim, 2)
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 [batch, 8] 特征，返回 [batch, 2] 分类 logits。"""
        x = self.input(x)
        for block in self.blocks:
            if self.training and self.use_checkpoint:
                # 反向重算以节省激活内存
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.head(x)


if __name__ == "__main__":
    print(CheckpointedClassifier()(torch.zeros(2, 8)).shape)
