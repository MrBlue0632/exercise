"""最小 LoRA 分类器。"""

import copy
import math

import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    """冻结线性层上的低秩增量。"""

    def __init__(self, base: nn.Linear, rank: int = 2, alpha: float = 4.0) -> None:
        """输入：基线线性层、秩和缩放；返回：初始化模块。"""
        super().__init__()
        self.base = copy.deepcopy(base)
        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, value: Tensor) -> Tensor:
        """输入: value[B,D]。
        返回: output[B,C]。
        """
        update = (value @ self.lora_a.t()) @ self.lora_b.t()
        return self.base(value) + update * self.scale


class BaseClassifier(nn.Module):
    """用于预训练的普通分类器。"""

    def __init__(self, width: int = 16) -> None:
        """输入：隐藏维度；返回：初始化模块。"""
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(2, width), nn.Tanh())
        self.head = nn.Linear(width, 2)

    def forward(self, value: Tensor) -> Tensor:
        """输入: value[B,2]。
        返回: logits[B,2]。
        """
        return self.head(self.feature(value))


class LoRAClassifier(nn.Module):
    """只训练 LoRA 分支的分类器。"""

    def __init__(self, base: BaseClassifier, rank: int = 2) -> None:
        """输入：基线分类器和秩；返回：初始化模块。"""
        super().__init__()
        self.feature = copy.deepcopy(base.feature)
        self.head = LoRALinear(base.head, rank=rank)

    def forward(self, value: Tensor) -> Tensor:
        """输入: value[B,2]。
        返回: logits[B,2]。
        """
        return self.head(self.feature(value))


def build_model() -> BaseClassifier:
    """输入: 无。
    返回: BaseClassifier。
    """
    return BaseClassifier()
