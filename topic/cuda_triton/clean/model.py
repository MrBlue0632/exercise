"""输入: 特征张量与线性权重。
返回: 融合线性层输出。"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FusedLinear(nn.Module):
    """输入: 特征维度和输出维度。
    返回: 可训练线性算子。"""

    def __init__(self, features: int = 32, outputs: int = 16) -> None:
        """输入：特征与输出维度；返回：初始化模块。"""
        super().__init__()
        self.weight = nn.Parameter(torch.randn(outputs, features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(outputs))

    def forward(self, values: Tensor) -> Tensor:
        """输入: [B,F] 特征张量。
        返回: [B,O] 激活结果。"""
        # 保持算子边界清晰
        return F.gelu(F.linear(values, self.weight, self.bias))


def backend_name(values: Tensor) -> str:
    """输入: 任意设备特征张量。
    返回: 当前可用后端名称。"""
    return "cuda" if values.is_cuda else "torch_cpu"
