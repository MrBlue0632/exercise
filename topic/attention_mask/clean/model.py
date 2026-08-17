"""显式掩码的语言模型。"""

import math

import torch
from torch import Tensor, nn


class MaskedAttention(nn.Module):
    """接收布尔可见矩阵的注意力。"""

    def __init__(self, width: int = 48, heads: int = 4) -> None:
        """输入：隐藏维度和头数；返回：初始化模块。"""
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, hidden: Tensor, allow: Tensor) -> tuple[Tensor, Tensor]:
        """输入: hidden[B,T,D], allow[B,T,T]。
        返回: output[B,T,D], weight[B,H,T,T]。
        """
        batch, steps, width = hidden.shape
        qkv = self.qkv(hidden).view(batch, steps, 3, self.heads, self.head_width)
        query, key, value = qkv.unbind(dim=2)
        query, key, value = (item.transpose(1, 2) for item in (query, key, value))
        score = query @ key.transpose(-2, -1) / math.sqrt(self.head_width)
        score = score.masked_fill(~allow[:, None], torch.finfo(score.dtype).min)
        weight = torch.softmax(score, dim=-1)
        output = weight @ value
        output = output.transpose(1, 2).contiguous().view(batch, steps, width)
        return self.output(output), weight


class MaskedLM(nn.Module):
    """支持多类 mask 的小语言模型。"""

    def __init__(self, vocab_size: int = 32, width: int = 48) -> None:
        """输入：词表大小和隐藏维度；返回：初始化模块。"""
        super().__init__()
        self.token = nn.Embedding(vocab_size, width, padding_idx=0)
        self.position = nn.Embedding(32, width)
        self.attention = MaskedAttention(width)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, ids: Tensor, allow: Tensor) -> tuple[Tensor, Tensor]:
        """输入: ids[B,T], allow[B,T,T]。
        返回: logits[B,T,V], weight[B,H,T,T]。
        """
        positions = torch.arange(ids.size(1), device=ids.device)
        hidden = self.token(ids) + self.position(positions)[None]
        attended, weight = self.attention(hidden, allow)
        return self.head(self.norm(hidden + attended)), weight


def build_model() -> MaskedLM:
    """输入: 无。
    返回: MaskedLM。
    """
    return MaskedLM()
