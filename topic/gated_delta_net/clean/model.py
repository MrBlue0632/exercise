"""Gated Delta Net 的最小语言模型。"""

import torch
from torch import Tensor, nn

from other import gated_delta_rule


class GatedDeltaLayer(nn.Module):
    """把注意力替换为 Delta Rule。"""

    def __init__(self, width: int = 48, heads: int = 4) -> None:
        """输入：隐藏维度和头数；返回：初始化模块。"""
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.decay = nn.Linear(width, heads)
        self.beta = nn.Linear(width, heads)
        self.output = nn.Linear(width, width, bias=False)

    def _split(self, value: Tensor) -> Tensor:
        """输入: value[B,T,D]。
        返回: value[B,H,T,Dh]。
        """
        batch, steps, _ = value.shape
        return value.view(batch, steps, self.heads, self.head_width).transpose(1, 2)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """输入: hidden[B,T,D]。
        返回: output[B,T,D], state[B,H,Dh,Dh]。
        """
        query = self._split(self.query(hidden))
        key = torch.nn.functional.normalize(self._split(self.key(hidden)), dim=-1)
        value = self._split(self.value(hidden))
        decay = torch.sigmoid(self.decay(hidden)).transpose(1, 2)
        beta = torch.sigmoid(self.beta(hidden)).transpose(1, 2)
        output, state = gated_delta_rule(query, key, value, decay, beta)
        output = output.transpose(1, 2).contiguous().view_as(hidden)
        return self.output(output), state


class GatedDeltaLM(nn.Module):
    """使用 Gated Delta 层的语言模型。"""

    def __init__(self, vocab_size: int = 32, width: int = 48) -> None:
        """输入：词表大小和隐藏维度；返回：初始化模块。"""
        super().__init__()
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(32, width)
        self.delta = GatedDeltaLayer(width)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        """输入: ids[B,T]。
        返回: logits[B,T,V], state[B,H,Dh,Dh]。
        """
        position = torch.arange(ids.size(1), device=ids.device)
        hidden = self.token(ids) + self.position(position)[None]
        update, state = self.delta(hidden)
        return self.head(self.norm(hidden + update)), state


def build_model() -> GatedDeltaLM:
    """输入: 无。
    返回: GatedDeltaLM。
    """
    return GatedDeltaLM()
