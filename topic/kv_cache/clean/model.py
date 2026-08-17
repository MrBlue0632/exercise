"""KV Cache 最小语言模型。"""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from other import causal_mask


class CachedAttention(nn.Module):
    """带 KV 缓存的因果注意力。"""

    def __init__(self, width: int, heads: int) -> None:
        """输入：隐藏维度和头数；返回：初始化模块。"""
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def _split_heads(self, value: Tensor) -> Tensor:
        """输入: value[B,T,D]。
        返回: value[B,H,T,Dh]。
        """
        batch, steps, width = value.shape
        value = value.view(batch, steps, self.heads, self.head_width)
        return value.transpose(1, 2)

    def forward(
        self, value: Tensor, cache: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """输入: value[B,T,D], cache。
        返回: output[B,T,D], cache。
        """
        query = self._split_heads(self.query(value))
        key = self._split_heads(self.key(value))
        item = self._split_heads(self.value(value))
        past = 0 if cache is None else cache[0].size(2)

        # 追加历史键和值
        if cache is not None:
            key = torch.cat((cache[0], key), dim=2)
            item = torch.cat((cache[1], item), dim=2)

        scale = math.sqrt(self.head_width)
        score = query @ key.transpose(-2, -1) / scale
        mask = causal_mask(query.size(2), key.size(2), past, value.device)
        score = score.masked_fill(~mask[None, None], torch.finfo(score.dtype).min)
        weight = torch.softmax(score, dim=-1)
        output = weight @ item
        output = output.transpose(1, 2).contiguous().view_as(value)
        return self.output(output), (key, item)


class TinyCachedLM(nn.Module):
    """可缓存的单层语言模型。"""

    def __init__(self, vocab_size: int = 32, width: int = 48, heads: int = 4) -> None:
        """输入：词表大小、隐藏维度和头数；返回：初始化模块。"""
        super().__init__()
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(64, width)
        self.attention = CachedAttention(width, heads)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size)

    def forward(
        self, ids: Tensor, cache: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """输入: ids[B,T], cache。
        返回: logits[B,T,V], cache。
        """
        start = 0 if cache is None else cache[0].size(2)
        positions = torch.arange(start, start + ids.size(1), device=ids.device)
        hidden = self.token(ids) + self.position(positions)[None]
        attended, next_cache = self.attention(hidden, cache)
        logits = self.head(self.norm(hidden + attended))
        return logits, next_cache


def build_model() -> TinyCachedLM:
    """输入: 无。
    返回: TinyCachedLM。
    """
    return TinyCachedLM()
