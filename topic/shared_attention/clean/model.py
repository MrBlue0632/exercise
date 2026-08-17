"""GQA 与共享前缀缓存。"""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from other import causal_mask, repeat_kv


class GroupedQueryAttention(nn.Module):
    """多个查询头共享较少 KV 头。"""

    def __init__(self, width: int = 48, query_heads: int = 4, kv_heads: int = 2) -> None:
        """输入：隐藏维度、查询头和 KV 头；返回：初始化模块。"""
        super().__init__()
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.head_width = width // query_heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, kv_heads * self.head_width, bias=False)
        self.value = nn.Linear(width, kv_heads * self.head_width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def _shape(self, value: Tensor, heads: int) -> Tensor:
        """输入: value[B,T,D] 与 heads。
        返回: value[B,H,T,Dh]。
        """
        batch, steps, _ = value.shape
        return value.view(batch, steps, heads, self.head_width).transpose(1, 2)

    def forward(
        self, hidden: Tensor, cache: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """输入: hidden[B,T,D], cache。
        返回: output[B,T,D], cache。
        """
        query = self._shape(self.query(hidden), self.query_heads)
        key = self._shape(self.key(hidden), self.kv_heads)
        value = self._shape(self.value(hidden), self.kv_heads)
        past = 0 if cache is None else cache[0].size(2)

        # 保存未复制的 KV
        if cache is not None:
            key = torch.cat((cache[0], key), dim=2)
            value = torch.cat((cache[1], value), dim=2)
        groups = self.query_heads // self.kv_heads
        shared_key = repeat_kv(key, groups)
        shared_value = repeat_kv(value, groups)

        score = query @ shared_key.transpose(-2, -1) / math.sqrt(self.head_width)
        mask = causal_mask(query.size(2), key.size(2), past, hidden.device)
        score = score.masked_fill(~mask[None, None], torch.finfo(score.dtype).min)
        output = torch.softmax(score, dim=-1) @ shared_value
        output = output.transpose(1, 2).contiguous().view_as(hidden)
        return self.output(output), (key, value)


class SharedAttentionLM(nn.Module):
    """用 GQA 的小语言模型。"""

    def __init__(self, vocab_size: int = 32, width: int = 48) -> None:
        """输入：词表大小和隐藏维度；返回：初始化模块。"""
        super().__init__()
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(64, width)
        self.attention = GroupedQueryAttention(width)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size)

    def forward(
        self, ids: Tensor, cache: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """输入: ids[B,T], cache。
        返回: logits[B,T,V], cache。
        """
        start = 0 if cache is None else cache[0].size(2)
        position = torch.arange(start, start + ids.size(1), device=ids.device)
        hidden = self.token(ids) + self.position(position)[None]
        attended, next_cache = self.attention(hidden, cache)
        return self.head(self.norm(hidden + attended)), next_cache


def build_model() -> SharedAttentionLM:
    """输入: 无。
    返回: SharedAttentionLM。
    """
    return SharedAttentionLM()
