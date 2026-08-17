"""全注意力、MLA、线性注意力与 AttnRes。"""

import math

import torch
from torch import Tensor, nn

from other import causal_mask, layer_attention_residual, linear_causal_attention


class AttentionBase(nn.Module):
    """多头张量变形工具。"""

    def __init__(self, width: int = 48, heads: int = 4) -> None:
        """输入：隐藏维度和头数；返回：初始化模块。"""
        super().__init__()
        self.heads = heads
        self.head_width = width // heads

    def _split(self, value: Tensor) -> Tensor:
        """输入: value[B,T,D]。
        返回: value[B,H,T,Dh]。
        """
        batch, steps, _ = value.shape
        return value.view(batch, steps, self.heads, self.head_width).transpose(1, 2)

    def _join(self, value: Tensor) -> Tensor:
        """输入: value[B,H,T,Dh]。
        返回: value[B,T,D]。
        """
        return value.transpose(1, 2).contiguous().flatten(2)


class FullAttention(AttentionBase):
    """标准因果多头注意力。"""

    def __init__(self, width: int = 48) -> None:
        """输入：隐藏维度；返回：初始化模块。"""
        super().__init__(width)
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        """输入: hidden[B,T,D]。
        返回: output[B,T,D]。
        """
        query, key, value = (self._split(item) for item in self.qkv(hidden).chunk(3, dim=-1))
        score = query @ key.transpose(-2, -1) / math.sqrt(self.head_width)
        mask = causal_mask(hidden.size(1), hidden.device)
        score = score.masked_fill(~mask[None, None], torch.finfo(score.dtype).min)
        return self.output(self._join(torch.softmax(score, dim=-1) @ value))


class LatentAttention(AttentionBase):
    """以低维潜变量重建 KV。"""

    def __init__(self, width: int = 48, latent_width: int = 12) -> None:
        """输入：隐藏维度和潜变量维度；返回：初始化模块。"""
        super().__init__(width)
        self.query = nn.Linear(width, width, bias=False)
        self.down = nn.Linear(width, latent_width, bias=False)
        self.key_up = nn.Linear(latent_width, width, bias=False)
        self.value_up = nn.Linear(latent_width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        """输入: hidden[B,T,D]。
        返回: output[B,T,D]。
        """
        query = self._split(self.query(hidden))
        latent = self.down(hidden)
        key, value = self._split(self.key_up(latent)), self._split(self.value_up(latent))
        score = query @ key.transpose(-2, -1) / math.sqrt(self.head_width)
        mask = causal_mask(hidden.size(1), hidden.device)
        score = score.masked_fill(~mask[None, None], torch.finfo(score.dtype).min)
        return self.output(self._join(torch.softmax(score, dim=-1) @ value))


class LinearAttention(AttentionBase):
    """前缀状态式线性注意力。"""

    def __init__(self, width: int = 48) -> None:
        """输入：隐藏维度；返回：初始化模块。"""
        super().__init__(width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        """输入: hidden[B,T,D]。
        返回: output[B,T,D]。
        """
        output = linear_causal_attention(
            self._split(self.query(hidden)), self._split(self.key(hidden)), self._split(self.value(hidden))
        )
        return self.output(self._join(output))


class AttentionLM(nn.Module):
    """切换四种注意力写法。"""

    def __init__(self, kind: str = "mla", vocab_size: int = 32, width: int = 48) -> None:
        """输入：注意力类型、词表和隐藏维度；返回：初始化模块。"""
        super().__init__()
        self.kind = kind
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(32, width)
        table = {"full": FullAttention(width), "mla": LatentAttention(width), "linear": LinearAttention(width), "attnres": FullAttention(width)}
        if kind not in table:
            raise ValueError(f"unknown attention kind: {kind}")
        self.attention = table[kind]
        self.second = FullAttention(width) if kind == "attnres" else None
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, ids: Tensor) -> Tensor:
        """输入: ids[B,T]。
        返回: logits[B,T,V]。
        """
        position = torch.arange(ids.size(1), device=ids.device)
        hidden = self.token(ids) + self.position(position)[None]
        first = self.attention(hidden)
        if self.second is not None:
            second = self.second(hidden + first)
            hidden = layer_attention_residual(second, torch.stack((hidden, first), dim=1))
        else:
            hidden = hidden + first
        return self.head(self.norm(hidden))


def build_model(kind: str = "mla") -> AttentionLM:
    """输入: kind。
    返回: AttentionLM。
    """
    return AttentionLM(kind)
