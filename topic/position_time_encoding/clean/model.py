"""位置与时间编码的最小模型。"""

import math

import torch
from torch import Tensor, nn

from other import apply_rope, rope_cache, sinusoidal_positions, timestep_embedding


class PositionAttention(nn.Module):
    """可选 RoPE 的因果注意力。"""

    def __init__(self, width: int = 48, heads: int = 4, use_rope: bool = False) -> None:
        """输入：隐藏维度、头数和 RoPE 开关；返回：初始化模块。"""
        super().__init__()
        self.heads, self.head_width, self.use_rope = heads, width // heads, use_rope
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def _split(self, value: Tensor) -> Tensor:
        """输入: value[B,T,D]。
        返回: value[B,H,T,Dh]。
        """
        batch, steps, _ = value.shape
        return value.view(batch, steps, self.heads, self.head_width).transpose(1, 2)

    def forward(self, hidden: Tensor) -> Tensor:
        """输入: hidden[B,T,D]。
        返回: output[B,T,D]。
        """
        query, key, value = (self._split(item) for item in self.qkv(hidden).chunk(3, dim=-1))
        if self.use_rope:
            cos, sin = rope_cache(hidden.size(1), self.head_width, hidden.device)
            query, key = apply_rope(query, cos, sin), apply_rope(key, cos, sin)
        score = query @ key.transpose(-2, -1) / math.sqrt(self.head_width)
        mask = torch.ones(hidden.size(1), hidden.size(1), device=hidden.device, dtype=torch.bool).tril()
        score = score.masked_fill(~mask[None, None], torch.finfo(score.dtype).min)
        output = torch.softmax(score, dim=-1) @ value
        return self.output(output.transpose(1, 2).contiguous().flatten(2))


class PositionLM(nn.Module):
    """对照绝对位置与 RoPE。"""

    def __init__(self, mode: str = "rope", vocab_size: int = 32, width: int = 48) -> None:
        """输入：位置模式、词表大小和隐藏维度；返回：初始化模块。"""
        super().__init__()
        if mode not in ("absolute", "rope"):
            raise ValueError(f"unknown position mode: {mode}")
        self.mode, self.width = mode, width
        self.token = nn.Embedding(vocab_size, width)
        self.absolute = nn.Embedding(32, width)
        self.attention = PositionAttention(width, use_rope=mode == "rope")
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, ids: Tensor) -> Tensor:
        """输入: ids[B,T]。
        返回: logits[B,T,V]。
        """
        hidden = self.token(ids)
        if self.mode == "absolute":
            position = torch.arange(ids.size(1), device=ids.device)
            hidden = hidden + self.absolute(position)[None]
        else:
            hidden = hidden + sinusoidal_positions(ids.size(1), self.width, ids.device)[None]
        return self.head(self.norm(hidden + self.attention(hidden)))


class TimeDenoiser(nn.Module):
    """时间条件的噪声预测器。"""

    def __init__(self, width: int = 16) -> None:
        """输入：特征维度；返回：初始化模块。"""
        super().__init__()
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.net = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.width = width

    def forward(self, noisy: Tensor, times: Tensor) -> Tensor:
        """输入: noisy[B,D], times[B]。
        返回: predicted_noise[B,D]。
        """
        return self.net(noisy + self.time(timestep_embedding(times, self.width)))


def build_models() -> tuple[PositionLM, PositionLM, TimeDenoiser]:
    """输入: 无。
    返回: absolute、rope、denoiser。
    """
    return PositionLM("absolute"), PositionLM("rope"), TimeDenoiser()
