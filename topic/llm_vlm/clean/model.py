"""最小 Llama 与视觉语言模型。"""

import math

import torch
from torch import Tensor, nn

from other import apply_rope, rope_cache


class RMSNorm(nn.Module):
    """Llama 使用的均方根归一化。"""

    def __init__(self, width: int, eps: float = 1e-5) -> None:
        """输入：隐藏维度和稳定项；返回：初始化模块。"""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, value: Tensor) -> Tensor:
        """输入: value[...,D]。
        返回: normalized[...,D]。
        """
        scale = torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return value * scale * self.weight


class GroupedRotaryAttention(nn.Module):
    """RoPE 与 GQA 的因果注意力。"""

    def __init__(self, width: int, heads: int = 4, kv_heads: int = 2) -> None:
        """输入：隐藏维度、查询头和 KV 头；返回：初始化模块。"""
        super().__init__()
        self.heads, self.kv_heads = heads, kv_heads
        self.head_width = width // heads
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

    def forward(self, hidden: Tensor) -> Tensor:
        """输入: hidden[B,T,D]。
        返回: output[B,T,D]。
        """
        query = self._shape(self.query(hidden), self.heads)
        key = self._shape(self.key(hidden), self.kv_heads)
        value = self._shape(self.value(hidden), self.kv_heads)
        cos, sin = rope_cache(hidden.size(1), self.head_width, hidden.device)
        query, key = apply_rope(query, cos, sin), apply_rope(key, cos, sin)

        # 扩展组查询共享键值
        groups = self.heads // self.kv_heads
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        score = query @ key.transpose(-2, -1) / math.sqrt(self.head_width)
        mask = torch.ones(hidden.size(1), hidden.size(1), device=hidden.device, dtype=torch.bool).tril()
        score = score.masked_fill(~mask[None, None], torch.finfo(score.dtype).min)
        output = torch.softmax(score, dim=-1) @ value
        return self.output(output.transpose(1, 2).contiguous().view_as(hidden))


class SwiGLU(nn.Module):
    """Llama 风格的门控前馈层。"""

    def __init__(self, width: int, hidden: int = 96) -> None:
        """输入：隐藏维度和中间维度；返回：初始化模块。"""
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.value = nn.Linear(width, hidden, bias=False)
        self.output = nn.Linear(hidden, width, bias=False)

    def forward(self, value: Tensor) -> Tensor:
        """输入: value[B,T,D]。
        返回: output[B,T,D]。
        """
        return self.output(nn.functional.silu(self.gate(value)) * self.value(value))


class LlamaBlock(nn.Module):
    """预归一化的 Llama 块。"""

    def __init__(self, width: int) -> None:
        """输入：隐藏维度；返回：初始化模块。"""
        super().__init__()
        self.attn_norm = RMSNorm(width)
        self.ffn_norm = RMSNorm(width)
        self.attention = GroupedRotaryAttention(width)
        self.feed_forward = SwiGLU(width)

    def forward(self, hidden: Tensor) -> Tensor:
        """输入: hidden[B,T,D]。
        返回: hidden[B,T,D]。
        """
        hidden = hidden + self.attention(self.attn_norm(hidden))
        return hidden + self.feed_forward(self.ffn_norm(hidden))


class TinyLlama(nn.Module):
    """含 RoPE、GQA、SwiGLU 的 LLM。"""

    def __init__(self, vocab_size: int = 16, width: int = 48) -> None:
        """输入：词表大小和隐藏维度；返回：初始化模块。"""
        super().__init__()
        self.token = nn.Embedding(vocab_size, width)
        self.block = LlamaBlock(width)
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)

    def forward_embeddings(self, hidden: Tensor) -> Tensor:
        """输入: hidden[B,T,D]。
        返回: logits[B,T,V]。
        """
        return self.head(self.norm(self.block(hidden)))

    def forward(self, ids: Tensor) -> Tensor:
        """输入: ids[B,T]。
        返回: logits[B,T,V]。
        """
        return self.forward_embeddings(self.token(ids))


class TinyVLM(nn.Module):
    """图像前缀接入 TinyLlama。"""

    def __init__(self, vocab_size: int = 16, width: int = 48) -> None:
        """输入：词表大小和隐藏维度；返回：初始化模块。"""
        super().__init__()
        self.image_encoder = nn.Conv2d(3, width, kernel_size=4, stride=4)
        self.text_model = TinyLlama(vocab_size, width)
        self.vision_tokens = 4

    def forward(self, images: Tensor, ids: Tensor) -> Tensor:
        """输入: images[B,3,8,8], ids[B,T]。
        返回: logits[B,P+T,V]。
        """
        visual = self.image_encoder(images).flatten(2).transpose(1, 2)
        text = self.text_model.token(ids)
        return self.text_model.forward_embeddings(torch.cat((visual, text), dim=1))


def build_model() -> TinyVLM:
    """输入: 无。
    返回: TinyVLM。
    """
    return TinyVLM()
