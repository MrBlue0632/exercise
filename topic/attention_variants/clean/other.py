"""注意力变体的公共计算。"""

import math

import torch
from torch import Tensor


def causal_mask(steps: int, device: torch.device) -> Tensor:
    """输入: steps。
    返回: allow[T,T]。
    """
    index = torch.arange(steps, device=device)
    return index[None] <= index[:, None]


def linear_causal_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """输入: q,k,v[B,H,T,D]。
    返回: output[B,H,T,D]。
    """
    batch, heads, steps, width = query.shape
    state = query.new_zeros(batch, heads, width, width)
    normalizer = query.new_zeros(batch, heads, width)
    outputs: list[Tensor] = []
    for step in range(steps):
        key_t = torch.nn.functional.elu(key[:, :, step]) + 1.0
        query_t = torch.nn.functional.elu(query[:, :, step]) + 1.0
        value_t = value[:, :, step]

        # 累积线性注意力状态
        state = state + torch.einsum("bhd,bhe->bhde", key_t, value_t)
        normalizer = normalizer + key_t
        numerator = torch.einsum("bhd,bhde->bhe", query_t, state)
        denominator = (query_t * normalizer).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        outputs.append(numerator / denominator)
    return torch.stack(outputs, dim=2)


def layer_attention_residual(query: Tensor, history: Tensor) -> Tensor:
    """输入: query[B,T,D], history[B,L,T,D]。
    返回: mixed[B,T,D]。
    """
    score = torch.einsum("btd,bltd->blt", query, history) / math.sqrt(query.size(-1))
    weight = torch.softmax(score, dim=1)
    return query + (weight[..., None] * history).sum(dim=1)
