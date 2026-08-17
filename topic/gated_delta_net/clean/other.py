"""Gated Delta Rule 的朴素递归。"""

import torch
from torch import Tensor


def gated_delta_rule(query: Tensor, key: Tensor, value: Tensor, decay: Tensor, beta: Tensor) -> tuple[Tensor, Tensor]:
    """输入: q,k,v[B,H,T,D] 与门[B,H,T]。
    返回: output[B,H,T,D], state[B,H,D,D]。
    """
    batch, heads, steps, width = query.shape
    state = query.new_zeros(batch, heads, width, width)
    outputs: list[Tensor] = []
    for step in range(steps):
        key_t, value_t = key[:, :, step], value[:, :, step]
        prediction = torch.einsum("bhde,bhe->bhd", state, key_t)
        correction = value_t - prediction

        # 衰减旧状态并写入误差
        update = torch.einsum("bhd,bhe->bhde", correction, key_t)
        state = decay[:, :, step, None, None] * state + beta[:, :, step, None, None] * update
        outputs.append(torch.einsum("bhde,bhe->bhd", state, query[:, :, step]))
    return torch.stack(outputs, dim=2), state
