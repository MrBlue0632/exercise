"""位置编码与时间编码函数。"""

import math

import torch
from torch import Tensor


def sinusoidal_positions(steps: int, width: int, device: torch.device) -> Tensor:
    """输入: steps 与偶数 width。
    返回: position[T,D]。
    """
    rate = torch.exp(-math.log(10000.0) * torch.arange(0, width, 2, device=device) / width)
    phase = torch.arange(steps, device=device)[:, None] * rate[None]
    output = torch.zeros(steps, width, device=device)
    output[:, 0::2], output[:, 1::2] = phase.sin(), phase.cos()
    return output


def rope_cache(steps: int, width: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """输入: steps 与偶数 width。
    返回: cos[T,D/2], sin[T,D/2]。
    """
    rate = 1.0 / (10000 ** (torch.arange(0, width, 2, device=device) / width))
    phase = torch.arange(steps, device=device)[:, None] * rate[None]
    return phase.cos(), phase.sin()


def apply_rope(value: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """输入: value[B,H,T,D], cos、sin。
    返回: rotated[B,H,T,D]。
    """
    even, odd = value[..., 0::2], value[..., 1::2]
    output = torch.stack((even * cos[None, None] - odd * sin[None, None], even * sin[None, None] + odd * cos[None, None]), dim=-1)
    return output.flatten(-2)


def timestep_embedding(times: Tensor, width: int) -> Tensor:
    """输入: times[B], width。
    返回: embedding[B,D]。
    """
    rate = torch.exp(-math.log(10000.0) * torch.arange(0, width, 2, device=times.device) / width)
    phase = times.float()[:, None] * rate[None]
    return torch.cat((phase.sin(), phase.cos()), dim=-1)
