"""Llama 与 VLM 的公共工具。"""

import torch
from torch import Tensor


def rope_cache(steps: int, width: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """输入: steps 与偶数 width。
    返回: cos[T,D/2], sin[T,D/2]。
    """
    inv_freq = 1.0 / (10000 ** (torch.arange(0, width, 2, device=device) / width))
    phase = torch.arange(steps, device=device)[:, None] * inv_freq[None, :]
    return phase.cos(), phase.sin()


def apply_rope(value: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """输入: value[B,H,T,D], cos、sin。
    返回: 旋转后的 value[B,H,T,D]。
    """
    even, odd = value[..., 0::2], value[..., 1::2]
    rotated = torch.stack((even * cos[None, None] - odd * sin[None, None], even * sin[None, None] + odd * cos[None, None]), dim=-1)
    return rotated.flatten(-2)


def make_vlm_labels(ids: Tensor, targets: Tensor, vision_tokens: int) -> Tensor:
    """输入: ids[B,T], targets[B], vision_tokens。
    返回: labels[B,P+T]。
    """
    labels = torch.full((ids.size(0), vision_tokens + ids.size(1)), -100, device=ids.device, dtype=torch.long)
    labels[:, -1] = targets
    return labels
