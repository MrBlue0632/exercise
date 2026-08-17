"""注意力掩码工具。"""

import torch
from torch import Tensor


def causal_mask(steps: int, device: torch.device) -> Tensor:
    """输入: steps。
    返回: allow[T,T]。
    """
    index = torch.arange(steps, device=device)
    return index[None, :] <= index[:, None]


def padding_mask(lengths: Tensor, steps: int) -> Tensor:
    """输入: lengths[B], steps。
    返回: valid[B,T]。
    """
    return torch.arange(steps, device=lengths.device)[None] < lengths[:, None]


def prefix_lm_mask(steps: int, prefix: int, device: torch.device) -> Tensor:
    """输入: steps 与 prefix。
    返回: allow[T,T]。
    """
    query = torch.arange(steps, device=device)[:, None]
    key = torch.arange(steps, device=device)[None, :]
    prefix_view = (query < prefix) & (key < prefix)
    suffix_view = (query >= prefix) & (key <= query)
    return prefix_view | suffix_view


def combine_masks(sequence: Tensor, valid: Tensor) -> Tensor:
    """输入: sequence[T,T], valid[B,T]。
    返回: allow[B,T,T]。
    """
    allowed = sequence[None] & valid[:, None, :]
    eye = torch.eye(sequence.size(0), device=sequence.device, dtype=torch.bool)[None]
    return allowed | ((~valid)[:, :, None] & eye)
