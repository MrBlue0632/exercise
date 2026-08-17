"""共享注意力的辅助函数。"""

import torch
from torch import Tensor


def causal_mask(query_steps: int, key_steps: int, past: int, device: torch.device) -> Tensor:
    """输入: query、key 与 past。
    返回: allow[Q,K]。
    """
    query = torch.arange(past, past + query_steps, device=device)[:, None]
    key = torch.arange(key_steps, device=device)[None, :]
    return key <= query


def repeat_kv(value: Tensor, groups: int) -> Tensor:
    """输入: value[B,Hkv,T,Dh]。
    返回: value[B,Hq,T,Dh]。
    """
    return value.repeat_interleave(groups, dim=1)


def expand_prefix_cache(cache: tuple[Tensor, Tensor], batch: int) -> tuple[Tensor, Tensor]:
    """输入: 单条 prefix cache。
    返回: 共享视图 cache[B]。
    """
    key, value = cache
    return key.expand(batch, *key.shape[1:]), value.expand(batch, *value.shape[1:])


def cache_nbytes(cache: tuple[Tensor, Tensor]) -> int:
    """输入: cache。
    返回: 缓存字节数。
    """
    return sum(item.numel() * item.element_size() for item in cache)
