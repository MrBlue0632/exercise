"""MoE 的路由统计函数。"""

import torch
from torch import Tensor


def load_balance_loss(router_probs: Tensor) -> Tensor:
    """输入: router_probs[B,T,E]。
    返回: 标量均衡损失。
    """
    mean_prob = router_probs.mean(dim=(0, 1))
    return router_probs.size(-1) * mean_prob.pow(2).sum()


def expert_histogram(indices: Tensor, experts: int) -> Tensor:
    """输入: indices[B,T,K], experts。
    返回: counts[E]。
    """
    return torch.bincount(indices.flatten(), minlength=experts)
