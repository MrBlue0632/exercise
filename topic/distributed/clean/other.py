"""分布式初始化工具。"""

from __future__ import annotations

import os
import random

import torch
import torch.distributed as dist
from torch import nn


def setup_distributed() -> tuple[int, int, int, torch.device]:
    """输入环境变量，返回 rank、world size、local rank 和设备。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    """输入为空，返回无；销毁已创建的进程组。"""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int = 42) -> None:
    """输入种子，返回无；固定本进程随机性。"""
    random.seed(seed)
    torch.manual_seed(seed)


def unwrap_model(model: nn.Module) -> nn.Module:
    """输入普通模型或 DDP 模型，返回底层模型。"""
    return getattr(model, "module", model)


if __name__ == "__main__":
    print(setup_distributed())
    cleanup_distributed()
