"""视频流匹配模型的通用工具。"""

from pathlib import Path
import math
import random

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def set_seed(seed: int) -> None:
    """输入：整数种子；返回：无。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """输入：无；返回：可用设备。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sinusoidal_embedding(value: Tensor, dim: int) -> Tensor:
    """输入：[B]标量、维度；返回：[B,D]正余弦编码。"""
    value = value.float().reshape(-1, 1)
    half = dim // 2
    scale = torch.arange(half, device=value.device, dtype=value.dtype)
    scale = torch.exp(-math.log(10000.0) * scale / max(half - 1, 1))
    embedding = torch.cat((torch.sin(value * scale), torch.cos(value * scale)), dim=1)
    return F.pad(embedding, (0, dim - embedding.size(1))) if embedding.size(1) < dim else embedding


def make_flow_pair(video: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
    """输入：真实视频、时间；返回：插值视频、速度目标。"""
    noise = torch.randn_like(video)
    shape = (video.size(0),) + (1,) * (video.ndim - 1)
    time = time.view(shape)
    return (1.0 - time) * noise + time * video, video - noise


@torch.no_grad()
def euler_sample(
    model: nn.Module, direction: Tensor, frames: int = 4, size: int = 16, steps: int = 16
) -> Tensor:
    """输入：模型、方向条件；返回：生成视频。"""
    sample = torch.randn(direction.size(0), 1, frames, size, size, device=direction.device)
    delta = 1.0 / steps
    for step in range(steps):
        time = torch.full((direction.size(0),), step * delta, device=direction.device)
        sample = sample + delta * model(sample, time, direction)
    return sample.clamp(0.0, 1.0)


def save_tensor(value: object, path: str) -> Path:
    """输入：可序列化对象、路径；返回：保存路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, target)
    return target
