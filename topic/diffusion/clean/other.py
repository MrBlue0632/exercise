"""扩散与流匹配的采样工具。"""

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


def timestep_embedding(time: Tensor, dim: int) -> Tensor:
    """输入：[B]时间、维度；返回：[B,D]正余弦编码。"""
    time = time.float().reshape(-1, 1)
    half = dim // 2
    scale = torch.arange(half, device=time.device, dtype=time.dtype)
    scale = torch.exp(-math.log(10000.0) * scale / max(half - 1, 1))
    embedding = torch.cat((torch.sin(time * scale), torch.cos(time * scale)), dim=1)
    return F.pad(embedding, (0, dim - embedding.size(1))) if embedding.size(1) < dim else embedding


def linear_betas(steps: int, device: torch.device) -> Tensor:
    """输入：扩散步数、设备；返回：[T]噪声方差。"""
    return torch.linspace(1e-4, 2e-2, steps, device=device)


def q_sample(clean: Tensor, index: Tensor, betas: Tensor, noise: Tensor | None = None) -> Tensor:
    """输入：干净样本、步号、方差；返回：加噪样本。"""
    noise = torch.randn_like(clean) if noise is None else noise
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    shape = (clean.size(0),) + (1,) * (clean.ndim - 1)
    signal = alpha_bar[index].sqrt().view(shape)
    random_part = (1.0 - alpha_bar[index]).sqrt().view(shape)
    return signal * clean + random_part * noise


def make_flow_pair(data: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
    """输入：数据、连续时间；返回：插值点、速度目标。"""
    noise = torch.randn_like(data)
    shape = (data.size(0),) + (1,) * (data.ndim - 1)
    time = time.view(shape)
    return (1.0 - time) * noise + time * data, data - noise


@torch.no_grad()
def sample_ddpm(model: nn.Module, shape: tuple[int, ...], label: Tensor, betas: Tensor) -> Tensor:
    """输入：噪声模型、形状、类别；返回：DDPM 样本。"""
    sample = torch.randn(shape, device=label.device)
    alpha = 1.0 - betas
    alpha_bar = torch.cumprod(alpha, dim=0)

    # 从最后一步逐步去噪
    for index in reversed(range(len(betas))):
        time = torch.full((shape[0],), index / max(len(betas) - 1, 1), device=label.device)
        prediction = model(sample, time, label)
        beta = betas[index]
        mean = (sample - beta * prediction / (1.0 - alpha_bar[index]).sqrt()) / alpha[index].sqrt()
        if index > 0:
            variance = beta * (1.0 - alpha_bar[index - 1]) / (1.0 - alpha_bar[index])
            sample = mean + variance.sqrt() * torch.randn_like(sample)
        else:
            sample = mean
    return sample.clamp(0.0, 1.0)


@torch.no_grad()
def sample_flow(model: nn.Module, shape: tuple[int, ...], label: Tensor, steps: int = 16) -> Tensor:
    """输入：速度模型、形状、类别；返回：流匹配样本。"""
    sample = torch.randn(shape, device=label.device)
    delta = 1.0 / steps
    for index in range(steps):
        time = torch.full((shape[0],), index * delta, device=label.device)
        sample = sample + delta * model(sample, time, label)
    return sample.clamp(0.0, 1.0)


def save_tensor(value: object, path: str) -> Path:
    """输入：可序列化对象、路径；返回：保存路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, target)
    return target
