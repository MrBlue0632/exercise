"""输入: 线性模型和回归样本。
返回: 单步训练损失。"""

import torch
from torch import Tensor
from torch.nn import functional as F

from data import make_batch
from model import FusedLinear


def train_step(model: FusedLinear, values: Tensor, target: Tensor, optimizer: torch.optim.Optimizer) -> float:
    """输入: 模型、样本和优化器。
    返回: 当前均方误差。"""
    prediction = model(values)
    loss = F.mse_loss(prediction, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def run_demo(steps: int = 3) -> FusedLinear:
    """输入: 最小训练迭代数。
    返回: 已训练线性模型。"""
    model = FusedLinear()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    values, target = make_batch()
    for _ in range(steps):
        train_step(model, values, target, optimizer)
    return model
