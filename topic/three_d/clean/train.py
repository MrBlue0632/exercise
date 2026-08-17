"""输入: NeRF模型和合成射线。
返回: 单步颜色重建损失。"""

import torch
from torch import Tensor
from torch.nn import functional as F

from data import make_rays
from model import TinyNeRF
from other import render_rays


def train_step(model: TinyNeRF, origins: Tensor, directions: Tensor, targets: Tensor, optimizer: torch.optim.Optimizer) -> float:
    """输入: 模型、光线和优化器。
    返回: 颜色重建标量损失。"""
    prediction = render_rays(model, origins, directions)
    loss = F.mse_loss(prediction, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def run_demo(steps: int = 3) -> TinyNeRF:
    """输入: 最小训练迭代次数。
    返回: 已训练的微型NeRF。"""
    model = TinyNeRF()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    origins, directions, targets = make_rays()
    for _ in range(steps):
        train_step(model, origins, directions, targets, optimizer)
    return model
