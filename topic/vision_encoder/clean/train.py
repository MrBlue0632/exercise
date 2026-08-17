"""输入: 图文模型和训练批次。
返回: 单步对比学习损失。"""

import torch
from torch import Tensor
from torch.nn import functional as F

from data import make_batch
from model import TinyClip


def contrastive_loss(image_features: Tensor, text_features: Tensor, scale: Tensor) -> Tensor:
    """输入: 两路特征与缩放值。
    返回: 对称对比学习损失。"""
    logits = image_features @ text_features.T * scale.exp()
    labels = torch.arange(logits.size(0), device=logits.device)
    # 同时监督两个检索方向
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def train_step(model: TinyClip, images: Tensor, tokens: Tensor, optimizer: torch.optim.Optimizer) -> float:
    """输入: 模型、批次和优化器。
    返回: 当前批次标量损失。"""
    image_features, text_features = model(images, tokens)
    loss = contrastive_loss(image_features, text_features, model.logit_scale)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def run_demo(steps: int = 3) -> TinyClip:
    """输入: 最小训练迭代次数。
    返回: 已训练演示模型。"""
    model = TinyClip()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    images, tokens = make_batch()
    for _ in range(steps):
        train_step(model, images, tokens, optimizer)
    return model
