"""评估逻辑。"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """输入模型、加载器和设备，返回分类准确率。"""
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        prediction = model(x.to(device)).argmax(dim=-1)
        correct += (prediction == y.to(device)).sum().item()
        total += y.numel()
    return correct / total


if __name__ == "__main__":
    from data import make_data, make_loader
    from model import CheckpointedClassifier

    print(f"accuracy={evaluate(CheckpointedClassifier(), make_loader(make_data()), torch.device('cpu')):.3f}")
