"""分布式评估。"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """输入模型、加载器和设备，返回所有 rank 聚合后的准确率。"""
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        prediction = model(x.to(device)).argmax(dim=-1)
        correct += (prediction == y.to(device)).sum().item()
        total += y.numel()
    stats = torch.tensor([correct, total], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return (stats[0] / stats[1]).item()


if __name__ == "__main__":
    from data import make_classification_data, make_loader
    from model import Classifier

    loader, _ = make_loader(make_classification_data(), 64, distributed=False)
    print(f"accuracy={evaluate(Classifier(), loader, torch.device('cpu')):.3f}")
