"""激活重计算训练入口。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import make_data, make_loader
from eval import evaluate
from model import CheckpointedClassifier
from other import get_device, parameter_count, set_seed


@dataclass
class TrainConfig:
    """训练超参数。"""

    epochs: int = 18
    learning_rate: float = 4e-3


def train(model: nn.Module, loader: DataLoader, device: torch.device, config: TrainConfig) -> float:
    """输入模型、数据、设备和配置，返回最后一轮平均损失。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    final_loss = 0.0
    model.train()
    for _ in range(config.epochs):
        total_loss = 0.0
        for x, y in loader:
            logits = model(x.to(device))
            loss = torch.nn.functional.cross_entropy(logits, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        final_loss = total_loss / len(loader)
    return final_loss


def main() -> None:
    """输入为空，返回无；训练启用 checkpoint 的模型。"""
    set_seed()
    device = get_device()
    loader = make_loader(make_data())
    model = CheckpointedClassifier(use_checkpoint=True).to(device)
    loss = train(model, loader, device, TrainConfig())
    accuracy = evaluate(model, loader, device)
    print(f"checkpoint={model.use_checkpoint}, params={parameter_count(model)}, loss={loss:.4f}, accuracy={accuracy:.3f}")


if __name__ == "__main__":
    main()
