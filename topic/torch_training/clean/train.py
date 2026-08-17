"""最小 PyTorch 训练循环。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import make_loader, make_regression_data
from eval import evaluate
from model import RegressionModel
from other import get_device, parameter_count, set_seed


@dataclass
class TrainConfig:
    """训练超参数。"""

    epochs: int = 80
    learning_rate: float = 3e-3


def train(model: nn.Module, loader: DataLoader, device: torch.device, config: TrainConfig) -> list[float]:
    """输入模型、数据、设备和配置，返回每轮平均训练损失。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[float] = []
    model.train()
    for _ in range(config.epochs):
        total_loss = 0.0
        for x, y in loader:
            prediction = model(x.to(device))
            loss = torch.nn.functional.mse_loss(prediction, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        history.append(total_loss / len(loader))
    return history


def main() -> None:
    """输入为空，返回无；训练并评估一个最小回归模型。"""
    set_seed()
    device = get_device()
    loader = make_loader(make_regression_data())
    model = RegressionModel().to(device)
    losses = train(model, loader, device, TrainConfig())
    mse = evaluate(model, loader, device)
    print(f"device={device}, params={parameter_count(model)}, loss={losses[-1]:.4f}, mse={mse:.4f}")


if __name__ == "__main__":
    main()
