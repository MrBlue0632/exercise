"""评估逻辑。"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """输入模型、数据加载器和设备，返回平均 MSE。"""
    model.eval()
    total_loss = 0.0
    total_items = 0
    for x, y in loader:
        prediction = model(x.to(device))
        loss = torch.nn.functional.mse_loss(prediction, y.to(device), reduction="sum")
        total_loss += loss.item()
        total_items += y.numel()
    return total_loss / total_items


if __name__ == "__main__":
    from data import make_loader, make_regression_data
    from model import RegressionModel

    print(f"mse={evaluate(RegressionModel(), make_loader(make_regression_data()), torch.device('cpu')):.4f}")
