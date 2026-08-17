"""训练数据。"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


def make_regression_data(num_samples: int = 512, noise: float = 0.08) -> TensorDataset:
    """输入样本数和噪声，返回 (特征, 标签) 数据集。"""
    x = torch.linspace(-3.0, 3.0, num_samples).unsqueeze(1)
    y = torch.sin(x) + 0.3 * x + noise * torch.randn_like(x)
    return TensorDataset(x, y)


def make_loader(dataset: TensorDataset, batch_size: int = 64) -> DataLoader:
    """输入数据集和批大小，返回打乱后的 DataLoader。"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    batch = next(iter(make_loader(make_regression_data())))
    print(tuple(t.shape for t in batch))
