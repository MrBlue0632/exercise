"""激活重计算演示数据。"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


def make_data(num_samples: int = 768) -> TensorDataset:
    """输入样本数，返回八维二分类 TensorDataset。"""
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(num_samples, 8, generator=generator)
    y = (x[:, :4].sum(dim=-1) - x[:, 4:].sum(dim=-1) > 0).long()
    return TensorDataset(x, y)


def make_loader(dataset: TensorDataset, batch_size: int = 64) -> DataLoader:
    """输入数据集和批大小，返回打乱后的 DataLoader。"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    print(next(iter(make_loader(make_data())))[0].shape)
