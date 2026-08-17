"""DDP 数据切分。"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


def make_classification_data(num_samples: int = 512) -> TensorDataset:
    """输入样本数，返回可线性分割的二维分类数据。"""
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(num_samples, 2, generator=generator)
    y = (x[:, 0] + 0.7 * x[:, 1] > 0).long()
    return TensorDataset(x, y)


def make_loader(
    dataset: TensorDataset, batch_size: int, distributed: bool
) -> tuple[DataLoader, Optional[DistributedSampler]]:
    """输入数据、批大小和 DDP 开关，返回加载器及可选采样器。"""
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler)
    return loader, sampler


if __name__ == "__main__":
    print(len(make_classification_data()))
