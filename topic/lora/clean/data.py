"""LoRA 的两阶段分类数据。"""

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class PlaneDataset(Dataset[tuple[Tensor, Tensor]]):
    """两个不同决策边界的平面数据。"""

    def __init__(self, target_task: bool = False, count: int = 128, seed: int = 0) -> None:
        """输入：任务开关、样本数和种子；返回：初始化模块。"""
        generator = torch.Generator().manual_seed(seed)
        self.values = torch.randn(count, 2, generator=generator)
        score = self.values[:, 0] - self.values[:, 1] if target_task else self.values.sum(dim=1)
        self.labels = (score > 0).long()

    def __len__(self) -> int:
        """输入: 无。
        返回: 样本数量。
        """
        return self.labels.numel()

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """输入: 样本索引。
        返回: value[2], label[]。
        """
        return self.values[index], self.labels[index]


def build_loaders(target_task: bool, batch_size: int = 16) -> tuple[DataLoader, DataLoader]:
    """输入: target_task 与 batch_size。
    返回: train_loader, valid_loader。
    """
    train = DataLoader(PlaneDataset(target_task, seed=1), batch_size=batch_size, shuffle=True)
    valid = DataLoader(PlaneDataset(target_task, count=48, seed=2), batch_size=batch_size)
    return train, valid
