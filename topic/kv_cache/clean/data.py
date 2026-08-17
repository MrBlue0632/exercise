"""KV Cache 的玩具数据。"""

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class PatternDataset(Dataset[tuple[Tensor, Tensor]]):
    """循环词元的下一个词预测。"""

    def __init__(self, count: int = 96, steps: int = 12, vocab_size: int = 32) -> None:
        """输入：样本数、长度和词表大小；返回：初始化模块。"""
        self.count = count
        self.steps = steps
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        """输入: 无。
        返回: 样本数量。
        """
        return self.count

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """输入: 样本索引。
        返回: ids[T], targets[T]。
        """
        tokens = (torch.arange(self.steps + 1) + index * 3) % self.vocab_size
        return tokens[:-1], tokens[1:]


def build_loaders(batch_size: int = 12) -> tuple[DataLoader, DataLoader]:
    """输入: batch_size。
    返回: train_loader, valid_loader。
    """
    train = DataLoader(PatternDataset(), batch_size=batch_size, shuffle=True)
    valid = DataLoader(PatternDataset(24), batch_size=batch_size)
    return train, valid
