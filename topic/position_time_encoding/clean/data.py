"""位置与时间编码的合成数据。"""

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class TokenDataset(Dataset[tuple[Tensor, Tensor]]):
    """位置敏感的循环序列。"""

    def __init__(self, count: int = 80, steps: int = 10, vocab_size: int = 32) -> None:
        """输入：样本数、长度和词表大小；返回：初始化模块。"""
        self.count, self.steps, self.vocab_size = count, steps, vocab_size

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


class NoiseDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """不同时间步的加噪向量。"""

    def __init__(self, count: int = 96, width: int = 16, seed: int = 4) -> None:
        """输入：样本数、特征维度和种子；返回：初始化模块。"""
        generator = torch.Generator().manual_seed(seed)
        self.clean = torch.randn(count, width, generator=generator)
        self.noise = torch.randn(count, width, generator=generator)
        self.times = torch.arange(count) % 10

    def __len__(self) -> int:
        """输入: 无。
        返回: 样本数量。
        """
        return self.times.numel()

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        """输入: 样本索引。
        返回: noisy[D], time[], noise[D]。
        """
        scale = (self.times[index].float() + 1.0) / 10.0
        return self.clean[index] + scale * self.noise[index], self.times[index], self.noise[index]


def build_token_loaders(batch_size: int = 10) -> tuple[DataLoader, DataLoader]:
    """输入: batch_size。
    返回: train_loader, valid_loader。
    """
    return DataLoader(TokenDataset(), batch_size=batch_size, shuffle=True), DataLoader(TokenDataset(20), batch_size=batch_size)


def build_noise_loaders(batch_size: int = 12) -> tuple[DataLoader, DataLoader]:
    """输入: batch_size。
    返回: train_loader, valid_loader。
    """
    return DataLoader(NoiseDataset(), batch_size=batch_size, shuffle=True), DataLoader(NoiseDataset(24), batch_size=batch_size)
