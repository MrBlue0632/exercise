"""变长序列与填充批处理。"""

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class VariableSequenceDataset(Dataset[tuple[Tensor, Tensor]]):
    """产生不同长度的循环序列。"""

    def __init__(self, count: int = 80, vocab_size: int = 32) -> None:
        """输入：样本数和词表大小；返回：初始化模块。"""
        self.count = count
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
        steps = 4 + index % 7
        tokens = 1 + (torch.arange(steps + 1) + index) % (self.vocab_size - 1)
        return tokens[:-1], tokens[1:]


def pad_collate(items: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor, Tensor]:
    """输入: 变长样本列表。
    返回: ids[B,T], targets[B,T], lengths[B]。
    """
    lengths = torch.tensor([ids.numel() for ids, _ in items])
    steps = int(lengths.max())
    ids = torch.zeros(len(items), steps, dtype=torch.long)
    targets = torch.full((len(items), steps), -100, dtype=torch.long)
    for row, (source, target) in enumerate(items):
        ids[row, : source.numel()] = source
        targets[row, : target.numel()] = target
    return ids, targets, lengths


def build_loaders(batch_size: int = 10) -> tuple[DataLoader, DataLoader]:
    """输入: batch_size。
    返回: train_loader, valid_loader。
    """
    train = DataLoader(VariableSequenceDataset(), batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    valid = DataLoader(VariableSequenceDataset(20), batch_size=batch_size, collate_fn=pad_collate)
    return train, valid
