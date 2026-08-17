"""输入: 样本数量与随机种子。
返回: 配对图文数据集。"""

import torch
from torch import Tensor
from torch.utils.data import Dataset


class VisionTextDataset(Dataset[tuple[Tensor, Tensor]]):
    """输入: 样本数量和类别数。
    返回: 图像词元配对样本。"""

    def __init__(self, size: int = 64, classes: int = 8, seed: int = 0) -> None:
        """输入：数量类别与种子；返回：初始化数据。"""
        generator = torch.Generator().manual_seed(seed)
        labels = torch.randint(classes, (size,), generator=generator)
        images = torch.randn(size, 3, 16, 16, generator=generator)
        self.images = images + labels[:, None, None, None].float() / classes
        self.tokens = labels[:, None].repeat(1, 4)

    def __len__(self) -> int:
        """输入: 无额外输入。
        返回: 数据集样本数量。"""
        return self.images.size(0)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """输入: 单个样本索引。
        返回: 图像和词元张量。"""
        return self.images[index], self.tokens[index]


def make_batch(size: int = 8) -> tuple[Tensor, Tensor]:
    """输入: 需要生成的样本数。
    返回: 一批图像词元样本。"""
    dataset = VisionTextDataset(size=size)
    return dataset.images, dataset.tokens
