"""视觉语言的合成样本。"""

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class VisionLanguageDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """图像亮暗决定最后词元。"""

    def __init__(self, count: int = 96) -> None:
        """输入：样本数；返回：初始化模块。"""
        self.count = count

    def __len__(self) -> int:
        """输入: 无。
        返回: 样本数量。
        """
        return self.count

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        """输入: 样本索引。
        返回: image[3,8,8], ids[3], target[]。
        """
        bright = index % 2 == 0
        image = torch.full((3, 8, 8), 1.0 if bright else -1.0)
        ids = torch.tensor([3, 4, 5], dtype=torch.long)
        return image, ids, torch.tensor(1 if bright else 2, dtype=torch.long)


def build_loaders(batch_size: int = 12) -> tuple[DataLoader, DataLoader]:
    """输入: batch_size。
    返回: train_loader, valid_loader。
    """
    return (
        DataLoader(VisionLanguageDataset(), batch_size=batch_size, shuffle=True),
        DataLoader(VisionLanguageDataset(24), batch_size=batch_size),
    )
