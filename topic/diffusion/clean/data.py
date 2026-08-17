"""扩散模型使用的带类别玩具图像。"""

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class ToyShapesDataset(Dataset):
    """生成四类低分辨率图形。"""

    def __init__(self, count: int = 256, image_size: int = 16, seed: int = 0):
        """输入：样本数、尺寸、种子；返回：数据集。"""
        if image_size < 12:
            raise ValueError("image_size 至少为 12")
        self.count = count
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        """输入：无；返回：样本数量。"""
        return self.count

    def _draw(self, label: int, generator: torch.Generator) -> Tensor:
        """输入：类别、随机器；返回：单通道图像。"""
        size = self.image_size
        image = torch.zeros(1, size, size)
        shift = torch.randint(-2, 3, (2,), generator=generator)
        cy = size // 2 + int(shift[0])
        cx = size // 2 + int(shift[1])

        # 绘制条件对应的图形
        if label == 0:
            image[0, cy, 3 : size - 3] = 1.0
        elif label == 1:
            image[0, 3 : size - 3, cx] = 1.0
        elif label == 2:
            row = torch.arange(3, size - 3)
            col = (row - 3 + cx - 4).clamp(0, size - 1)
            image[0, row, col] = 1.0
        else:
            image[0, cy - 2 : cy + 3, cx - 2 : cx + 3] = 1.0
        noise = torch.randn(image.shape, generator=generator) * 0.03
        return (image + noise).clamp(0.0, 1.0)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """输入：样本索引；返回：图像、类别。"""
        generator = torch.Generator().manual_seed(self.seed + index)
        label = index % 4
        return self._draw(label, generator), torch.tensor(label, dtype=torch.long)


def build_loader(batch_size: int = 16, count: int = 256, seed: int = 0) -> DataLoader:
    """输入：批量参数；返回：图像加载器。"""
    dataset = ToyShapesDataset(count=count, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=generator)
