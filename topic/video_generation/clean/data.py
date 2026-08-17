"""移动光点视频的最小条件数据集。"""

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class MovingDotsDataset(Dataset):
    """按四个方向生成短视频。"""

    def __init__(self, count: int = 256, frames: int = 4, size: int = 16, seed: int = 0):
        """输入：样本数、帧数、尺寸、种子；返回：数据集。"""
        self.count = count
        self.frames = frames
        self.size = size
        self.seed = seed

    def __len__(self) -> int:
        """输入：无；返回：样本数量。"""
        return self.count

    def _draw_video(self, index: int) -> tuple[Tensor, Tensor]:
        """输入：样本索引；返回：视频、方向类别。"""
        generator = torch.Generator().manual_seed(self.seed + index)
        direction = index % 4
        moves = ((1, 0), (-1, 0), (0, 1), (0, -1))
        dx, dy = moves[direction]
        start = torch.randint(0, self.size, (2,), generator=generator)
        video = torch.zeros(1, self.frames, self.size, self.size)

        # 沿指定方向移动亮点
        for frame in range(self.frames):
            x = (int(start[0]) + frame * dx) % self.size
            y = (int(start[1]) + frame * dy) % self.size
            video[0, frame, y, x] = 1.0
        noise = torch.randn(video.shape, generator=generator) * 0.02
        return (video + noise).clamp(0.0, 1.0), torch.tensor(direction, dtype=torch.long)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """输入：样本索引；返回：视频、方向类别。"""
        return self._draw_video(index)


def build_loader(
    batch_size: int = 8, count: int = 256, frames: int = 4, size: int = 16, seed: int = 0
) -> DataLoader:
    """输入：批量参数；返回：视频加载器。"""
    dataset = MovingDotsDataset(count=count, frames=frames, size=size, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=generator)
