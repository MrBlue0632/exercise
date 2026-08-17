"""生成并按索引读取二次函数样本。"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class QuadraticDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """保存 y=x² 的单变量监督样本。"""

    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """输入 [N, 1] 的 x、y，返回可索引数据集。"""
        if x.shape != y.shape or x.ndim != 2 or x.size(1) != 1:
            raise ValueError("x 和 y 必须同为 [N, 1]")
        self.x = x
        self.y = y

    def __len__(self) -> int:
        """输入为空，返回样本总数 N。"""
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """输入样本下标，返回 x、y 均为 [1]。"""
        return self.x[index], self.y[index]


def make_quadratic_data(
    num_samples: int = 256,
    x_range: tuple[float, float] = (-2.0, 2.0),
    noise_std: float = 0.0,
    seed: int = 42,
) -> QuadraticDataset:
    """输入采样配置，返回 y=x² 加可选噪声的数据集。"""
    if num_samples <= 0 or x_range[0] >= x_range[1] or noise_std < 0:
        raise ValueError("采样配置不合法")

    generator = torch.Generator().manual_seed(seed)
    # 在给定区间均匀采样 x。
    x = torch.empty(num_samples, 1).uniform_(*x_range, generator=generator)
    # 根据目标函数计算监督标签。
    y = x.square()
    # 噪声模拟观测误差。
    noise = torch.randn(y.shape, generator=generator, dtype=y.dtype)
    y = y + noise_std * noise
    return QuadraticDataset(x, y)


def make_loader(dataset: Dataset, batch_size: int = 32) -> DataLoader:
    """输入数据集和批大小，返回乱序训练加载器。"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    dataset = make_quadratic_data(num_samples=8, noise_std=0.1)
    x, y = dataset[0]
    print(f"单条样本: x={x.item():.3f}, y={y.item():.3f}")
