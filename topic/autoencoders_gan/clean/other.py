"""自动编码器与 GAN 的通用工具。"""

from pathlib import Path
import random

import torch


def set_seed(seed: int) -> None:
    """输入：整数种子；返回：无。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """输入：无；返回：可用设备。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_tensor(value: object, path: str) -> Path:
    """输入：可序列化对象、路径；返回：保存路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, target)
    return target
