"""最小 DDP 训练入口。

单进程运行：python train.py
多进程运行：torchrun --standalone --nproc_per_node=2 train.py
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from data import make_classification_data, make_loader
from eval import evaluate
from model import Classifier
from other import cleanup_distributed, set_seed, setup_distributed, unwrap_model


def train(
    model: nn.Module,
    loader: DataLoader,
    sampler: DistributedSampler | None,
    device: torch.device,
    epochs: int = 12,
) -> float:
    """输入模型、数据、采样器和设备，返回最后一轮平均损失。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    final_loss = 0.0
    model.train()
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)  # 每轮改变各 rank 的数据顺序。
        total_loss = 0.0
        for x, y in loader:
            loss = torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        final_loss = total_loss / len(loader)
    return final_loss


def main() -> None:
    """输入为空，返回无；在单进程或 torchrun 下训练模型。"""
    rank, world_size, local_rank, device = setup_distributed()
    try:
        set_seed(42)
        loader, sampler = make_loader(make_classification_data(), batch_size=64, distributed=world_size > 1)
        base_model = Classifier().to(device)
        model: nn.Module = (
            DDP(base_model, device_ids=[local_rank]) if world_size > 1 and device.type == "cuda"
            else DDP(base_model) if world_size > 1
            else base_model
        )
        loss = train(model, loader, sampler, device)
        accuracy = evaluate(model, loader, device)
        if rank == 0:
            print(f"world_size={world_size}, loss={loss:.4f}, accuracy={accuracy:.3f}")
            print(f"base parameters={sum(p.numel() for p in unwrap_model(model).parameters())}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
