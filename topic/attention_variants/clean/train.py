"""可切换注意力的训练循环。"""

import sys

import torch
from torch import nn

from data import build_loaders
from eval import evaluate
from model import AttentionLM, build_model


def train_epoch(
    model: AttentionLM, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, device: torch.device
) -> float:
    """输入: model、loader、optimizer。
    返回: 平均训练损失。
    """
    model.train()
    total = 0.0
    for ids, targets in loader:
        logits = model(ids.to(device))
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), targets.to(device).flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def main() -> None:
    """输入: 可选 attention kind。
    返回: 无。
    """
    torch.manual_seed(11)
    kind = sys.argv[1] if len(sys.argv) > 1 else "mla"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(kind).to(device)
    loader, _ = build_loaders()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for epoch in range(3):
        print({"kind": kind, "epoch": epoch, "loss": train_epoch(model, loader, optimizer, device)})
    print(evaluate(model.eval(), device))


if __name__ == "__main__":
    main()
