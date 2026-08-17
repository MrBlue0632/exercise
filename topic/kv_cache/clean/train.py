"""KV Cache 的最小训练循环。"""

import torch
from torch import Tensor, nn

from data import build_loaders
from eval import evaluate
from model import TinyCachedLM, build_model


def train_epoch(
    model: TinyCachedLM, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, device: torch.device
) -> float:
    """输入: model、loader、optimizer。
    返回: 平均训练损失。
    """
    model.train()
    total = 0.0
    for ids, targets in loader:
        ids, targets = ids.to(device), targets.to(device)
        logits, _ = model(ids)
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def main() -> None:
    """输入: 无。
    返回: 无。
    """
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    train_loader, _ = build_loaders()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for epoch in range(4):
        print({"epoch": epoch, "loss": train_epoch(model, train_loader, optimizer, device)})
    print(evaluate(model.eval(), device))


if __name__ == "__main__":
    main()
