"""图像前缀语言建模训练。"""

import torch
from torch import nn

from data import build_loaders
from eval import evaluate
from model import TinyVLM, build_model
from other import make_vlm_labels


def train_epoch(
    model: TinyVLM, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, device: torch.device
) -> float:
    """输入: model、loader、optimizer。
    返回: 平均训练损失。
    """
    model.train()
    total = 0.0
    for images, ids, targets in loader:
        images, ids, targets = images.to(device), ids.to(device), targets.to(device)
        logits = model(images, ids)
        labels = make_vlm_labels(ids, targets, model.vision_tokens)
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def main() -> None:
    """输入: 无。
    返回: 无。
    """
    torch.manual_seed(10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    loader, _ = build_loaders()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for epoch in range(5):
        print({"epoch": epoch, "loss": train_epoch(model, loader, optimizer, device)})
    print(evaluate(model.eval(), device))


if __name__ == "__main__":
    main()
