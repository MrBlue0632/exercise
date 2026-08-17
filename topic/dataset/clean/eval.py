"""分类评估。"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """输入模型、加载器和设备，返回下一个 token 准确率。"""
    model.eval()
    correct = 0
    total = 0
    for context, target in loader:
        logits = model(context.to(device))
        correct += (logits.argmax(dim=-1) == target.to(device)).sum().item()
        total += target.numel()
    return correct / total


if __name__ == "__main__":
    from data import TextWindowDataset, build_vocab, encode, make_loader
    from model import TokenPredictor

    text = "hello pytorch"
    vocab, _ = build_vocab(text)
    loader = make_loader(TextWindowDataset(encode(text, vocab)))
    print(f"accuracy={evaluate(TokenPredictor(len(vocab)), loader, torch.device('cpu')):.2f}")
