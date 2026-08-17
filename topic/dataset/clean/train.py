"""使用自定义 Dataset 训练字符预测器。"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import TextWindowDataset, build_vocab, encode, make_loader
from eval import evaluate
from model import TokenPredictor
from other import make_corpus, set_seed


def train(model: nn.Module, loader: DataLoader, device: torch.device, epochs: int = 20) -> list[float]:
    """输入模型、数据、设备和轮数，返回每轮交叉熵。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-2)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        total_loss = 0.0
        for context, target in loader:
            logits = model(context.to(device))
            loss = torch.nn.functional.cross_entropy(logits, target.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / len(loader))
    return losses


def main() -> None:
    """输入为空，返回无；展示 Dataset 到训练循环的完整路径。"""
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, inverse_vocab = build_vocab(make_corpus())
    dataset = TextWindowDataset(encode(make_corpus(), vocab), context_size=4)
    loader = make_loader(dataset)
    model = TokenPredictor(len(vocab)).to(device)
    losses = train(model, loader, device)
    accuracy = evaluate(model, loader, device)
    print(f"vocab={len(inverse_vocab)}, loss={losses[-1]:.4f}, accuracy={accuracy:.3f}")


if __name__ == "__main__":
    main()
