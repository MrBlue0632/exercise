"""LoRA 分类准确率。"""

import torch
from torch import nn

from data import build_loaders
from model import BaseClassifier, build_model


@torch.no_grad()
def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """输入: model、loader、device。
    返回: 分类准确率。
    """
    model.eval()
    correct, total = 0, 0
    for values, labels in loader:
        prediction = model(values.to(device)).argmax(dim=-1).cpu()
        correct += int((prediction == labels).sum())
        total += labels.numel()
    return correct / total


def main() -> None:
    """输入: 无。
    返回: 无。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, loader = build_loaders(target_task=True)
    print({"target_accuracy": evaluate(build_model().to(device), loader, device)})


if __name__ == "__main__":
    main()
