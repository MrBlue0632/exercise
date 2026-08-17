"""先预训练，再仅训练 LoRA。"""

import torch
from torch import nn

from data import build_loaders
from eval import evaluate
from model import BaseClassifier, LoRAClassifier, build_model
from other import freeze_for_lora, move_batch, trainable_parameters


def fit(
    model: nn.Module, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, epochs: int
) -> float:
    """输入: model、loader、optimizer、epochs。
    返回: 最后平均损失。
    """
    for _ in range(epochs):
        model.train()
        total = 0.0
        for batch in loader:
            values, labels = move_batch(batch, device)
            loss = nn.functional.cross_entropy(model(values), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
    return total / len(loader)


def main() -> None:
    """输入: 无。
    返回: 无。
    """
    torch.manual_seed(12)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_train, _ = build_loaders(target_task=False)
    target_train, target_valid = build_loaders(target_task=True)

    # 先得到冻结底座
    base: BaseClassifier = build_model().to(device)
    fit(base, base_train, torch.optim.AdamW(base.parameters(), lr=1e-2), device, 5)
    adapter = LoRAClassifier(base).to(device)
    freeze_for_lora(adapter)
    loss = fit(adapter, target_train, torch.optim.AdamW(filter(lambda p: p.requires_grad, adapter.parameters()), lr=3e-2), device, 8)
    print({"loss": loss, "trainable": trainable_parameters(adapter), "target_accuracy": evaluate(adapter, target_valid, device)})


if __name__ == "__main__":
    main()
