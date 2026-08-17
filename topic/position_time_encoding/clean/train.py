"""位置和时间编码的训练示例。"""

import torch

from data import build_noise_loaders, build_token_loaders
from eval import evaluate
from model import PositionLM, TimeDenoiser, build_models


def train_language(model: PositionLM, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """输入: language model、loader、device。
    返回: 最后训练损失。
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    total = 0.0
    for _ in range(2):
        for ids, targets in loader:
            logits = model(ids.to(device))
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.to(device).flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total = loss.item()
    return total


def train_denoiser(model: TimeDenoiser, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    """输入: denoiser、loader、device。
    返回: 最后训练损失。
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    total = 0.0
    for _ in range(2):
        for noisy, times, noise in loader:
            prediction = model(noisy.to(device), times.to(device))
            loss = torch.nn.functional.mse_loss(prediction, noise.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total = loss.item()
    return total


def main() -> None:
    """输入: 无。
    返回: 无。
    """
    torch.manual_seed(15)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    absolute, rope, denoiser = (item.to(device) for item in build_models())
    token_loader, _ = build_token_loaders()
    noise_loader, _ = build_noise_loaders()
    print({"absolute_loss": train_language(absolute, token_loader, device)})
    print({"rope_loss": train_language(rope, token_loader, device)})
    print({"time_loss": train_denoiser(denoiser, noise_loader, device)})
    print(evaluate(rope.eval(), denoiser.eval(), device))


if __name__ == "__main__":
    main()
