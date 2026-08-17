"""位置与时间编码评估。"""

import torch

from data import build_noise_loaders, build_token_loaders
from model import PositionLM, TimeDenoiser, build_models


@torch.no_grad()
def evaluate(lm: PositionLM, denoiser: TimeDenoiser, device: torch.device) -> dict[str, float]:
    """输入: lm、denoiser、device。
    返回: 语言与噪声损失。
    """
    _, token_loader = build_token_loaders()
    ids, targets = next(iter(token_loader))
    logits = lm(ids.to(device))
    language_loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.to(device).flatten())
    _, noise_loader = build_noise_loaders()
    noisy, times, noise = next(iter(noise_loader))
    time_loss = torch.nn.functional.mse_loss(denoiser(noisy.to(device), times.to(device)), noise.to(device))
    return {"language_loss": language_loss.item(), "time_loss": time_loss.item()}


def main() -> None:
    """输入: 无。
    返回: 无。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, rope, denoiser = (item.to(device) for item in build_models())
    print(evaluate(rope.eval(), denoiser.eval(), device))


if __name__ == "__main__":
    main()
