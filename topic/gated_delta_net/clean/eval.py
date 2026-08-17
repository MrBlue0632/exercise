"""Gated Delta 状态检查。"""

import torch

from data import build_loaders
from model import GatedDeltaLM, build_model


@torch.no_grad()
def evaluate(model: GatedDeltaLM, device: torch.device) -> dict[str, float]:
    """输入: model 与 device。
    返回: 状态检查指标。
    """
    _, loader = build_loaders()
    ids, targets = next(iter(loader))
    logits, state = model(ids.to(device))
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.to(device).flatten())
    return {"loss": loss.item(), "state_norm": state.norm().item(), "finite": float(torch.isfinite(state).all())}


def main() -> None:
    """输入: 无。
    返回: 无。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(evaluate(build_model().to(device).eval(), device))


if __name__ == "__main__":
    main()
