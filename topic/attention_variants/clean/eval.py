"""注意力变体的形状检查。"""

import torch

from data import build_loaders
from model import AttentionLM, build_model


@torch.no_grad()
def evaluate(model: AttentionLM, device: torch.device) -> dict[str, float]:
    """输入: model 与 device。
    返回: 输出检查指标。
    """
    _, loader = build_loaders()
    ids, targets = next(iter(loader))
    logits = model(ids.to(device))
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.to(device).flatten())
    return {"loss": loss.item(), "finite": float(torch.isfinite(logits).all())}


def main() -> None:
    """输入: 无。
    返回: 无。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for kind in ("full", "mla", "linear", "attnres"):
        print(kind, evaluate(build_model(kind).to(device).eval(), device))


if __name__ == "__main__":
    main()
