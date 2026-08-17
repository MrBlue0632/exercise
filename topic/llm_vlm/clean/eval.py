"""最小 VLM 的评估。"""

import torch

from data import build_loaders
from model import TinyVLM, build_model


@torch.no_grad()
def evaluate(model: TinyVLM, device: torch.device) -> dict[str, float]:
    """输入: model 与 device。
    返回: 分类准确率与长度。
    """
    _, loader = build_loaders()
    images, ids, targets = next(iter(loader))
    logits = model(images.to(device), ids.to(device))
    accuracy = (logits[:, -1].argmax(dim=-1).cpu() == targets).float().mean().item()
    return {"accuracy": accuracy, "sequence_length": float(logits.size(1))}


def main() -> None:
    """输入: 无。
    返回: 无。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(evaluate(build_model().to(device).eval(), device))


if __name__ == "__main__":
    main()
