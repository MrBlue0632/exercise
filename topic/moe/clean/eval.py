"""MoE 的专家路由评估。"""

import torch

from data import build_loaders
from model import MoELM, build_model
from other import expert_histogram


@torch.no_grad()
def evaluate(model: MoELM, device: torch.device) -> dict[str, float]:
    """输入: model 与 device。
    返回: 路由均衡指标。
    """
    _, loader = build_loaders()
    ids, targets = next(iter(loader))
    logits, _, indices = model(ids.to(device))
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.to(device).flatten())
    counts = expert_histogram(indices.cpu(), model.moe.expert_count).float()
    return {"loss": loss.item(), "min_routes": counts.min().item(), "max_routes": counts.max().item()}


def main() -> None:
    """输入: 无。
    返回: 无。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(evaluate(build_model().to(device).eval(), device))


if __name__ == "__main__":
    main()
