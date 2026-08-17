"""KV Cache 的一致性检查。"""

import torch
from torch import Tensor

from data import build_loaders
from model import TinyCachedLM, build_model
from other import greedy_decode


@torch.no_grad()
def evaluate(model: TinyCachedLM, device: torch.device) -> dict[str, float]:
    """输入: model 与 device。
    返回: 一致性指标。
    """
    _, loader = build_loaders()
    ids, _ = next(iter(loader))
    ids = ids.to(device)
    full_logits, _ = model(ids)

    # 逐词复用历史缓存
    cache = None
    chunks: list[Tensor] = []
    for step in range(ids.size(1)):
        logits, cache = model(ids[:, step : step + 1], cache)
        chunks.append(logits)
    cached_logits = torch.cat(chunks, dim=1)
    error = (full_logits - cached_logits).abs().max().item()
    generated = greedy_decode(model, ids[:1, :3], 3)
    return {"max_error": error, "generated_steps": float(generated.size(1))}


def main() -> None:
    """输入: 无。
    返回: 无。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device).eval()
    print(evaluate(model, device))


if __name__ == "__main__":
    main()
