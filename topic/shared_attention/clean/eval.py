"""GQA 与共享前缀检查。"""

import torch

from data import build_loaders
from model import SharedAttentionLM, build_model
from other import cache_nbytes, expand_prefix_cache


@torch.no_grad()
def evaluate(model: SharedAttentionLM, device: torch.device) -> dict[str, float]:
    """输入: model 与 device。
    返回: 共享缓存指标。
    """
    _, loader = build_loaders()
    ids, _ = next(iter(loader))
    ids = ids.to(device)
    prefix, suffix = ids[:1, :4], ids[:2, 4:8]

    # 两个请求复用同一前缀
    _, prefix_cache = model(prefix)
    shared_logits, _ = model(suffix, expand_prefix_cache(prefix_cache, suffix.size(0)))
    full_ids = torch.cat((prefix.expand(suffix.size(0), -1), suffix), dim=1)
    full_logits, _ = model(full_ids)
    error = (shared_logits - full_logits[:, -suffix.size(1) :]).abs().max().item()
    shared = cache_nbytes(prefix_cache)
    return {"prefix_error": error, "saved_bytes": float(shared * (suffix.size(0) - 1))}


def main() -> None:
    """输入: 无。
    返回: 无。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(evaluate(build_model().to(device).eval(), device))


if __name__ == "__main__":
    main()
