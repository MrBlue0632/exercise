"""掩码可见性的检查。"""

import torch

from data import build_loaders
from model import MaskedLM, build_model
from other import causal_mask, combine_masks, padding_mask, prefix_lm_mask


@torch.no_grad()
def evaluate(model: MaskedLM, device: torch.device) -> dict[str, float]:
    """输入: model 与 device。
    返回: mask 检查指标。
    """
    _, loader = build_loaders()
    ids, _, lengths = next(iter(loader))
    ids, lengths = ids.to(device), lengths.to(device)
    valid = padding_mask(lengths, ids.size(1))
    allow = combine_masks(causal_mask(ids.size(1), device), valid)
    _, weight = model(ids, allow)

    # 验证禁止位置无权重
    blocked = weight.masked_select((~allow[:, None]).expand_as(weight))
    prefix = prefix_lm_mask(ids.size(1), 2, device)
    return {"blocked_max": blocked.abs().max().item(), "prefix_edges": float(prefix.sum())}


def main() -> None:
    """输入: 无。
    返回: 无。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(evaluate(build_model().to(device).eval(), device))


if __name__ == "__main__":
    main()
