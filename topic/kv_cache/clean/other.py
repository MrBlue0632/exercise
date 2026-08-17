"""KV Cache 辅助函数。"""

import torch
from torch import Tensor


def causal_mask(query_steps: int, key_steps: int, past: int, device: torch.device) -> Tensor:
    """输入: query、key 长度与 past。
    返回: allow[Q,K]。
    """
    query_pos = torch.arange(past, past + query_steps, device=device)[:, None]
    key_pos = torch.arange(key_steps, device=device)[None, :]
    return key_pos <= query_pos


@torch.no_grad()
def greedy_decode(model: torch.nn.Module, prompt: Tensor, new_tokens: int = 4) -> Tensor:
    """输入: model 与 prompt[B,T]。
    返回: generated[B,T+N]。
    """
    cache = None
    generated = prompt
    for _ in range(new_tokens):
        step = generated if cache is None else generated[:, -1:]
        logits, cache = model(step, cache)
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_id), dim=1)
    return generated
