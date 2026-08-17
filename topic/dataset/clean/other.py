"""文本与随机性工具。"""

from __future__ import annotations

import random

import torch


def make_corpus(repeats: int = 80) -> str:
    """输入重复次数，返回用于演示的短字符语料。"""
    return "pytorch dataset makes batches " * repeats


def decode(token_ids: torch.Tensor, inverse_vocab: list[str]) -> str:
    """输入 token 张量和反向词表，返回解码文本。"""
    return "".join(inverse_vocab[index] for index in token_ids.tolist())


def set_seed(seed: int = 42) -> None:
    """输入随机种子，返回无；固定演示随机性。"""
    random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    print(make_corpus(1))
