"""输入: 两个矩阵和可选 CUDA。
返回: Torch 或 Triton 矩阵乘法。"""

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    # 输入双数组与长度，输出写入结果
    @triton.jit
    def _add_kernel(left, right, output, size: tl.constexpr, block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < size
        values = tl.load(left + offsets, mask=mask) + tl.load(right + offsets, mask=mask)
        tl.store(output + offsets, values, mask=mask)


def vector_add(left: Tensor, right: Tensor) -> Tensor:
    """输入: 形状相同的一维张量。
    返回: 两者逐元素加法。"""
    if triton is None or not left.is_cuda:
        return left + right
    output = torch.empty_like(left)
    block = 256
    # 在 CUDA 上调度最小内核
    _add_kernel[(triton.cdiv(left.numel(), block),)](left, right, output, left.numel(), block=block)
    return output


def tiled_matmul(left: Tensor, right: Tensor, tile: int = 16) -> Tensor:
    """输入: 两个二维浮点矩阵。
    返回: 分块累加矩阵乘积。"""
    output = torch.zeros(left.size(0), right.size(1), device=left.device)
    for start in range(0, left.size(1), tile):
        # 显式展示分块累加
        output += left[:, start : start + tile] @ right[start : start + tile]
    return output
