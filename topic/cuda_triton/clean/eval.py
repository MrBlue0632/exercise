"""输入: 两种矩阵乘法函数。
返回: 正确性与耗时结果。"""

import time

import torch
from torch import Tensor


def benchmark(function: object, left: Tensor, right: Tensor, repeats: int = 5) -> float:
    """输入: 可调用算子和矩阵。
    返回: 平均执行毫秒数。"""
    start = time.perf_counter()
    for _ in range(repeats):
        function(left, right)
    # 使用短循环稳定计时
    return (time.perf_counter() - start) * 1000 / repeats


def check_operator(left: Tensor, right: Tensor) -> dict[str, float]:
    """输入: 两个矩阵乘法输入。
    返回: 误差与基线耗时。"""
    expected = left @ right
    actual = torch.matmul(left, right)
    error = (expected - actual).abs().max()
    return {"max_error": float(error), "torch_ms": benchmark(torch.matmul, left, right)}
