import torch

print("torch.tensor([1, 2, 3]):\n", torch.tensor([1, 2, 3]))     # 从数据创建
print("-" * 100)
print("torch.empty(2, 3):\n", torch.empty(2, 3))           # 空张量
print("-" * 100)
print("torch.zeros(2, 3):\n", torch.zeros(2, 3))           # 全 0
print("torch.ones(2, 3):\n", torch.ones(2, 3))            # 全 1
print("torch.full((2, 3), 5):", torch.full((2, 3), 5))       # 填充指定值
print("-" * 100)
print("torch.arange(0, 10, 2):\n", torch.arange(0, 10, 2))      # 等间隔整数
print("torch.linspace(0, 1, 5):\n", torch.linspace(0, 1, 5))     # 等间隔浮点数
print("torch.rand(2, 3):", torch.rand(2, 3))            # [0, 1) 均匀随机
print("-" * 100)
print("torch.randn(2, 3):\n", torch.randn(2, 3))           # 标准正态随机
print("torch.randint(0, 10, (2, 3)):\n", torch.randint(0, 10, (2, 3)))
print("torch.eye(3):\n", torch.eye(3))                # 单位矩阵
print("-" * 100)
print("\n"*3)   # 换行


x = torch.randn(2, 3, 4)
print("x.shape:", x.shape)          # 张量形状
print("x.ndim:", x.ndim)          # 张量维度
print("x.dtype:", x.dtype)          # 张量数据类型
print("x.size():", x.size())          # 张量大小
print("x.numel():", x.numel())          # 张量元素数量
print("x.mean().item():", x.item())          # 张量元素
print("x.tolist():", x.tolist())          # 张量列表
print("x.numpy():", x.numpy())          # 张量numpy数组
print("x.device:", x.device)          # 张量设备

print("\n"*3)   # 换行
print("x.requires_grad:", x.requires_grad)          # 张量是否需要梯度
print("x.grad:", x.grad)          # 张量梯度
print("x.grad_fn:", x.grad_fn)          # 张量梯度函数
print("x.is_leaf:", x.is_leaf)          # 张量是否是叶子节点
print("x.is_sparse:", x.is_sparse)          # 张量是否是稀疏张量
print("x.is_tensor:", x.is_tensor)          # 张量是否是张量
print("x.is_variable:", x.is_variable)          # 张量是否是变量
print("\n"*3)   # 换行
print("x.cuda():", x.cuda())          # 张量cuda
print("x.cpu():", x.cpu())          # 张量cpu
print("x.to(torch.float32):", x.to(torch.float32))          # 张量转换为float32
print("x.to(torch.float64):", x.to(torch.float64))          # 张量转换为float64
print("x.to(torch.int32):", x.to(torch.int32))          # 张量转换为int32
print("x.to(torch.int64):", x.to(torch.int64))          # 张量转换为int64