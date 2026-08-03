"""PyTorch Tensor attributes, dtype conversion, and device transfer example."""

import torch


def main():
    if not hasattr(torch, "tensor"):
        raise ImportError(
            "PyTorch is unavailable. This directory contains a local 'torch' folder "
            "that shadows the PyTorch package. Rename that folder or run the script "
            "from another directory after installing PyTorch."
        )

    # Default integer tensor on CPU
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])

    print("Original tensor:\n", x)
    print("shape:", x.shape)      # torch.Size([2, 3])
    print("dtype:", x.dtype)      # torch.int64
    print("device:", x.device)    # cpu

    # Convert data types (the original x is unchanged)
    x_float = x.float()
    x_long = x.long()
    print("\nAfter x.float():")
    print(x_float)
    print("dtype:", x_float.dtype)  # torch.float32
    print("\nAfter x.long():", x_long.dtype)  # torch.int64

    # Use GPU when it is available; otherwise keep the tensor on CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_on_device = x_float.to(device)
    print("\nCUDA available:", torch.cuda.is_available())
    print("x_float.to(device):", x_on_device.device)


if __name__ == "__main__":
    main()
