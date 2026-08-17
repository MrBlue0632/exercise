# CNN 卷积效果预览

这个小实验把两类内容分开显示：

1. **固定卷积核**：Gaussian、Sobel、Laplacian、Sharpen。
2. **真实 CNN 特征**：ImageNet 预训练 ResNet-18 的第一阶段
   `conv1 → batch norm → ReLU`。

输入采用之前约定的 `scikit-image` astronaut 图像，并统一缩放至
224×224。第一次运行会下载输入图片和约 45 MB 的 ResNet-18 预训练权重。

## 运行

```bash
cd mine/cnn_test
python cnn_preview.py
```

额外生成 32×32 / 16×16 RGB 压缩对比（普通缩小 vs. 缩小前增强高频）：

```bash
python resolution_preview.py
```

生成结果位于 `outputs/`：

- `cnn_preview.png`：所有主要效果的总览；
- `resnet18_feature_maps.png`：真实 CNN 第一阶段中方差最大的 16 个通道；
- `01_...png` 至 `08_...png`：每种效果的独立图片；
- `metadata.json`：卷积核和值、CNN 特征形状及所选通道。

## 如何阅读

- Gaussian 是低通滤波，会抑制细节和噪声。
- Sobel 和 Laplacian 权重和为 0，均匀区域响应弱，边缘响应强。
- 高频残差图以中灰色表示 0；比中灰更亮或更暗表示不同符号的快速变化。
- Unsharp mask 将高频残差加回原图，因此比单独显示高通结果更接近实际锐化。
- CNN 特征图不是普通图片；不同通道会分别响应颜色、方向、边缘和纹理。

## 环境

依赖 Python 3.10+、NumPy、Pillow、Matplotlib、PyTorch 和 TorchVision。
