# 模型代码专题检索结果

> 统计口径：日期为 GitHub 仓库创建日，Star 为 2026-08-09 快照。优先选择可读、现代、Star 1k+ 的实现；少数较早仓库因其教学价值而保留，并在说明中标出。

| 专题 | 优先参考仓库（创建日，Star） | 已放入 `reference` | 选择理由 |
|---|---|---|---|
| KV Cache | [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)（2023-07-23，★101,791）；[Mistral inference](https://github.com/mistralai/mistral-inference)（2023-09-27，★10,840）；[Attention Gym](https://github.com/meta-pytorch/attention-gym)（2024-07-31，★1,222） | 是 | 从基础缓存到 paged KV 均有短实现。 |
| Attention Mask | [Attention Gym](https://github.com/meta-pytorch/attention-gym)（2024-07-31，★1,222）；[build-nanogpt](https://github.com/karpathy/build-nanogpt)（2024-06-09，★5,413） | 是 | 覆盖因果、文档、共享前缀与 BlockMask。 |
| VAE / CVAE / VQ-VAE / GAN | [PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE)（2020-01-10，★7,664）；[vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)（2020-06-09，★3,996）；[PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)（2018-04-21，★17,455） | 是 | 虽较早，但模块边界清楚、可直接学习训练环。 |
| PyTorch 训练 | [build-nanogpt](https://github.com/karpathy/build-nanogpt)（2024-06-09，★5,413）；[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)（2023-07-23，★101,791）；[LitGPT](https://github.com/Lightning-AI/litgpt)（2023-05-04，★13,612） | 是 | 从单卡最小循环递进到现代训练工程。 |
| Dataset / DataLoader | [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)（2023-07-23，★101,791） | 是 | `Dataset`、滑窗 token 和 collate 写法易读。 |
| 共享 Attention | [Attention Gym](https://github.com/meta-pytorch/attention-gym)（2024-07-31，★1,222）；[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)（2023-07-23，★101,791） | 是 | 分清 GQA、跨层 KV、共享前缀三种含义。 |
| Wan 类视频生成 | [Wan2.1](https://github.com/Wan-Video/Wan2.1)（2025-02-25，★16,777）；[LTX-Video](https://github.com/Lightricks/LTX-Video)（2024-11-20，★10,825） | 是 | 官方视频扩散流水线，适合对照时空模块。 |
| Vision Encoder | [ViT-pytorch](https://github.com/lucidrains/vit-pytorch)（2020-10-03，★25,467）；[CLIP](https://github.com/openai/CLIP)（2020-12-16，★34,148）；[OpenCLIP](https://github.com/mlfoundations/open_clip)（2021-07-28，★14,051）；[DINOv2](https://github.com/facebookresearch/dinov2)（2023-03-29，★13,210）；[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)（2024-06-13，★8,616） | 是 | 从极简 ViT 到对比、蒸馏和深度编码器。 |
| LLM / VLM | [Mistral inference](https://github.com/mistralai/mistral-inference)（2023-09-27，★10,840）；[LitGPT](https://github.com/Lightning-AI/litgpt)（2023-05-04，★13,612）；[LLaVA](https://github.com/haotian-liu/LLaVA)（2023-04-17，★24,974）；[DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)（2024-12-26，★104,149） | 是 | Llama 系组件、VLM 拼接、MLA/MoE 都有对应源码。 |
| Diffusion / Stable Diffusion / DiT / Flow Matching | [pytorch-stable-diffusion](https://github.com/hkproj/pytorch-stable-diffusion)（2023-09-24，★1,075）；[DiT](https://github.com/facebookresearch/DiT)（2022-12-16，★8,691）；[flow_matching](https://github.com/facebookresearch/flow_matching)（2024-12-07，★4,671） | 是 | 覆盖 latent diffusion、DiT 与通用 flow-matching 目标。 |
| Attention 变体 / MLA / 线性 Attention | [Attention Gym](https://github.com/meta-pytorch/attention-gym)（2024-07-31，★1,222）；[Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)（2023-12-20，★5,528）；[DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)（2024-12-26，★104,149） | 是 | 有 MLA、GDN、KDA 的 eager 与高性能对照。 |
| 分布式 | [TorchTitan](https://github.com/pytorch/torchtitan)（2023-12-13，★5,604）；[build-nanogpt](https://github.com/karpathy/build-nanogpt)（2024-06-09，★5,413）；[GPU Mode lectures](https://github.com/gpu-mode/lectures)（2024-01-20，★6,411） | 是 | DDP 到 FSDP2、TP、PP、CP 的层级明确。 |
| LoRA / QLoRA | [QLoRA](https://github.com/artidoro/qlora)（2023-05-11，★10,986）；[TorchTune](https://github.com/meta-pytorch/torchtune)（2023-10-20，★5,794）；[LitGPT](https://github.com/Lightning-AI/litgpt)（2023-05-04，★13,612） | 是 | 包含 adapter 注入、量化微调和训练配方。 |
| Activation Checkpoint | [TorchTitan](https://github.com/pytorch/torchtitan)（2023-12-13，★5,604） | 是 | 官方重计算策略及与并行组合的范例。 |
| CUDA / Triton / Profiling | [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal)（2024-03-07，★1,174）；[GPU Mode lectures](https://github.com/gpu-mode/lectures)（2024-01-20，★6,411）；[llm.c](https://github.com/karpathy/llm.c)（2024-04-08，★30,761）；[KernelBench](https://github.com/ScalingIntelligence/KernelBench)（2024-10-25，★1,192）；[Triton](https://github.com/triton-lang/triton)（2014-08-30，★19,905） | 是 | 从百行 kernel 到 profiler、CUDA/NCCL 与 Triton 教程。 |
| 3D：VGGT / 3DGS / NeRF | [VGGT](https://github.com/facebookresearch/vggt)（2025-02-18，★14,208）；[gsplat](https://github.com/nerfstudio-project/gsplat)（2023-08-25，★5,507）；[gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)（2023-07-04，★22,909）；[nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)（2020-04-05，★6,049） | 是 | 官方/主流 3D 表示与可读 NeRF 基线并存。 |
| MoE | [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)（2023-07-23，★101,791）；[Mistral inference](https://github.com/mistralai/mistral-inference)（2023-09-27，★10,840）；[DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)（2024-12-26，★104,149） | 是 | 教学路由、官方推理 MoE 与 MLA 可交叉读。 |
| Gated Delta Net | [Attention Gym](https://github.com/meta-pytorch/attention-gym)（2024-07-31，★1,222）；[Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)（2023-12-20，★5,528） | 是 | 提供短 eager reference 与 Triton 优化实现。 |
| 位置 / 时间编码 | [Mistral inference](https://github.com/mistralai/mistral-inference)（2023-09-27，★10,840）；[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)（2023-07-23，★101,791）；[DiT](https://github.com/facebookresearch/DiT)（2022-12-16，★8,691）；[flow_matching](https://github.com/facebookresearch/flow_matching)（2024-12-07，★4,671） | 是 | RoPE/2D-RoPE、频率缩放、sinusoidal timestep、adaLN。 |

## Kimi 补充

| 项目 | 创建日 / Star | 结论 | 本地状态 |
|---|---:|---|---|
| [Kimi K3](https://github.com/MoonshotAI/Kimi-K3) | 2026-07-27 / ★8,273 | 报告提及 KDA、Attention Residuals、Gated MLA；官方仓库没有可运行模型源码。 | `reference/kimi_k3`，文档参考。 |
| [Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals) | 2026-03-15 / ★3,454 | 只有 README/PDF/assets；不应作为源码实现依据。 | `reference/attention_residuals`，文档参考。 |

Kimi 的实际可读代码参考已由 `attention_gym`（KDA/GDN/MLA）和 `flash_linear_attention`（GDN/KDA）补齐；`topic/attention_variants/clean` 中也保留了最小教学实现。

## 争议与边界

| 项目 | 争议点 | 当前处理 |
|---|---|---|
| “共享 attention” | 可能指 GQA、跨层 KV sharing 或共享前缀 mask，三者机制不同。 | 拆成三类分别引用。 |
| Kimi K3 / Attention Residuals | 论文/文档热门，但官方未公开可运行源码。 | 仅文档软链；以 Attention Gym/FLA 学习实现。 |
| DeepSeek-V3 | 官方 `inference/model.py` 很有价值，但属于推理模型文件，并非极简教程。 | 用作现代 MLA/MoE 对照，不作首份手写参考。 |
| Wan / VGGT / 3DGS | 官方工程规模大，依赖、CUDA 与数据管线较重。 | `clean` 只保留对应核心思路，不声称完整复刻。 |
| 较早项目 | VAE、GAN、CLIP、ViT、NeRF 仓库早于筛选时间范围。 | 因实现干净、教学价值高而保留，并标注年份。 |
| 分布式 / CUDA | 真实验证依赖多卡或 CUDA 环境。 | 当前最小代码可在 CPU 跑基础路径；性能路径需 GPU 验证。 |
