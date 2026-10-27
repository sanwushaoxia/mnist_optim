# 02 · 硬件架构：为什么同一模型在不同硬件上快慢差 40 倍

> 面向初学者。读完后你会理解：CPU 和 GPU 分别怎么算神经网络、指令集是什么、为什么 bf16 在 CPU 上慢 40 倍在 GPU 上反而快一倍。核心结论一句话：**精度格式的实际速度取决于硬件是否为它提供原生执行单元，软件只是把硬件能力翻译出来。**

## 1. 本机硬件规格（实测）

### CPU：Intel Core i5-13490F（Raptor Lake 消费级，2023）

| 项目 | 规格 |
|---|---|
| 架构 | x86-64，P 核 + E 核混合（6P ×2 超线程 + 4E = 16 线程） |
| 频率 | 最高 4.8 GHz |
| 缓存 | L1d 416 KiB / L2 9.5 MiB / L3 24 MiB |
| 向量指令集 | AVX2、FMA、**AVX_VNNI** |
| **关键缺失** | **无 AVX-512（因此无 AVX512-BF16）、无 AMX**（AMX 只在服务器 Xeon 上） |

### GPU：NVIDIA GeForce RTX 3060 12GB（Ampere，GA106）

| 项目 | 规格 |
|---|---|
| 计算能力 | 8.6（Ampere 架构） |
| 流式多处理器 | 28 个 SM |
| CUDA 核心 | 3,584 个（fp32/fp16 通用单元） |
| Tensor Core | 每 SM 4 个（第三代：FP16/BF16/INT8/INT4 矩阵乘专用单元） |
| 显存 | 12 GB GDDR6，192-bit，约 360 GB/s |
| L2 缓存 | 2.25 MB（实测） |

## 2. 硬件-精度匹配矩阵（本项目全部结论浓缩）

![硬件-精度匹配矩阵](../assets/hw_precision_matrix.png)

**图中术语说明**（矩阵每格三行：计算单元/指令集 → 测试框架 → 耗时）：

| 术语 | 说明 |
|---|---|
| AVX2 / FMA | x86 向量指令（256 位宽）：`vaddps`/`vmulps` 一条指令 8 个 fp32 加/乘，FMA（`vfmadd231ps`）一条指令 8 组乘加；也支持 int8/int32 的加减与按位运算。fp32 推理的乘加主体 |
| AVX_VNNI | int8 点积专用指令（`vpdpbusd`），一条指令 4 组 × 8 对 int8 乘加进 int32 累加器；不支持 fp32 运算。与 AVX2 配合（乘加走 VNNI，requantize/搬移/裁剪走 AVX2）构成完整 int8 推理路径 |
| AMX | Intel 服务器 CPU 矩阵引擎（Sapphire Rapids+），消费级桌面 CPU 没有。**AMX vs AVX512**：AVX512 是更宽的向量指令集（512 位，一条指令 16 个 fp32/32 个 int8，仍需自己组织循环）；AMX 是瓦片矩阵加速器（一条 `TDPBSSD` 指令完成整块矩阵乘，思路接近 Tensor Core）。两者消费级 Raptor Lake 都已砍掉 |
| oneDNN | Intel CPU 深度学习算子库（PyTorch CPU 后端）；无硬件支持时用 fp32 模拟 fp16/bf16 |
| CUDA cores | NVIDIA 通用 FP32 单元；fp32 在这里跑（Tensor Core 不支持 fp32 矩阵乘） |
| Tensor Core (TC) | NVIDIA 矩阵乘专用单元（Ampere 三代：fp16/bf16/int8/int4），无 fp32 模式 |
| TRT / int8 QDQ | TensorRT 推理引擎；QDQ = 通过 Quantize/Dequantize 节点的显式量化 |
| tinygemm | CUDA int8/int4 weight-only GEMM kernel（torchao），按组反量化权重 |
| PyTorch eager | PyTorch 默认逐算子执行方式（无图编译/融合） |
| not tested | int4 在 CPU 上无算子（PyTorch/torchao 均未实现），本项目未测耗时 |

### 精度格式对照表

| 格式 | 位宽 | 结构（符号/指数/尾数） | 数值范围 | 相对精度 | 典型用途 |
|---|---|---|---|---|---|
| fp32 | 32 bit | 1 + 8 + 23 | ~10^±38（基线） | 最高（基线） | 训练金标准、精度兜底 |
| bf16 | 16 bit | 1 + 8 + 7 | **与 fp32 相同** ~10^±38 | 尾数只有 7 位，约 2~3 位十进制 | 大模型训练（不易溢出）、GPU 混合精度 |
| fp16 | 16 bit | 1 + 5 + 10 | ~±65504（小） | 尾数 10 位，约 3~4 位十进制 | GPU 推理（消费卡 Tensor Core 全速） |
| int8 | 8 bit | 纯整数（+scale 映射） | 由 scale 决定，256 级 | 量化误差 ~scale/2 | CPU 部署首选、TRT int8 |
| int4 | 4 bit | 纯整数（+scale/group） | 由 group scale 决定，16 级 | 量化误差最大，常配 per-group scale | 显存极致敏感的大模型权重（weight-only） |

**记忆要点**：bf16 和 fp16 同为 16 位，区别在指数/尾数的分配——bf16 拿尾数换指数（范围 = fp32，训练不溢出），fp16 拿指数换尾数（精度更高但会溢出）；int8/int4 不再是浮点，需要 scale 把浮点范围线性映射到整数级，int4 的 16 级分辨率必须靠 per-group scale（每 64~128 个权重一组）才能保住精度。

**CPU 行**：fp32 是一等公民（0.59ms）；int8 最快（VNNI 原生，0.24ms）；fp16/bf16 **没有原生支持**，oneDNN 走仿真路径——把 bf16 转 fp32 再算，慢 40 倍。

**GPU 行**：fp16 最快（Tensor Core 双倍吞吐）；bf16 快但只有 fp16 一半（消费卡差异，见第 4 节）；int8 需 TRT 才有 GPU kernel。

## 3. CPU 上发生了什么

CPU 做神经网络计算，靠的是 **SIMD 向量指令**：一条指令同时处理多个数据。

- `AVX2`：256 位宽向量寄存器 → 一条指令做 8 个 fp32 加/乘；
- `FMA`：乘加融合，一条指令完成 a×b+c；
- `AVX_VNNI`：int8 点积专用，一条指令完成 4 组 8-bit 乘加（`vpdpbusd`）——int8 乘加主体走 VNNI，requantize/裁剪/搬移等辅助运算仍走 AVX2，两者配合构成完整 int8 推理路径；
- `AVX512-BF16` / `AMX`：bf16 向量运算 / 瓦片矩阵乘——**本机没有**。

**为什么 bf16 在 CPU 上慢 40 倍**（22.4 vs 0.573 ms）：

1. 没有 bf16 执行单元 → oneDNN 检测后走**仿真路径**：bf16 → fp32 转换 → fp32 计算 → 转回 bf16。运算量一点没省，还多了两次转换；
2. **batch=1 时固定开销占满延迟**：每层计算只要几十微秒，但每次 forward 都要把几 MB 权重转一遍 dtype，这套开销不变——计算越少，转换占比越高；
3. fp32 路径有十几年优化（寄存器分块、缓存预取）+ 4.8 GHz 高频，本来就极快。

**为什么 int8 在 CPU 上反而最快**（0.242 ms）：

- 量化是**一次性完成、写死进模型**的，推理时直接 int8 运算，没有反复转换；
- AVX_VNNI 原生支持 int8 点积；
- 权重体积减半 → 内存带宽压力减半（CPU 小模型推理的瓶颈常在内存访问而非计算）。

## 4. GPU 上发生了什么

GPU 靠**海量并行**：3,584 个 CUDA 核心同时算不同数据，矩阵乘还有专用的 **Tensor Core**（矩阵乘加速单元）。

**为什么 bf16 在 GPU 上快一倍**（0.266 vs 0.564 ms）：RTX 3060 的第三代 Tensor Core 原生支持 BF16 矩阵乘，PyTorch 的 `autocast(bf16)` 把 conv/GEMM 分派给 Tensor Core，吞吐直接翻倍。batch=256 时计算量大，转换开销摊薄可忽略。

**为什么 TRT bf16 只有 TRT fp16 一半**（约 110k vs 320k 条/秒）——本机 GPU 上最有意思的结果：

![GPU 吞吐对比](../assets/speed_gpu_bar.png)

- GA106（消费卡）的 Tensor Core 主打 **FP16 双倍吞吐**（游戏光影需要）；
- **BF16 加速的重点在数据中心卡**（A100 那类，训练大模型的地方），消费卡上是"顺带支持"，矩阵乘吞吐为 fp16 的一半；
- 结论：**bf16 的价值在训练和数据中心卡；消费级 GPU 推理选 fp16**。

**为什么 TensorRT 比 PyTorch eager 快 30-100 倍**（1.8k → 10 万+ 条/秒）——差距不在精度在执行方式：

![eager vs TRT 时间线](../assets/precision_timeline.png)

1. **无 Python 分发**：eager 每个 op 走 Python→C++ 分发，小模型上分发占比 >80%；
2. **算子融合**：Conv+BN+ReLU → 一个 kernel，消除中间显存读写；
3. **kernel 自动调优**：构建期对本机 28 个 SM 实测选最快实现。

## 5. 一张表总结

| 精度 | CPU（AVX2+VNNI，无 AMX） | GPU（Ampere 8.6 消费卡） |
|---|---|---|
| fp32 | 快（一等公民 + 高频）0.59ms | 一般（无 Tensor Core） |
| fp16 | 无原生支持（仿真）22.1ms | **最快**（TC 双倍吞吐）0.12ms |
| bf16 | 无原生支持（**最慢**）22.4ms | 较快但只有 fp16 一半 0.16ms |
| int8 | **最快**（VNNI + 带宽减半）0.24ms | 快（需 TRT；TC 支持） |

规律：**同一精度格式在不同硬件上速度可以差 40 倍，决定因素是硬件有没有为它造专用执行单元。** 选精度之前，先看目标硬件的指令集/架构支持。

## 6. 给初学者的检查清单

部署前问三个问题：

1. **目标 CPU 有什么指令集？** `lscpu | grep -o 'avx[^ ]*\|amx[^ ]*'`——有 AVX_VNNI 选 int8，有 AVX512-BF16/AMX 才考虑 bf16；
2. **目标 GPU 什么架构？** `nvidia-smi --query-gpu=compute_cap --format=csv`——8.6+ 才有 bf16/INT8 Tensor Core；FP8 要 8.9+（Ada/40 系）；
3. **用什么推理引擎？** PyTorch 的 int8 量化只有 CPU 实现；GPU int8 走 TensorRT（QDQ 显式量化）或 torchao；追求极致吞吐一律 TensorRT。

---

上一篇：[01 · 量化入门](./01_quantization_basics.md) | 下一篇：[03 · 如何读懂性能数据](./03_reading_benchmarks.md)
