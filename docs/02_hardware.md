# 02 · 硬件架构：为什么同一模型在不同硬件上快慢差 40 倍

> **学习目标**：理解 CPU 和 GPU 分别怎么算神经网络、指令集是什么、为什么同一精度在不同硬件上速度差 40 倍，以及部署前如何检查目标硬件。
> **前置知识**：读完 [01 · 量化入门](./01_quantization_basics.md)（认识 fp32/int8/bf16 等精度格式）。
> **本篇不覆盖**：这些速度数字是怎么测出来的（[03 · 如何读懂性能数据](./03_reading_benchmarks.md)）。
>
> 核心结论一句话：**精度格式的实际速度取决于硬件是否为它提供原生执行单元，软件只是把硬件能力翻译出来。** 深水区内容集中在文末「**进一步学习**」一节，初学可跳过，不影响主线。

本篇的展开方式：先用两个最反直觉的实验现象提问（bf16 在 CPU 上慢 40 倍、在 GPU 上反而快一倍），再看 CPU/GPU 内部发生了什么来回答，最后把所有现象收进一张硬件-精度匹配矩阵。

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

## 2. CPU 上发生了什么

CPU 做神经网络计算，靠的是 **[SIMD](./00_glossary.md) 向量指令**：一条指令同时处理多个数据。

- `AVX2`：256 位宽向量寄存器 → 一条指令做 8 个 fp32 加/乘；
- `FMA`：乘加融合，一条指令完成 a×b+c；
- `AVX_VNNI`：int8 点积专用，一条指令完成 4 组 8-bit 乘加（`vpdpbusd`）——int8 乘加主体走 VNNI，[requantize](./00_glossary.md)/裁剪/搬移等辅助运算仍走 AVX2，两者配合构成完整 int8 推理路径；
- `AVX512-BF16` / `AMX`：bf16 向量运算 / 瓦片矩阵乘——**本机没有**。

**为什么 bf16 在 CPU 上慢 40 倍**（22.4 vs 0.573 ms）：

1. 没有 bf16 执行单元 → [oneDNN](./00_glossary.md) 检测后走**仿真路径**：bf16 → fp32 转换 → fp32 计算 → 转回 bf16。运算量一点没省，还多了两次转换；
2. **batch=1 时固定开销占满延迟**：每层计算只要几十微秒，但每次 forward 都要把几 MB 权重转一遍 dtype，这套开销不变——计算越少，转换占比越高；
3. fp32 路径有十几年优化（寄存器分块、缓存预取）+ 4.8 GHz 高频，本来就极快。

**为什么 int8 在 CPU 上反而最快**（0.242 ms）：

- 量化是**一次性完成、写死进模型**的，推理时直接 int8 运算，没有反复转换；
- AVX_VNNI 原生支持 int8 点积；
- 权重体积减半 → 内存带宽压力减半（CPU 小模型推理的瓶颈常在内存访问而非计算）。

## 3. GPU 上发生了什么

GPU 靠**海量并行**：3,584 个 [CUDA 核心](./00_glossary.md)同时算不同数据，矩阵乘还有专用的 **[Tensor Core](./00_glossary.md)**。

**为什么 bf16 在 GPU 上快一倍**（0.266 vs 0.564 ms）：RTX 3060 的第三代 Tensor Core 原生支持 BF16 矩阵乘，PyTorch 的 [`autocast(bf16)`](./00_glossary.md) 把 conv/[GEMM](./00_glossary.md) 分派给 Tensor Core，吞吐直接翻倍。batch=256 时计算量大，转换开销摊薄可忽略。

**为什么 TRT bf16 只有 TRT fp16 一半**（约 110k vs 320k 条/秒）——本机 GPU 上最有意思的结果：

![GPU 吞吐对比](../assets/speed_gpu_bar.png)

- **硬件层面的直接原因**：Tensor Core 是固定功能电路，吞吐由芯片设计决定。Ampere 白皮书对 GA10x（RTX 3060 的 GA106 属于此系）的配置是：每个 SM 每 clock 的 FP16 Tensor 吞吐 256 FMA，**BF16 Tensor 吞吐 128 FMA——物理上只有一半的乘法阵列**。TRT 实测 bf16 kernel 比 fp16 慢，反映的就是这个电路比值，不是软件没调优。
- **为什么厂商这么配：市场定位**。fp16 是消费卡的主场（游戏光影、DLSS、创作渲染都以 fp16 为主力），所以 GA10x 给 fp16 配满吞吐；bf16 存在的理由是深度学习训练（指数位与 fp32 相同不溢出），训练发生在数据中心卡上——A100 的 FP16 与 BF16 Tensor 吞吐完全相等（都是 312 [TFLOPS](./00_glossary.md)），消费卡则作为产品分层"能跑但减半"（A100 当时卖 1 万多美元，RTX 3060 卖 300 多）。
- **bf16 的速度谱系（取决于硬件是否把它当一等公民）**：

| 硬件 | bf16 vs fp16 | 原因 |
|---|---|---|
| 消费卡（本机 3060，GA10x） | **一半** | BF16 乘法阵列只有 FP16 的一半 |
| 数据中心卡（A100/H100） | 相等 | 为训练服务，吞吐对齐 |
| CPU（本机 i5，无 AVX512-BF16/AMX） | 两者都慢 40 倍 | 都无原生支持，走仿真 |

- 结论：**bf16 的价值在训练（不溢出）和数据中心卡（吞吐对齐）；消费级 GPU 推理选 fp16 永远不会吃亏**（要么更快要么持平）。

**为什么 TensorRT 比 PyTorch eager 快 30-100 倍**（1.8k → 10 万+ 条/秒）——差距不在精度在执行方式：

![eager vs TRT 时间线](../assets/precision_timeline.png)

| 提速来源 | eager 的问题 | TRT 的做法 | 收益 |
|---|---|---|---|
| 无 Python 分发 | 每个 [op（算子）](./00_glossary.md)走一次 Python→C++ 分发，小模型单层计算只要几微秒，分发占比 >80% | 引擎是编译好的 CUDA [kernel](./00_glossary.md) 序列，一次调用执行整图 | 消除调度开销 |
| 算子融合 | Conv/BN/ReLU 每层单独执行，中间结果反复读写显存 | Conv+BN+ReLU 融合成一个 kernel | 消除中间显存读写 |
| kernel 自动调优 | 通用 kernel，不针对具体 GPU 形状 | 构建期对本机 GPU 实测几十种实现选最快 | 针对 28 个 SM 精调 |

**注意**：PyTorch 的 `quantize_dynamic` 只有 CPU 实现（依赖 oneDNN/FBGEMM），int8 想在 GPU 上跑必须走 TensorRT（QDQ 显式量化）或 torchao。

## 4. 硬件-精度匹配矩阵（把前面的现象收进一张地图）

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
| tf32 | 19 bit（存储仍 32） | 1 + 8 + 10 | 与 fp32 相同 ~10^±38 | 接近 fp16 的尾数（10 位） | Ampere+ 上 fp32 矩阵乘的自动加速路径（需显式开启） |
| bf16 | 16 bit | 1 + 8 + 7 | **与 fp32 相同** ~10^±38 | 尾数只有 7 位，约 2~3 位十进制 | 大模型训练（不易溢出）、GPU 混合精度 |
| fp16 | 16 bit | 1 + 5 + 10 | ~±65504（小） | 尾数 10 位，约 3~4 位十进制 | GPU 推理（消费卡 Tensor Core 全速） |
| int8 | 8 bit | 纯整数（+scale 映射） | 由 scale 决定，256 级 | 量化误差 ~scale/2 | CPU 部署首选、TRT int8 |
| int4 | 4 bit | 纯整数（+scale/group） | 由 group scale 决定，16 级 | 量化误差最大，常配 per-group scale | 显存极致敏感的大模型权重（weight-only） |

**CPU 行**：fp32 是一等公民（0.59ms）；int8 最快（VNNI 原生，0.24ms）；fp16/bf16 **没有原生支持**，oneDNN 走仿真路径——慢 40 倍。

**GPU 行**：fp16 最快（Tensor Core 双倍吞吐）；bf16 快但只有 fp16 一半（消费卡差异，见第 3 节）；int8 需 TRT 才有 GPU kernel。

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

## 进一步学习：精度损失到底发生在哪一步

### 乘法与加法的精度行为差异

| 格式 | 乘法行为 | 加法行为 | 精度损耗发生在哪 |
|---|---|---|---|
| fp32 | 乘、加全程 fp32（乘积 48 位尾数里保留高位，加法逐位对齐） | 全程 fp32 | 无输入端截断；每步舍入到 24 位有效数字（误差 ~2^-24，基线） |
| tf32 | **输入先截断到 19 位**再进 Tensor Core 相乘 | 累加用 fp32 | 乘法输入端（每操作数丢 13 位尾数） |
| bf16 | 输入截断到 bf16（尾数 7 位）再乘 | 累加通常 fp32 | 乘法输入端（尾数只剩 7 位，误差 ~1%） |
| fp16 | 输入截断到 fp16（尾数 10 位）再乘 | 累加通常 fp32 | 乘法输入端（尾数 10 位，误差较小但可能溢出） |
| int8 | 乘法精确：int8×int8 的乘积完整落在 int32 里，**无任何舍入** | 累加用 int32，也是精确的 | **不在乘加**，而在量化映射本身（fp32→int8 的取整）和 requantize（int32→int8 再缩放） |
| int4 | 同 int8，乘法精确 | 累加 int32 精确 | 不在乘加，在量化映射（16 级取整，误差最大）+ per-group scale 恢复 |

### 为什么是"24 位 × 24 位 → 48 位"？fp32 不是 32 位吗

32 位是**存储信封**的大小（1 符号 + 8 指数 + 23 尾数）；乘法真正作用于**尾数的 24 位有效精度**（23 位 + 规格化隐含首位——为什么隐含首位存在，见 [01 · 第 1 节](./01_quantization_basics.md)）。指数只决定小数点位置——相乘时指数直接相加，不占尾数。十进制类比（假设尾数只存 3 位）：

```
a = 3.14 × 10^2，b = 2.72 × 10^1
精确乘积 = 8.5408 × 10^3      ← 3 位 × 3 位 = 最多 6 位（类比 24×24 → 48 位"草稿"）
存回 3 位尾数 → 8.54 × 10^3   ← 舍入一次写回（类比舍入回 24 位）
```

加法同理：指数差决定对齐要移多少位，移出的低位由 3 个保护位（guard/round/sticky）临时接住，保证"如同无限精度算完只舍入一次"。所以 fp32 并非无损，而是**每次运算只舍入一次到 24 位**（误差 ~2^-24 级累积）；对比 bf16：输入在进乘法器前就被砍到 8 位有效数字，误差大一个数量级以上。

**关键洞察**：浮点格式（tf32/bf16/fp16）的精度损失发生在**乘法输入端**——数据一进来就被截断尾数，然后硬件用这些低位数相乘；而 int8/int4 的乘加本身是**精确的整数运算**（int8 乘积完整保留在 int32），损失全部发生在进入整数域之前的**量化映射**那一步。例：`(-37) × 96 = -3552`，int8 乘积完整落在 int32 里，一个 bit 都不丢——int8 推理的全部误差都在量化映射那一步（0.37 → 47 的取整），之后的乘加是纯精确运算。

这解释了两个实验现象：① int8 量化后精度无损（乘加不引入新误差，只有映射误差且 MNIST 上可忽略）；② fp16 训练需要 GradScaler（累加虽是 fp32，但输入端截断让小梯度更易下溢）。

**记忆要点**：tf32 是"内部截断"格式——数据仍以 fp32 存放，Tensor Core 计算时只取高 19 位（1+8+10）参与矩阵乘，范围不变、精度介于 fp16 与 fp32 之间；Ampere 起 TRT 有 `TF32` 标志、PyTorch 有 `torch.backends.cuda.matmul.allow_tf32` 开关，本实验均未开启（fp32 走 CUDA cores）。bf16 和 fp16 同为 16 位，区别在指数/尾数的分配——bf16 拿尾数换指数（范围 = fp32，训练不溢出），fp16 拿指数换尾数（精度更高但会溢出）；int8/int4 不再是浮点，需要 scale 把浮点范围线性映射到整数级，int4 的 16 级分辨率必须靠 per-group scale（每 64~128 个权重一组）才能保住精度。

## 本篇小结

- CPU 靠 SIMD 向量指令（AVX2/FMA/VNNI）算神经网络；GPU 靠海量并行 + Tensor Core 专用矩阵乘单元。
- bf16 慢 40 倍/快一倍/只有 fp16 一半，三个现象同一个根源：**硬件有没有为这种格式造原生执行单元、造了多少**。
- TRT 快 30-100 倍的根源是执行方式（无 Python 分发 + 算子融合 + kernel 调优），不是精度。
- 下一篇解决实操问题：这些速度数字该怎么测才可信——[03 · 如何读懂性能数据](./03_reading_benchmarks.md)。

## 自测：你学会了吗

<details>
<summary>1. 同一个 bf16 模型，为什么在 CPU 上慢 40 倍、在消费级 GPU 上却比 fp32 快一倍？</summary>

同一根源：硬件有没有为 bf16 造原生执行单元。CPU（i5-13490F）没有 AVX512-BF16/AMX，oneDNN 走仿真路径（转 fp32 再算，还多两次转换）；RTX 3060 的 Tensor Core 原生支持 BF16 矩阵乘，吞吐翻倍。
</details>

<details>
<summary>2. 消费级 GPU 上，TRT bf16 的吞吐为什么只有 TRT fp16 的一半？</summary>

Tensor Core 是固定功能电路：Ampere GA10x 每 SM 每 clock 的 FP16 Tensor 吞吐 256 FMA、BF16 只有 128 FMA——物理上乘法阵列就少一半。原因是市场定位：fp16 是消费卡主场，bf16 服务数据中心训练卡（A100 上两者相等）。
</details>

<details>
<summary>3. TRT 比 PyTorch eager 快 30-100 倍，根源是什么？</summary>

执行方式而非精度：无 Python 分发（引擎一次调用执行整图）、算子融合（Conv+BN+ReLU 一个 kernel）、kernel 自动调优（构建期对本机 GPU 实测选最快实现）。
</details>

<details>
<summary>4. int8 的乘加本身有舍入误差吗？</summary>

没有。int8×int8 的乘积完整落在 int32 里，乘加是纯精确整数运算；int8 的全部误差发生在进入整数域之前的量化映射（round + clip）那一步。浮点格式（bf16/fp16）则相反——误差在乘法输入端截断尾数。
</details>

<details>
<summary>5. 部署前应该对目标硬件检查哪三件事？</summary>

① CPU 指令集（lscpu 查 AVX_VNNI/AVX512-BF16/AMX）；② GPU 架构（nvidia-smi 查 compute capability，8.6+ 才有 bf16/INT8 Tensor Core）；③ 推理引擎（PyTorch int8 只有 CPU 实现，GPU int8 走 TRT/torchao）。
</details>

---

上一篇：[01 · 量化入门](./01_quantization_basics.md) | 下一篇：[03 · 如何读懂性能数据](./03_reading_benchmarks.md)
