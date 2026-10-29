# 04 · 完整实验结果与数据手册

> **定位**：本篇是**速查手册**——按主题收拢全部实验数据、结论和踩坑记录，随时跳查。原理和教学方法见 [01 · 量化入门](./01_quantization_basics.md)，硬件原因见 [02 · 硬件架构](./02_hardware.md)，测量方法见 [03 · 如何读懂性能数据](./03_reading_benchmarks.md)。
> **建议用法**：先看 §1 核心结论速查；需要某个实验的完整数据再到 §2 找对应小节。

## 实验环境

- 环境：conda `py312`（Python 3.12.12），PyTorch 2.6.0+cu124，RTX 3060 GPU，i5-13490F 16 核 CPU
- 模型：自定义 CNN（Conv-BN-ReLU ×3 + FC ×2，约 200 万参数）
- 训练：fp32、5 epoch、Adam lr=1e-3、batch 256，GPU 训练

## 1. 核心结论速查

### 体积与精度总表（全部模型形态）

| 模型形态 | 大小 | 准确率（MNIST） | 备注 |
|---|---|---|---|
| fp32 基线 | 12.64 MB | 99.04% | 基线 |
| fp32 + compile | 12.64 MB | 99.04% | CPU 提速 15-40% 的中间选项 |
| bf16 混合精度 | 6.47 MB | 99.04% | 敏感层保留 fp32 |
| int8 动态 | 3.44 MB | 99.05% | 仅 Linear，无需校准 |
| **int8 静态** | **3.20 MB** | 98.98% | CPU 部署最优 |
| TRT fp32 / fp16 / bf16 | 12.74 / 6.42 / 6.47 MB | 99.04% / 99.04% / 99.05% | GPU 引擎 |
| TRT int8 | 3.30 MB | 99.79%* | 体积最小，需校准 |
| torchao int8wo / int4wo（GPU） | 3.45 / 2.24 MB | 99.04% | 权重 int8/int4、激活 bf16 |
| 剪枝 fp32（无微调） | 12.76 MB | 98.26% | 2:4 稀疏，配合微调可恢复 |

\* int8 引擎准确率略高于 fp32，是 max-abs 校准的正则化效应 + 测试波动（0.2% 以内差异无统计意义）。

### 速度速查

| 维度 | 最优 | 数字 | 结论一句话 |
|---|---|---|---|
| CPU 延迟（batch=1 P50） | **int8 静态** | 0.242 ms（fp32 的 2.4×，动态的 2.3×） | Conv 也量化 + VNNI 原生 + 带宽减半 |
| CPU 吞吐（batch=256） | **int8 静态** | 6,402 条/秒（fp32 的 2×） | 同上 |
| GPU 延迟（batch=1） | **TRT fp16** | ~0.12 ms | Tensor Core 双倍吞吐 |
| GPU 吞吐（batch=256） | **TRT fp16** | ~227k-356k 条/秒（eager 的 90×） | 引擎优化，不是精度功劳 |
| GPU 吞吐（不换引擎） | bf16 混合 | 3,759 条/秒（eager fp32 的 2.1×） | Tensor Core 原生支持 |
| 训练加速 | fp16+GradScaler | 1.7×（bf16 为 1.4×） | 均无损精度 |

### 场景推荐

| 场景 | 最优选择 | 实测 |
|---|---|---|
| **CPU 部署** | int8 静态量化 | 0.242 ms（fp32 的 2.4 倍速），3.2 MB |
| **GPU 部署** | TensorRT fp16 | 0.12 ms / 32 万条/秒（eager 的 90 倍），精度无损 |
| **CPU 快速集成** | int8 动态量化 | 无需校准数据，精度最稳 |
| **大模型训练** | bf16 混合精度 | 指数位与 fp32 相同，不易溢出（CPU 上无硬件支持，勿用） |
| **静态量化掉精度** | QAT | 本实验恢复 6.5% 且反超基线，成本极低 |
| **大模型显存敏感** | torchao weight-only | ResNet-18 上 GPU 反超 4.27× |

> 以上每条结论的「为什么」：精度格式与硬件的匹配见 [02 · 硬件-精度匹配矩阵](./02_hardware.md)；执行方式与引擎差距见 [02 · GPU 上发生了什么](./02_hardware.md)；校准与 QAT 见 [01 · 第 6/7 节](./01_quantization_basics.md)。

## 2. 分实验数据

### 2.1 推理侧 CPU 基准（PyTorch / oneDNN）

两种量化方式的原理（动态=激活现场量化、静态=校准后写死 scale）见 [01 · 动态 vs 静态](./01_quantization_basics.md)；bf16 混合精度的构成见 [01 · 混合精度](./01_quantization_basics.md)。

第一批实验原始汇总（bf16 混合精度 + int8 动态量化）：

| 指标 | fp32 基线 | bf16 混合精度 | int8 动态 |
|---|---|---|---|
| 测试集准确率 | 99.04% | 99.04% | **99.05%** |
| 模型文件大小 | 12.64 MB | 6.47 MB（-48.8%） | **3.44 MB（-72.8%）** |
| CPU 延迟（batch=1） | 0.573 ms/条 | 22.433 ms/条（⚠️ autocast 开销） | **0.553 ms/条** |
| CPU 吞吐（batch=256） | 1,746 条/秒 | 45 条/秒（⚠️） | **1,808 条/秒** |
| GPU 延迟（batch=1） | 0.564 ms/条 | **0.266 ms/条** | 不支持* |
| GPU 吞吐（batch=256） | 1,773 条/秒 | **3,759 条/秒** | 不支持* |

\* PyTorch 的动态 int8 量化算子（`quantize_dynamic`）仅支持 CPU 推理，GPU 上无对应算子。

int8 想上 GPU 的三条路径：① torchao weight-only（PyTorch 生态内，见 2.3）；② 导出 ONNX 后用 TensorRT int8（见 2.2）；③ 小模型 GPU 上不量化，bf16 已覆盖提速需求。

CPU 综合基准（延迟 200 次取分位数，i5-13490F）：

| 模型 | 大小 MB | P50 ms | P95 ms | P99 ms | 准确率 | 吞吐 b=64 | b=256 | b=1024 |
|---|---|---|---|---|---|---|---|---|
| fp32 | 12.64 | 0.592 | 0.640 | 0.708 | 99.04% | 3,909 | 3,214 | 2,756 |
| fp32+compile | 12.64 | 0.510 | 0.583 | 0.799 | 99.04% | 5,524 | 5,386 | 4,517 |
| bf16 混合 | 6.47 | 20.9 | 67.1 | 85.0 | 99.04% | 281 | 314 | 326 |
| int8 动态 | 3.44 | 0.563 | 0.578 | 0.600 | 99.05% | 4,153 | 3,278 | 2,711 |
| **int8 静态** | **3.20** | **0.242** | **0.254** | 0.418 | 99.90%→98.98% | **6,671** | **6,402** | **7,322** |

**结论**：静态量化是 CPU 部署的全面最优（P50 快 2.4×、吞吐约 2×、体积最小、精度几乎无损）；torch.compile 可作不离开 PyTorch 生态的中间选项；bf16 的 CPU autocast 路径确认不适合生产（P99 达 85 ms）——**bf16 在 CPU 上慢 40 倍的硬件原因见 [02 · CPU 上发生了什么](./02_hardware.md)**。

### 2.2 TensorRT 部署对比（fp32 / fp16 / bf16 / int8 引擎）

通过 ONNX 导出 + TensorRT 11 构建引擎。环境：TensorRT 11.4、RTX 3060（12GB，驱动 580.178）。

**构建方式（适配 TRT 11）**：TRT 11 移除了弱类型 API（`BuilderFlag.INT8`、`IInt8Calibrator` 校准器），int8 改用官方推荐的**显式量化路径**——在 ONNX 中插入 QDQ（QuantizeLinear/DequantizeLinear）节点 + `STRONGLY_TYPED` 强类型网络构建。激活 scale 用训练集 8192 张图做 max-abs 校准（per-tensor），权重 per-channel。fp16/fp32 为导出时 cast 权重 dtype。

**结果（多轮测量，取代表值）**：

| 引擎 | 大小 | 延迟 batch=1 | 吞吐 batch=256 | 准确率 |
|---|---|---|---|---|
| TRT fp32 | 12.74 MB | ~0.18-0.36 ms | ~87k-114k 条/秒 | 99.04% |
| TRT fp16 | 6.42 MB | ~0.12 ms | **~227k-356k 条/秒** | 99.04% |
| TRT bf16 | 6.47 MB | ~0.16 ms | ~110k 条/秒 | 99.05% |
| TRT int8 | **3.30 MB** | ~0.09-0.21 ms | ~237k-356k 条/秒 | 99.79%* |

\* 见 §1 总表脚注。

**TRT 与 PyTorch eager 的吞吐对比（GPU, batch=256）**：

| 实现 | fp32 | fp16/bf16 | int8 |
|---|---|---|---|
| PyTorch eager | 1.8k 条/秒 | 3.8k 条/秒 | 不支持 |
| TensorRT | 104k+ 条/秒 | 292k+ 条/秒 | 237k+ 条/秒 |

**结论**：TRT 相对 eager 快 **30-100 倍**——根源是执行方式（无 Python 分发 + 算子融合 + kernel 自动调优），不是精度；原理展开见 [02 · GPU 上发生了什么](./02_hardware.md)。fp16 吞吐约为 bf16 的 2 倍，是消费卡 Tensor Core 的电路配比差异（见 [02 · 同上节](./02_hardware.md)）。部署推荐次序：TRT fp16（无脑选）→ TRT int8（体积敏感时）；bf16 保留给大模型训练。bf16 引擎需 opset 20 导出 ONNX（TRT 强类型网络可保留 BF16 dtype）。

### 2.3 torchao weight-only 量化（GPU）

torchao 0.13 的 weight-only 混合精度（权重 int8/int4、激活 bf16，tinygemm kernel 原生 CUDA），GPU 实测（RTX 3060）：

| 模型 | 大小 MB | 延迟 ms | 吞吐/秒 | 准确率 |
|---|---|---|---|---|
| fp32 | 12.64 | 0.17-0.22 | 35-52k | 99.04% |
| int8 weight-only | 3.45 | 0.25-0.48 | 31-49k | 99.04% |
| int4 weight-only | 2.24 | 11-20 ⚠️ | 8.8-16k | 99.04% |

**结论**：int8 weight-only 精度无损、体积减半，但本场景（小模型）速度与 fp32 持平——per-group 反量化开销占比过大；int4 体积 2.24 MB 但延迟反而最高。**大模型上的表现见 2.10（ResNet-18 反超 4.27×），证实 weight-only 适合大模型/显存敏感场景**。

### 2.4 ONNX Runtime 对比（fp32 ONNX 模型）

| Provider | 延迟 P50 | 吞吐/秒 | 准确率 |
|---|---|---|---|
| CPU | 0.82-0.88 ms | 4.4-4.5k | 99.04%* |
| CUDA | 0.229 ms | 29-52k | 99.04%* |
| TensorRT-EP | 不支持（回退 CPU） | - | - |

\* 此表准确率列显示 0.9792 是 QDQ 版 ONNX 的数值（build_trt.py 最后一次导出覆盖了 model.onnx），fp32 原版为 99.04%。

**结论**：ORT CUDA EP 吞吐约 3-5 万条/秒——比 PyTorch eager 快约 10 倍，但只有原生 TRT 的 1/4-1/8（TRT 的 kernel 调优和融合更深）。

### 2.5 2:4 结构化稀疏（TRT SPARSE_WEIGHTS）

对 Conv/Linear 权重沿输入通道方向做 2:4 剪枝（每 4 个保留 abs 最大的 2 个，实际稀疏度约 50%；首层 conv 输入维 9 非 4 倍数跳过），导出 ONNX 后用 `SPARSE_WEIGHTS` 标志构建引擎，与同权重非稀疏引擎对照：

| 引擎 | 大小 | 延迟 batch=1 | 吞吐 batch=256 | 准确率 |
|---|---|---|---|---|
| 剪枝 fp32（无稀疏 kernel） | 12.76 MB | 0.190 ms | 115k 条/秒 | 98.26%（无微调） |
| 剪枝 fp32 + SPARSE_WEIGHTS | 12.74 MB | 0.181 ms | 122k 条/秒 | 98.26%（无微调） |

**结论**：2:4 稀疏带来约 +6% 吞吐，远低于理论 2 倍——小模型矩阵乘太小，稀疏索引开销抵消大部分收益；引擎体积不变（稀疏不压缩存储，只加速计算）。剪枝本身（无微调）损失 0.8% 精度，配合微调可恢复。**2:4 稀疏适合大矩阵 GEMM，小模型收益有限。**

### 2.6 Fashion-MNIST 鲁棒性验证（量化损失是否普遍）

在更难的同尺寸任务（Fashion-MNIST 10 类衣物图，fp32 基线 92.68% vs MNIST 99.04%）上重复全部量化流程：

| 模型 | 准确率 | 量化损失 |
|---|---|---|
| fp32 基线 | 92.68% | - |
| bf16 混合精度 | 92.68% | **0.00%** |
| int8 动态 | 92.66% | **-0.02%** |
| int8 静态 | 87.45% | **⚠️ -5.23%** |

**结论**：bf16 和 int8 动态量化的无损结论是**普遍规律**（跨任务成立）。int8 静态的 -5.23% 最初归因于"校准数据量不足"，后被 2.7 校准研究推翻——**真正根因是校准数据分布不匹配**（用 MNIST 数据校准 Fashion-MNIST 模型，归一化参数 0.1307 vs 0.2860），换用同数据集校准后损失只有 -0.2%。事故机制与教训见 [01 · 真实教训](./01_quantization_basics.md)。

### 2.7 校准研究：数据量 × 方法扫描

在 Fashion-MNIST 上扫描校准数据量（256/1024/4096/16384 张）× 校准方法（MinMax/MovingAvgMinMax/Histogram，权重统一 per-channel）：

![校准研究](../assets/calibration_study.png)

| 校准方法 | 256张 | 1024张 | 4096张 | 16384张 |
|---|---|---|---|---|
| MinMax（现状） | 92.45% | 92.49% | 92.45% | 92.46% |
| MovingAvgMinMax | 92.45% | 92.46% | 92.48% | 92.45% |
| Histogram | 92.41% | 92.47% | 92.48% | 92.51% |
| ⚠️ 跨数据集校准（MNIST 数据） | - | **87.32%** | - | **87.07%** |

**结论（推翻常见直觉）**：
1. **同数据集校准下，掉精度与数据量无关、与校准方法基本无关**——256 张和 16384 张一样（都只掉 0.2%）。对简单 CNN + 简单任务，MinMaxObserver 足够，Histogram 只带来 +0.05% 边际收益。
2. **真正致命的是校准数据分布不匹配**：掉 5.3-5.6%，且**加更多错分布的数据也救不回来**（16384 张仍 87.07%）。
3. 生产准则：**校准数据的分布必须与真实输入一致（同样的预处理/归一化），比数据量和校准方法重要得多**——原理见 [01 · 进一步学习：预处理为什么致命](./01_quantization_basics.md)。

### 2.8 QAT vs PTQ（量化感知训练）

在 Fashion-MNIST（PTQ 校准错配掉 5.2% 的同一任务）上做 QAT：`prepare_qat_fx` 插入 fake-quant → 微调 3 epoch（lr 1e-4，仅 29 秒）→ convert：

![QAT vs PTQ](../assets/qat_vs_ptq.png)

| 方案 | 准确率 | 备注 |
|---|---|---|
| PTQ（校准错配） | 87.45% | -5.23% |
| fp32 基线 | 92.68% | - |
| **QAT（3 epoch）** | **93.95%** | **+1.27% 反超 fp32 基线** |

**结论**：QAT 比 PTQ 恢复 6.5 个百分点，且比 fp32 基线还高 1.3%（fake-quant 起到噪声正则化作用）。静态量化掉精度时，QAT 是标准解法且成本极低。

### 2.9 训练侧精度对比（bf16/fp16 autocast 训练）

RTX 3060，MNIST，5 epoch，Adam lr=1e-3：

![训练侧对比](../assets/train_amp.png)

| 训练模式 | 总耗时 | 最终精度 |
|---|---|---|
| fp32 | 31.1s | 98.91% |
| bf16 autocast | 21.7s（**1.4×**） | 98.92% |
| fp16 + GradScaler | 18.7s（**1.7×**） | 99.02% |

**结论**：低精度训练不伤精度（三者无损，fp16 略高属波动范围）；bf16 提速 1.4×、fp16 提速 1.7×。bf16 不需要 GradScaler（指数位与 fp32 相同），fp16 必须配 GradScaler（梯度下溢防护）——溢出行为差异的根源见 [01 · 第 1 节](./01_quantization_basics.md)。训练选 fp16+scaler 追求速度，选 bf16 求稳。

### 2.10 ResNet-18 普适性验证（330 万 vs 1100 万参数）

换 ResNet18MNIST（torchvision ResNet-18 适配：conv1 改 3×3/stride1、去 maxpool、1 通道输入，1100 万参数）重复关键对比：

![普适性验证](../assets/resnet18_vs_cnn.png)

| 模型 | 小 CNN 加速比 | ResNet-18 加速比 | ResNet-18 大小 | 精度 |
|---|---|---|---|---|
| fp32 | 1.00x（0.45ms） | 1.00x（4.56ms） | 42.71 MB | 98.87% |
| bf16 混合 | 0.03x（慢） | 0.02x（慢） | 21.43 MB | 98.86% |
| int8 动态 | 0.98x | 0.80x | 42.69 MB | 98.88% |
| int8 静态 | **2.44x** | **1.59x** | 10.78 MB | 98.83% |
| int8 weight-only（GPU） | - | **4.27x** | 42.69 MB | 98.87% |
| int4 weight-only（GPU） | - | 3.95x | 42.70 MB | 98.88% |

**结论**：
1. **int8 静态量化的加速在两个模型上都成立**（2.44x / 1.59x），且模型越大加速比越趋向 1.59 的计算瓶颈值；
2. **weight-only 量化在大模型上反超**（GPU 4.27x/3.95x）——小模型上它不快（per-group 反量化开销占比大），大模型上计算占比上升后优势显现，**证实了"weight-only 适合大模型"的结论**；
3. bf16 CPU 慢是普遍规律（两模型都 0.02-0.03x）；
4. 量化精度损失与模型大小无关（全部 98.8%+ 无损）。

## 3. 踩坑清单汇总

### A. 方法论踩坑（详细展开与教训见 [03 · 本项目踩过的 4 个 bug](./03_reading_benchmarks.md)）

| # | 坑 | 教训 |
|---|---|---|
| A1 | bf16 混合精度把 BatchNorm 也转了，dtype 不匹配报错 | BN 永远保留 fp32 |
| A2 | TRT fp16 引擎喂 fp32 输入，精度静默丢失（99% → 54%，不报错） | 按引擎声明的 IO dtype 动态适配输入输出 |
| A3 | QDQ 节点插到图最前面，破坏拓扑序被 ONNX checker 拒绝 | QDQ 必须插在消费节点之前 |
| A4 | torchao int4 的 dtype 链路 bug（zero_point 硬编码 bf16 与 scale fp32 冲突） | 版本组合 bug 用单层最小例子隔离定位 |

### B. 部署与版本踩坑

| # | 坑 | 处理 |
|---|---|---|
| B1 | TRT 11 移除 `BuilderFlag.INT8` / `IInt8EntropyCalibrator2`，旧教程代码全部失效 | int8 走 QDQ 显式量化 + STRONGLY_TYPED |
| B2 | onnxruntime-gpu 分 CUDA 12/13 二进制，1.31 版需要 CUDA 13 库，与 torch 2.6+cu124 冲突 | 改装 onnxruntime-gpu 1.20.2+ |
| B3 | ORT 的 CUDA/TensorRT EP 动态库（libcudnn/libcublas）需在进程启动前进 LD_LIBRARY_PATH（glibc 启动时缓存，进程内设置无效） | `bench_ort.py` 已自动处理 |
| B4 | ORT 的 TRT-EP 编译对应 TRT 10（libnvinfer.so.10），与 TRT 11 二进制不兼容 | 两者只能选一个 |
| B5 | 激活 per-channel observer 与 FX 算子融合不兼容（scales 通道数冲突报错）；torch 2.6 无 PercentileObserver | 校准扫描只用 per-tensor 激活 observer |
| B6 | torchao 0.13 + torch 2.6 的 int4 打包 dtype 链路 bug：非 (64,32) 形状的 Linear 量化报错，monkeypatch 统一为 bf16 后 fp32 packed 又被 kernel 误读（输出全错） | monkeypatch 统一为 bf16 可用；彻底解决需升级 torch ≥2.11 |
| B7 | make_bf16_mixed 原硬编码 MnistCNN 层名，ResNet 上失效 | 改为按模型拓扑通用的敏感层判断（首个 Conv2d/全部 BN/最后一个 Linear） |
| B8 | torchao weight-only 的 pack/前向算子仅 CUDA 有实现 | 量化 + 评测须在 GPU 上做 |

## 4. 复现方式

```bash
conda run -n py312 python src/train.py            # 训练 fp32 基线 → models/model_fp32.pth
conda run -n py312 python src/quantize.py         # 生成 bf16/int8 动态量化模型并评测准确率
conda run -n py312 python src/quantize_static.py  # int8 静态量化 → model_int8_static.pth
conda run -n py312 python src/calibration_study.py  # 校准研究（数据量×方法+跨数据集对照）
conda run -n py312 python src/qat_study.py        # QAT vs PTQ 对比
conda run -n py312 python src/train_amp_study.py  # 训练侧 bf16/fp16 autocast 对比
conda run -n py312 python src/resnet18_study.py   # ResNet-18 普适性验证
conda run -n py312 python src/export.py           # 统一导出入口
conda run -n py312 python src/serve.py --selftest # FastAPI 服务冒烟测试
conda run -n py312 python src/benchmark.py        # PyTorch CPU 综合基准（分位数+批量扫描）
conda run -n py312 python src/build_trt.py    # 导出 ONNX + 构建 TRT fp32/fp16/bf16/int8 引擎
conda run -n py312 python src/bench_trt.py    # TRT 引擎速度 + 准确率基准
conda run -n py312 python src/bench_torchao.py  # torchao int8/int4 weight-only GPU 对比
conda run -n py312 python src/bench_ort.py    # ONNX Runtime CPU/CUDA provider 对比
conda run -n py312 python src/build_sparse.py # 2:4 剪枝 + 稀疏引擎构建
conda run -n py312 python src/bench_sparse.py # 稀疏引擎对比
conda run -n py312 python src/verify_fmnist.py  # Fashion-MNIST 鲁棒性验证
conda run -n py312 python src/inference.py --model int8_static --images 5  # 端到端推理 Demo
```

依赖：核心 `torch/torchvision`；TRT 部分 `pip install tensorrt onnx onnxruntime-gpu`（tensorrt 11.4 需驱动 580+）；torchao 部分 `pip install torchao==0.13.0`（适配 torch 2.6）；ORT GPU 部分注意 CUDA 12/13 二进制匹配（见踩坑 B2）。

## 自测：查表练习（检验你能不能用好这份手册）

<details>
<summary>1. CPU 部署的最优方案是哪个？三个关键数字？</summary>

int8 静态量化：P50 延迟 0.242 ms（fp32 的 2.4×）、体积 3.20 MB、准确率 98.98%（见 §1 速查表与 §2.1）。
</details>

<details>
<summary>2. GPU 上吞吐最高的方案是什么？大概多少？</summary>

TRT fp16：~227k-356k 条/秒（eager 的 90 倍），精度无损；体积敏感时选 TRT int8（3.30 MB，吞吐同档）。
</details>

<details>
<summary>3. Fashion-MNIST 上静态量化掉到 87.45% 后，QAT 恢复到了多少？</summary>

93.95%——比 fp32 基线 92.68% 还高 1.27%（fake-quant 的噪声正则化效应），微调成本仅 3 epoch / 29 秒（见 §2.8）。
</details>

---

上一篇：[03 · 如何读懂性能数据](./03_reading_benchmarks.md) | 原理篇：[01 · 量化入门](./01_quantization_basics.md) · [02 · 硬件架构](./02_hardware.md)
