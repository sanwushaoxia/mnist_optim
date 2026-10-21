# 6 个新方向：校准曲线 / QAT / 训练侧 bf16 / ResNet-18 / 服务 Demo / 导出整理

## 现状（已确认）
- 静态量化目前：`get_default_qconfig_mapping("x86")` 默认 MinMaxObserver，仅 1024 张校准 → Fashion-MNIST 掉 5.2%（待修复的缺口）
- 训练循环无 AMP 代码，加 autocast+GradScaler 容易
- 量化/导出/评测函数都是黑盒接受 nn.Module，ResNet-18 换入无障碍；需 1 通道适配（conv1 改 3×3/stride1）并同步硬编码的 `(1,1,28,28)`

## 实施内容

**① 校准改进曲线实验（`src/calibration_study.py`）**
在 Fashion-MNIST（fp32 基线 92.68%，PTQ 掉 5.2% 的任务）上扫描：
- 校准数据量：256 / 1024 / 4096 / 16384 张
- 校准方法：MinMax（现状）/ MovingAverageMinMax / Percentile(99.9) / Histogram
输出精度矩阵 + 曲线图，回答"校准怎么做好"。GPU 训练好的 `model_fmnist_fp32.pth` 已存在，直接复用。

**② QAT vs PTQ（`src/qat_study.py`）**
Fashion-MNIST 上：加载 fp32 权重 → FX `prepare_qat_fx`（插入 fake-quant）→ 微调 3 epoch（lr 1e-4）→ convert。与 PTQ（掉到 87.45%）和 fp32 基线（92.68%）三方对比。

**③ 训练侧 bf16 对比（`src/train_amp_study.py`）**
MNIST 上 fp32 训练 vs bf16 autocast 训练（无 GradScaler，bf16 不需要）vs fp16 autocast + GradScaler：对比每 epoch 耗时、最终精度。GPU 上预期 bf16 快 1.5-2 倍且精度无损。

**④ ResNet-18 普适性验证（`src/resnet18_study.py`）**
`model.py` 加 `build_model(name)` 工厂：MnistCNN / ResNet18MNIST（conv1 改 3×3/stride1/p1 保留 1 通道、去掉 maxpool，约 1100 万参数）。训练 fp32 → 重复关键对比（bf16 混合/int8 动态/静态/体积/速度），验证小模型结论是否普适。同步修正 build_trt.py/bench_trt.py/quantize_static.py 的硬编码 shape（改为从模型推导或参数化）。

**⑤ FastAPI 服务 Demo（`src/serve.py`）**
`/predict` 端点（上传图片 → 指定模型推理），启动时按设备自动选模型；`/health` 端点列可用模型。用 TestClient 做冒烟验证（不长期起服务）。

**⑥ 统一导出入口（`src/export.py`）**
`python src/export.py --model int8_static --format all`：统一导出 TorchScript/ONNX/TRT 引擎到 `models/`，打印产物清单与大小。

**⑦ 文档更新**
- 新图表：校准曲线图、QAT vs PTQ 柱状图、训练耗时对比、ResNet-18 vs 小模型对比 → `assets/`（visual-judge 验收）
- `docs/04_results.md` 加 4 个新章节；`docs/01` 补 QAT 一节；README 脚本清单/学习路径更新

## 执行顺序
①② 共用 Fashion-MNIST fp32 权重可并行写；③④ 训练较长放后台；⑤⑥ 纯工程最后写；⑦ 汇总。运行时脚本串行（GPU 争用）。

## 验收
- 每个实验脚本运行成功且数据合理
- 新图表过 visual-judge
- docs 交叉链接完整，README 结构表更新