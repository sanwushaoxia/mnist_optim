"""生成全部对比图表到 assets/（供 docs/ 教程引用）。

图表清单：
1. quantization_concept.png  — 量化原理示意（fp32→int8 的 scale/zero-point 映射）
2. dynamic_vs_static.png     — 动态 vs 静态量化工作流对比
3. size_bar.png              — 模型体积对比（条形）
4. speed_cpu_bar.png         — CPU 延迟 P50 对比（条形，log 刻度）
5. speed_gpu_bar.png         — GPU 吞吐对比（PyTorch vs TRT，分组条形）
6. tradeoff_scatter.png      — 体积 vs 准确率权衡（散点）
7. hw_precision_matrix.png   — 硬件-精度匹配矩阵（热力图）
8. precision_timeline.png    — 各精度格式的指数位/尾数位示意
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(ASSETS, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"


def save(fig, name):
    fig.savefig(os.path.join(ASSETS, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved:", name)


# ---------- 1. 量化原理示意 ----------
def quantization_concept():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    x = np.linspace(-1.2, 1.2, 400)
    y = np.clip(x, -1, 1)
    ax.plot(x, x, "--", color="gray", lw=1, label="原始 fp32 值")
    ax.plot(x, y, color=C_BLUE, lw=2.5, label="int8 量化后（clamp + 离散化）")
    levels = np.linspace(-1, 1, 9)
    for lv in levels:
        ax.axhline(lv, color=C_ORANGE, alpha=0.25, lw=0.8)
        ax.axvline(lv, color=C_ORANGE, alpha=0.25, lw=0.8)
    sample = np.array([-1.0, -0.6, -0.2, 0.1, 0.5, 1.0])
    q = np.round(sample * 4) / 4
    ax.scatter(sample, q, color=C_RED, zorder=5, s=45, label="离散网格点（256 级）")
    ax.set_title("量化：连续 fp32 → 离散 int8")
    ax.set_xlabel("fp32 数值"); ax.set_ylabel("int8 映射值")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)

    ax = axes[1]
    dtypes = [("fp32\n(1+8+23)", 1, 8, 23, C_BLUE), ("fp16\n(1+5+10)", 1, 5, 10, C_GREEN),
              ("bf16\n(1+8+7)", 1, 8, 7, C_ORANGE), ("int8\n(纯整数)", 0, 0, 8, C_RED)]
    for i, (name, sign, exp, man, color) in enumerate(dtypes):
        x0 = 0
        for bits, lab, c in ((sign, "符号", color), (exp, "指数", color), (man, "尾数", color)):
            if bits:
                ax.barh(i, bits, left=x0, height=0.55, color=c, alpha=0.35 + 0.2 * (lab != "符号"), edgecolor="white")
                ax.text(x0 + bits / 2, i, f"{lab} {bits}bit" if bits >= 5 else f"{bits}", ha="center", va="center", fontsize=8)
            x0 += bits
    ax.set_yticks(range(4)); ax.set_yticklabels([d[0] for d in dtypes])
    ax.set_xlim(0, 34); ax.set_xlabel("位宽（bit）")
    ax.set_title("精度格式对比：bf16 指数位 = fp32（不易溢出），尾数位少（精度低）")
    ax.grid(alpha=0.2, axis="x")
    save(fig, "quantization_concept.png")


# ---------- 2. 动态 vs 静态量化工作流 ----------
def dynamic_vs_static():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, title, steps, colors in (
        (axes[0], "动态量化（无需校准数据）",
         ["fp32 模型", "权重→int8\n（离线一次）", "推理时：\n激活 fp32\n→ 动态量化", "int8 GEMM\n→ 反量化"],
         [C_BLUE, C_ORANGE, C_GREEN, C_RED]),
        (axes[1], "静态量化（需校准数据）",
         ["fp32 模型", "校准：喂 N 张图\n统计每层激活范围", "权重+激活 scale\n→ 全部写死 int8", "推理时：\n全程 int8"],
         [C_BLUE, C_PURPLE, C_ORANGE, C_GREEN]),
    ):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        ax.set_title(title, fontsize=13, pad=10)
        bw, bh = 4.2, 1.6
        positions = [(0.6, 8), (5.2, 8), (0.6, 4.6), (5.2, 4.6)]
        for (px, py), text, c in zip(positions, steps, colors):
            ax.add_patch(plt.Rectangle((px, py - bh / 2), bw, bh, facecolor=c, alpha=0.18, edgecolor=c, lw=2))
            ax.text(px + bw / 2, py, text, ha="center", va="center", fontsize=9.5)
        arrows = [((2.8, 8), (5.2, 8)), ((7.3, 8), (7.3, 4.6 + bh / 2 + 0.15)), ((5.2, 4.6), (2.8, 4.6))]
        for (x1, y1), (x2, y2) in arrows:
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="gray", lw=1.6))
        note = "优点：简单/精度稳\n缺点：激活仍是 fp32\nConv 支持弱，体积/速度收益小" if title.startswith("动态") else \
               "优点：全程 int8，最快最小\n缺点：需代表性校准集\n（校不准会掉精度）"
        ax.text(5, 1.6, note, ha="center", va="center", fontsize=10, color="#333",
                bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#ccc"))
    save(fig, "dynamic_vs_static.png")


# ---------- 3. 模型体积对比 ----------
def size_bar():
    models = ["fp32", "fp32\ncompile", "bf16 混合", "TRT\nfp16", "TRT\nbf16", "int8\n动态", "int8\n静态", "TRT\nint8", "torchao\nint4 w-only"]
    sizes = [12.74, 12.64, 6.47, 6.42, 6.47, 3.44, 3.20, 3.30, 2.24]
    colors = [C_BLUE, "#9CB3D5", C_ORANGE, "#E8B48A", "#E8B48A", C_GREEN, C_GREEN, "#8BC79A", C_PURPLE]
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    bars = ax.bar(models, sizes, color=colors, edgecolor="white")
    for b, v in zip(bars, sizes):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}", ha="center", fontsize=9.5)
    ax.set_ylabel("模型大小（MB）")
    ax.set_title("模型体积对比（MNIST CNN，200 万参数；压缩至 17.6% 仍可用）")
    ax.grid(alpha=0.2, axis="y"); ax.set_axisbelow(True)
    save(fig, "size_bar.png")


# ---------- 4. CPU 延迟对比 ----------
def speed_cpu_bar():
    models = ["fp32", "fp32\ncompile", "int8 动态", "int8 静态", "bf16 混合"]
    p50 = [0.592, 0.510, 0.563, 0.242, 20.9]
    p95 = [0.640, 0.583, 0.578, 0.254, 67.1]
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - w / 2, p50, w, label="P50 延迟", color=C_BLUE)
    ax.bar(x + w / 2, p95, w, label="P95 延迟", color=C_ORANGE)
    for xi, v in zip(x - w / 2, p50):
        ax.text(xi, v * 1.08, f"{v:.3f}", ha="center", fontsize=9)
    for xi, v in zip(x + w / 2, p95):
        ax.text(xi, v * 1.08, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("单条延迟（ms，log 刻度）")
    ax.set_title("CPU 推理延迟（i5-13490F，batch=1）：int8 静态最快，bf16 autocast 慢 40 倍")
    ax.legend(); ax.grid(alpha=0.2, axis="y"); ax.set_axisbelow(True)
    save(fig, "speed_cpu_bar.png")


# ---------- 5. GPU 吞吐对比 ----------
def speed_gpu_bar():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    # PyTorch eager vs TRT (batch=256)
    ax = axes[0]
    groups = ["fp32", "fp16/bf16", "int8"]
    pytorch_vals = [1.8, 3.8, 0]      # k 条/秒（int8 eager 不支持）
    trt_vals = [110, 320, 300]
    x = np.arange(3); w = 0.38
    b1 = ax.bar(x - w / 2, pytorch_vals, w, label="PyTorch eager", color=C_BLUE)
    b2 = ax.bar(x + w / 2, trt_vals, w, label="TensorRT", color=C_ORANGE)
    for b, v in zip(list(b1) + list(b2), pytorch_vals + trt_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v:g}k" if v else "不支持", ha="center", fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("吞吐（千条/秒，batch=256）")
    ax.set_title("GPU 吞吐：TensorRT 比 PyTorch eager 快 30-100 倍")
    ax.legend(); ax.grid(alpha=0.2, axis="y"); ax.set_axisbelow(True)

    # TRT 内部四精度
    ax = axes[1]
    engines = ["fp32", "fp16", "bf16", "int8"]
    thr = [100, 320, 110, 300]
    colors = [C_BLUE, C_GREEN, C_ORANGE, C_RED]
    bars = ax.bar(engines, thr, color=colors, edgecolor="white")
    for b, v in zip(bars, thr):
        ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v}k", ha="center", fontsize=9.5)
    ax.set_ylabel("吞吐（千条/秒，batch=256）")
    ax.set_title("TRT 引擎内部对比：fp16/int8 最快，bf16 只有 fp16 一半\n（Ampere 消费卡 Tensor Core 对 fp16 优化更深）")
    ax.grid(alpha=0.2, axis="y"); ax.set_axisbelow(True)
    save(fig, "speed_gpu_bar.png")


# ---------- 6. 体积 vs 准确率权衡 ----------
def tradeoff_scatter():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # (体积MB, MNIST准确率, 名称, 颜色, 标注偏移)
    pts = [
        (12.64, 99.04, "fp32", C_BLUE, (8, 6)),
        (12.64, 99.04, "fp32+compile", "#9CB3D5", (8, -14)),
        (6.47, 99.04, "bf16 混合", C_ORANGE, (8, 6)),
        (3.44, 99.05, "int8 动态", C_GREEN, (8, 6)),
        (3.20, 98.98, "int8 静态", "#8BC79A", (8, -14)),
        (2.24, 99.04, "int4 weight-only", C_PURPLE, (10, -16)),
    ]
    for size, acc, name, c, offset in pts:
        ax.scatter(size, acc, s=140, color=c, edgecolor="white", zorder=5)
        ax.annotate(name, (size, acc), textcoords="offset points", xytext=offset, fontsize=10,
                    ha="left" if offset[0] > 0 else "right")
    ax.set_xlabel("模型大小（MB）"); ax.set_ylabel("MNIST 测试准确率（%）")
    ax.set_ylim(98.85, 99.25)
    ax.set_title("体积 vs 精度权衡：量化到 int4 仍保持 99%+（MNIST 任务简单）\n更难任务上静态量化会掉精度，见 Fashion-MNIST 验证")
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    save(fig, "tradeoff_scatter.png")


# ---------- 7. 硬件-精度匹配矩阵（英文，便于通用阅读） ----------
def hw_precision_matrix():
    fig, axes = plt.subplots(2, 1, figsize=(11, 9.5), height_ratios=[2.1, 1])
    ax = axes[0]
    plt.rcParams["font.family"] = "DejaVu Sans"  # 此图纯英文，用默认字体更清晰
    rows = ["CPU\n(AVX2+VNNI,\nno AMX)", "GPU\n(Ampere 8.6,\nconsumer)"]
    cols = ["fp32", "fp16", "bf16", "int8", "int4"]
    # 0=slow/unsupported, 1=fair, 2=fast, 3=fastest
    data = np.array([
        [2, 0, 0, 3, 1],   # CPU
        [1, 3, 2, 3, 1],   # GPU
    ], dtype=float)
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=3, aspect="auto", alpha=0.55)
    # 统一格式：第 1 行 = 计算单元/指令集（不支持则 no hw support），
    # 第 2 行 = 测试框架，第 3 行 = 耗时
    notes = [["AVX2 + FMA (oneDNN)\nPyTorch eager\n0.59 ms",
              "no hw support\n(oneDNN simulates fp32)\nPyTorch eager: 22.1 ms",
              "no hw support\n(oneDNN simulates fp32)\nPyTorch eager: 22.4 ms",
              "AVX_VNNI (oneDNN)\nPyTorch int8 static\n0.24 ms*",
              "no operators\nnot tested"],
             ["CUDA cores (no TC)\nTensorRT\n0.2-0.4 ms",
              "Tensor Core (TRT)\nTensorRT fp16\n0.12 ms*",
              "Tensor Core (TRT)\nTensorRT bf16\n0.16 ms",
              "Tensor Core (TRT)\nTensorRT int8 QDQ\n0.1-0.2 ms",
              "Tensor Core (tinygemm)\ntorchao int4 w-only\n~20 ms"]]
    for i in range(2):
        for j in range(5):
            ax.text(j, i, notes[i][j], ha="center", va="center", fontsize=9.5)
    ax.set_xticks(range(5)); ax.set_xticklabels(cols, fontsize=12)
    ax.set_yticks(range(2)); ax.set_yticklabels(rows, fontsize=10.5)
    ax.set_title("Hardware-Precision Fit Matrix:  green = fast, red = slow / unsupported\n"
                 "* = best choice per hardware  (CPU -> int8 static,  GPU -> fp16 / int8)", fontsize=11.5)

    # 下方术语表：图中名词 + 说明
    ax2 = axes[1]
    ax2.axis("off")
    glossary = [
        ("AVX2 / FMA", "x86 CPU vector instructions (256-bit), 8 fp32 ops per instruction; the baseline fast path"),
        ("AVX_VNNI", "int8 dot-product instruction (vpdpbusd); 4 groups of 8-bit multiply-add per instruction"),
        ("AMX", "Intel server-CPU tile matrix engine (Sapphire Rapids+); absent on consumer desktop CPUs"),
        ("oneDNN", "Intel's CPU DL kernel library; PyTorch CPU backend. Simulates fp16/bf16 via fp32 when no hw support"),
        ("CUDA cores", "NVIDIA general FP32 units; fp32 runs here (no Tensor Core support for fp32 matmul)"),
        ("Tensor Core (TC)", "NVIDIA matrix-multiply unit (Ampere 3rd gen: fp16/bf16/int8/int4), no fp32 mode"),
        ("TRT / int8 QDQ", "TensorRT inference engine; QDQ = explicit quantization via Quantize/Dequantize nodes"),
        ("tinygemm", "CUDA int8/int4 weight-only GEMM kernel (torchao); dequantizes weights per-group"),
        ("PyTorch eager", "PyTorch default op-by-op execution (no graph compilation/fusion)"),
        ("not tested", "int4 on CPU: no operators in PyTorch/torchao; timing never measured in this project"),
    ]
    ax2.set_xlim(0, 1); ax2.set_ylim(0, len(glossary))
    ax2.text(0.0, len(glossary) - 0.25, "Glossary", fontsize=12, fontweight="bold")
    for i, (term, desc) in enumerate(glossary):
        y = len(glossary) - 1.35 - i
        ax2.text(0.01, y, term, fontsize=9.5, fontweight="bold", va="top")
        ax2.text(0.24, y, desc, fontsize=9.5, va="top", color="#333333")
    fig.subplots_adjust(hspace=0.12)
    save(fig, "hw_precision_matrix.png")


# ---------- 8. 执行流程差异 ----------
def precision_timeline():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("同一次推理的时间线：PyTorch eager vs TensorRT（速度差 30-100 倍的根源）", fontsize=12.5)

    # eager: 一串小方块（每 op 一次分发）
    y1 = 7
    ops = ["input", "Conv1", "BN1", "ReLU1", "Conv2", "BN2", "ReLU2", "...", "Gemm", "out"]
    ax.text(0.1, y1 + 1.1, "PyTorch eager：每个 op 走一次 Python→C++ 分发（小模型上分发占比 >80%）", fontsize=10)
    for i, op in enumerate(ops):
        x0 = 0.3 + i * 1.32
        ax.add_patch(plt.Rectangle((x0, y1 - 0.55), 1.15, 1.1, facecolor=C_BLUE, alpha=0.25, edgecolor=C_BLUE))
        ax.text(x0 + 0.575, y1, op, ha="center", va="center", fontsize=8.5)
        if i < len(ops) - 1:
            ax.annotate("", xy=(x0 + 1.15, y1), xytext=(x0 + 1.32, y1), arrowprops=dict(arrowstyle="<-", color="gray", lw=1))

    # TRT: 融合大块
    y2 = 3
    ax.text(0.1, y2 + 1.1, "TensorRT：Conv+BN+ReLU 融合成一个 kernel，一次调用执行整图 + kernel 自动调优", fontsize=10)
    fused = [("Conv+BN+ReLU ×3", 4.2, C_GREEN), ("Flatten", 1.2, C_GREEN), ("Gemm+ReLU", 2.2, C_GREEN), ("Gemm", 1.2, C_GREEN)]
    x0 = 0.3
    for name, w, c in fused:
        ax.add_patch(plt.Rectangle((x0, y2 - 0.55), w, 1.1, facecolor=c, alpha=0.35, edgecolor=c, lw=2))
        ax.text(x0 + w / 2, y2, name, ha="center", va="center", fontsize=9.5)
        x0 += w + 0.3
    ax.text(0.3, y2 - 1.6, "↑ 融合消除中间显存读写；构建期对本机 28 个 SM 实测选最快 kernel", fontsize=9.5, color="#555")
    save(fig, "precision_timeline.png")


# ---------- 9. 校准研究：数据量×方法 ----------
def calibration_chart():
    fig, ax = plt.subplots(figsize=(10, 5))
    sizes = [256, 1024, 4096, 16384]
    data = {
        "MinMax（现状）": [0.9245, 0.9249, 0.9245, 0.9246],
        "MovingAvgMinMax": [0.9245, 0.9246, 0.9248, 0.9245],
        "Histogram": [0.9241, 0.9247, 0.9248, 0.9251],
    }
    colors = [C_ORANGE, C_GREEN, C_PURPLE]
    for (name, vals), c in zip(data.items(), colors):
        ax.plot(range(len(sizes)), [v * 100 for v in vals], "o-", color=c, label=name, lw=2, markersize=7)
    ax.axhline(92.68, color=C_BLUE, ls="--", lw=1.5, label="fp32 基线 92.68%")
    ax.axhline(87.32, color=C_RED, ls=":", lw=1.5, label="⚠️ 跨数据集校准 87.32%（部署事故）")
    ax.set_xticks(range(len(sizes))); ax.set_xticklabels([f"{n}张" for n in sizes])
    ax.set_xlabel("校准数据量"); ax.set_ylabel("Fashion-MNIST 准确率（%）")
    ax.set_ylim(86.5, 93.3)
    ax.set_title("校准研究：同数据集校准只掉 0.2% 且与方法/数据量无关\n真正致命的是校准数据分布不匹配（-5.4%）")
    ax.legend(fontsize=9.5, loc="lower right"); ax.grid(alpha=0.3)
    save(fig, "calibration_study.png")


# ---------- 10. QAT vs PTQ ----------
def qat_chart():
    fig, ax = plt.subplots(figsize=(9, 5))
    names = ["PTQ 静态\n（校准错配）", "fp32 基线", "QAT\n（微调 3 epoch）"]
    vals = [87.45, 92.68, 93.95]
    colors = [C_RED, C_BLUE, C_GREEN]
    bars = ax.bar(names, vals, color=colors, edgecolor="white", width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}%", ha="center", fontsize=12)
    ax.annotate("QAT 恢复 +6.5%\n且比 fp32 基线高 1.3%\n（量化噪声的正则化效应）",
                xy=(2, 93.95), xytext=(1.35, 89.5), fontsize=10.5,
                arrowprops=dict(arrowstyle="->", color="gray"), ha="center")
    ax.set_ylabel("Fashion-MNIST 准确率（%）"); ax.set_ylim(85, 95.5)
    ax.set_title("QAT vs PTQ：量化感知训练让精度反超 fp32 基线（仅 29 秒微调）")
    ax.grid(alpha=0.2, axis="y"); ax.set_axisbelow(True)
    save(fig, "qat_vs_ptq.png")


# ---------- 11. 训练侧精度对比 ----------
def train_amp_chart():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    modes = ["fp32", "bf16\nautocast", "fp16\n+GradScaler"]
    times = [31.1, 21.7, 18.7]
    accs = [98.91, 98.92, 99.02]
    ax = axes[0]
    bars = ax.bar(modes, times, color=[C_BLUE, C_ORANGE, C_GREEN], edgecolor="white")
    for b, v in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}s", ha="center", fontsize=11)
    ax.set_ylabel("5 epoch 训练总耗时（秒，RTX 3060）")
    ax.set_title("训练速度：bf16 快 1.4 倍，fp16 快 1.7 倍")
    ax.grid(alpha=0.2, axis="y"); ax.set_axisbelow(True)

    ax = axes[1]
    bars = ax.bar(modes, accs, color=[C_BLUE, C_ORANGE, C_GREEN], edgecolor="white")
    for b, v in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}%", ha="center", fontsize=11)
    ax.set_ylim(98.5, 99.3); ax.set_ylabel("MNIST 最终精度（%）")
    ax.set_title("训练精度：三者无损（低精度训练不伤精度）")
    ax.grid(alpha=0.2, axis="y"); ax.set_axisbelow(True)
    save(fig, "train_amp.png")


# ---------- 12. ResNet-18 vs 小模型普适性 ----------
def resnet_chart():
    fig, ax = plt.subplots(figsize=(11, 5))
    models = ["fp32", "bf16 混合", "int8 动态", "int8 静态"]
    cnn = [1.0, 0.03, 0.98, 2.44]       # MnistCNN 加速比
    rn = [1.0, 0.02, 0.80, 1.59]        # ResNet18
    x = np.arange(len(models)); w = 0.38
    ax.bar(x - w / 2, cnn, w, label="MnistCNN（330 万参数）", color=C_BLUE)
    ax.bar(x + w / 2, rn, w, label="ResNet-18（1100 万参数）", color=C_ORANGE)
    for xi, v in zip(x - w / 2, cnn):
        ax.text(xi, v + 0.04 if v < 0.1 else v + 0.05, f"{v:.2f}x", ha="center", fontsize=9.5)
    for xi, v in zip(x + w / 2, rn):
        ax.text(xi, v + 0.04 if v < 0.1 else v + 0.05, f"{v:.2f}x", ha="center", fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("CPU P50 加速比（vs fp32）")
    ax.set_title("普适性验证：int8 静态量化在两个模型上都加速（2.4x / 1.6x），bf16 都慢\n（GPU 上 weight-only 反超：int8 w-only 4.27x / int4 w-only 3.95x）")
    ax.legend(); ax.grid(alpha=0.2, axis="y"); ax.set_axisbelow(True)
    save(fig, "resnet18_vs_cnn.png")


if __name__ == "__main__":
    quantization_concept()
    dynamic_vs_static()
    size_bar()
    speed_cpu_bar()
    speed_gpu_bar()
    tradeoff_scatter()
    hw_precision_matrix()
    precision_timeline()
    calibration_chart()
    qat_chart()
    train_amp_chart()
    resnet_chart()
    print(f"\n全部图表已生成到 {ASSETS}")
