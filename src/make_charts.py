"""生成全部对比图表到 assets/（供 docs/ 教程引用）。

图表清单：
1. quantization_concept.png  — 量化三步示意（除 scale → 四舍五入 → clip）+ 精度格式位域对比
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
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    # (0,0) 第①步：除以 scale —— fp32 小范围拉伸到 [-127, 127] 的整数格
    ax = axes[0][0]
    s = 1.4 / 127
    ax.set_xlim(-135, 135); ax.set_ylim(-0.7, 3.1)
    ax.axhline(0.5, color="black", lw=2)                               # 缩放后数轴
    for t in range(-130, 131):                                         # 整数格
        ax.plot([t, t], [0.44, 0.56], color=C_ORANGE, lw=0.8, alpha=0.5)
    for t in (-127, -64, 0, 64, 127):
        ax.text(t, 0.14, str(t), ha="center", fontsize=9)
    fp = np.array([-1.4, -0.5, 0.37, 1.0])
    q = fp / s
    fv = fp * 25                                                       # fp32 轴视觉放大（示意，未按比例）
    ax.plot([fv.min() - 8, fv.max() + 8], [2.0, 2.0], color="gray", lw=2)
    for f, qq in zip(fv, q):
        ax.annotate("", xy=(qq, 0.64), xytext=(f, 1.88),
                    arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.2, alpha=0.7))
    ax.scatter(fv, [2.0] * 4, color=C_RED, s=50, zorder=5)
    ax.scatter(q, [0.5] * 4, color=C_BLUE, s=50, zorder=5)
    for f, lab in zip(fv, ["-1.4", "-0.5", "0.37", "1.0"]):
        ax.text(f, 2.2, lab, ha="center", fontsize=9, color=C_RED)
    for qq, lab in zip(q, ["-127", "-45.4", "33.6", "90.7"]):
        ax.text(qq, 0.74, lab, ha="center", fontsize=8, color=C_BLUE)
    ax.text(-128, 2.6, "÷ s", fontsize=14, color=C_PURPLE)
    ax.text(0, 2.9, "s = 范围/127 = 1.4/127 ≈ 0.011（fp32 里一个格子的宽度；fp32 轴未按比例）",
            ha="center", fontsize=9, color="dimgray")
    ax.text(-133, 2.0, "fp32 原值\n（连续）", fontsize=9, color="gray", va="center")
    ax.set_yticks([]); ax.set_xlabel("÷s 后的值（格）")
    ax.set_title("① 除以 scale：fp32 的连续范围 → 255 个整数格")
    ax.spines[["top", "right", "left"]].set_visible(False)

    # (0,1) 第②步：四舍五入 —— 误差 ≤ ±s/2
    ax = axes[0][1]
    xs = np.linspace(-4.5, 4.5, 500)
    ax.plot(xs, xs, "--", color="gray", lw=1, label="缩放后的原值（连续）")
    for lv in np.arange(-5, 6):  # 只画水平段：跳变处没有中间值，不画纵向连线
        ax.hlines(lv, lv - 0.5, lv + 0.5, color=C_BLUE, lw=2.5)
    ax.hlines([], [], [], color=C_BLUE, lw=2.5, label="round（取最近格）")
    for lv in np.arange(-4, 5):
        ax.axhline(lv, color=C_ORANGE, alpha=0.3, lw=0.8)
    samples = np.array([-3.6, -1.4, 0.3, 2.6])
    ax.scatter(samples, np.round(samples), color=C_RED, s=50, zorder=5, label="取整后落在格上")
    ax.annotate("", xy=(0.3, 0), xytext=(0.3, 0.3),
                arrowprops=dict(arrowstyle="<->", color=C_RED, lw=1.5))
    ax.text(-0.4, 0.15, "≤ s/2", fontsize=10, color=C_RED, ha="right")
    ax.set_xlim(-4.8, 4.8); ax.set_ylim(-4.8, 4.8)
    ax.set_xlabel("缩放后的值（格）\n（每次取整最多拖回 0.5 格 → 误差 ≤ ±s/2）", fontsize=10)
    ax.set_ylabel("取整后的值（格）")
    ax.set_title("② 四舍五入：只在半格内丢精度")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.2)

    # (1,0) 第③步：clip —— 越界值全部塌缩到边界
    ax = axes[1][0]
    xs = np.linspace(-155, 155, 500)
    ax.plot(xs, np.clip(xs, -127, 127), color=C_BLUE, lw=2.5, label="clip 到 [-127, 127]")
    for side in (-1, 1):
        ax.axvspan(side * 127, side * 155, color=C_RED, alpha=0.12)
    ax.axhline(127, color=C_ORANGE, lw=1, ls=":"); ax.axhline(-127, color=C_ORANGE, lw=1, ls=":")
    ax.scatter([100], [100], color=C_GREEN, s=50, zorder=5, label="界内：正常保留")
    ax.scatter([130, 150], [127, 127], color=C_RED, s=50, zorder=5, label="越界：塌缩到边界")
    for xx in (130, 150):
        ax.annotate("", xy=(xx, 125.2), xytext=(xx, 127.8),
                    arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2))
    ax.annotate("越界的 130 和 150 塌缩成\n同一个 127，再也无法区分",
                xy=(128, 125.5), xytext=(-30, 58), ha="center", fontsize=9, color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1))
    ax.text(-152, 60, "越界区", fontsize=9, color=C_RED, va="center")
    ax.text(152, 60, "越界区", fontsize=9, color=C_RED, va="center", ha="right")
    ax.set_xlim(-155, 155); ax.set_ylim(-135, 140)
    ax.set_xticks([-127, -64, 0, 64, 127])
    ax.set_xlabel("缩放后的值（格）"); ax.set_ylabel("最终量化值 q")
    ax.set_title("③ clip：超范围的值强制压到边界（不可恢复）")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.2)

    # (1,1) 四种格式的位域划分（保留原右图）
    ax = axes[1][1]
    dtypes = [("fp32\n(1+8+23)", 1, 8, 23, C_BLUE), ("fp16\n(1+5+10)", 1, 5, 10, C_GREEN),
              ("bf16\n(1+8+7)", 1, 8, 7, C_ORANGE), ("int8\n(纯整数)", 0, 0, 8, C_RED),
              ("int4\n(纯整数)", 0, 0, 4, C_PURPLE)]
    for i, (name, sign, exp, man, color) in enumerate(dtypes):
        x0 = 0
        is_int = name.startswith("int")
        for bits, lab, c in ((sign, "符号", color), (exp, "指数", color), (man, "尾数", color)):
            if bits:
                label = (f"整数 {bits}bit" if bits >= 8 else f"{bits}bit") if is_int \
                    else (f"{lab} {bits}bit" if bits >= 5 else f"{bits}")
                ax.barh(i, bits, left=x0, height=0.55, color=c, alpha=0.35 + 0.2 * (lab != "符号"), edgecolor="white")
                ax.text(x0 + bits / 2, i, label, ha="center", va="center", fontsize=8)
            x0 += bits
    ax.set_yticks(range(len(dtypes))); ax.set_yticklabels([d[0] for d in dtypes])
    ax.set_xlim(0, 34); ax.set_xlabel("位宽（bit）")
    ax.set_title("④ 四种格式的位域：指数保范围，尾数定精度")
    ax.grid(alpha=0.2, axis="x")
    fig.tight_layout(h_pad=3.0, w_pad=3.0)
    save(fig, "quantization_concept.png")


# ---------- 2. 动态 vs 静态量化工作流 ----------
def dynamic_vs_static():
    from matplotlib.patches import FancyBboxPatch
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4))
    fig.tight_layout(w_pad=4.0)
    specs = [
        ("动态量化", "无需校准数据",
         "仅权重 → int8\n（s_w = max|W|/127 写死；激活的 scale 此步不定）",
         "激活 fp32 进 → 现场统计范围 s_x\n→ int8 GEMM → 反量化回 fp32 出",
         "优点：零校准数据，s_x 实时贴合输入\n缺点：激活仍是 fp32，层间传输没省\n只支持 Linear（Conv 支持弱），收益小",
         2),
        ("静态量化", "需校准数据",
         "校准：喂 N 张无标签图，统计每层激活范围\n权重 + 激活的 scale 都写死（激活部分是关键差异）",
         "激活 int8 进 → int8 GEMM → requantize\n→ int8 出（全程整数，不经浮点）",
         "优点：全程 int8，最快最小；Conv 也量化\n缺点：校准分布必须与真实输入一致\n（错配会掉精度，见 01 · 第 6 节）",
         1),
    ]
    for ax, (name, sub, off_text, infer_text, note, hi) in zip(axes, specs):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        ax.set_title(f"{name}（{sub}）", fontsize=13, pad=10)
        bw, x0 = 7.0, 2.5
        rows = [("模型", 8.9, 1.0, "fp32 模型", C_BLUE),
                ("离线\n（一次性）", 6.2, 1.8, off_text, C_ORANGE),
                ("推理\n（每次前向）", 3.2, 1.9, infer_text, C_GREEN)]
        for label, y, h, text, c in rows:
            ax.text(0.2, y, label, fontsize=10, color="dimgray", fontweight="bold",
                    ha="left", va="center")
            w = 3.4 if label == "模型" else bw
            bx = 5.5 - w / 2
            ax.add_patch(FancyBboxPatch((bx, y - h / 2), w, h, boxstyle="round,pad=0.08",
                                        facecolor=c, alpha=0.16, edgecolor=c, lw=1.8))
            ax.text(bx + w / 2, y, text, ha="center", va="center", fontsize=9.8)
        for y1, y2 in ((8.4, 7.15), (5.25, 4.2)):
            ax.annotate("", xy=(5.5, y2), xytext=(5.5, y1),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1.8))
        if hi is not None:  # 高亮「激活 scale 在哪一步确定」——两者本质区别
            label, y, h, text, c = rows[hi]
            ax.add_patch(FancyBboxPatch((5.5 - bw / 2 - 0.28, y - h / 2 - 0.28), bw + 0.56, h + 0.56,
                                        boxstyle="round,pad=0.08", fill=False,
                                        edgecolor=C_RED, ls="--", lw=1.7))
            ax.text(5.5, y + h / 2 + 0.66, "★ 激活 scale 在这一步确定",
                    ha="center", fontsize=10, color=C_RED, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=C_RED, lw=1))
        ax.text(5.0, 1.05, note, ha="center", va="center", fontsize=9.5, color="#333",
                linespacing=1.5,
                bbox=dict(boxstyle="round", facecolor="#f7f7f7", edgecolor="#ccc"))
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
    fig, ax = plt.subplots(figsize=(11, 5.2))
    plt.rcParams["font.family"] = "DejaVu Sans"  # 此图纯英文，用默认字体更清晰
    rows = ["CPU\n(AVX2+VNNI,\nno AMX)", "GPU\n(Ampere 8.6,\nconsumer)"]
    cols = ["fp32", "bf16", "fp16", "int8", "int4"]
    # 0=slow/unsupported, 1=fair, 2=fast, 3=fastest（列序：fp32→bf16→fp16 按精度谱系排列）
    data = np.array([
        [2, 0, 0, 3, 1],   # CPU
        [1, 2, 3, 3, 1],   # GPU
    ], dtype=float)
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=3, aspect="auto", alpha=0.55)
    # 统一格式：第 1 行 = 计算单元/指令集（不支持则 no hw support），
    # 第 2 行 = 测试框架，第 3 行 = 耗时，第 4 行 = 该精度下模型体积
    notes = [["AVX2 + FMA (oneDNN)\nPyTorch eager\n0.590 ms\n12.64 MB",
              "no hw support\n(oneDNN simulates fp32)\nPyTorch eager: 22.4 ms\n6.47 MB",
              "no hw support\n(oneDNN simulates fp32)\nPyTorch eager: 22.1 ms\n6.32 MB",
              "AVX_VNNI (oneDNN)\nPyTorch int8 static\n0.242 ms*\n3.20 MB",
              "no operators\nnot tested\n—"],
             ["CUDA cores (no TC)\nTensorRT\n0.180 ms\n12.74 MB",
              "Tensor Core (TRT)\nTensorRT bf16\n0.158 ms\n6.47 MB",
              "Tensor Core (TRT)\nTensorRT fp16\n0.125 ms*\n6.42 MB",
              "Tensor Core (TRT)\nTensorRT int8 QDQ\n0.085 ms*\n3.30 MB",
              "Tensor Core (tinygemm)\ntorchao int4 w-only\n16.05 ms\n2.24 MB"]]
    for i in range(2):
        for j in range(5):
            ax.text(j, i, notes[i][j], ha="center", va="center", fontsize=9.5)
    ax.set_xticks(range(5)); ax.set_xticklabels(cols, fontsize=12)
    ax.set_yticks(range(2)); ax.set_yticklabels(rows, fontsize=10.5)
    ax.set_title("Hardware-Precision Fit Matrix:  green = fast, red = slow / unsupported\n"
                 "* = best choice per hardware  (CPU -> int8 static,  GPU -> fp16 / int8)\n"
                 "cell:  compute unit / instruction set  ->  framework  ->  latency (batch=1)  ->  model size", fontsize=11.5)
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
