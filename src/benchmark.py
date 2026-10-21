"""综合基准测试 v2：PyTorch 全部模型形态的体积、延迟分位数、批量吞吐。

模型形态：
- fp32 基线 / bf16 混合精度 / int8 动态量化 / int8 静态量化 / fp32+torch.compile

指标：
- 体积（MB）
- 单条延迟 P50/P95/P99（batch=1，200 次测量）
- 吞吐（batch ∈ {64, 256, 1024}，条/秒）

注意：int8 静态量化仅 CPU；bf16 CPU 走 autocast；compile 仅 CPU 对比（GPU eager 已有 TRT 对比）。
"""
import copy
import os
import statistics
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MnistCNN
from quantize import make_bf16_mixed, make_int8 as make_int8_dynamic
from train import get_loaders

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
WARMUP, N_MEASURE, BATCHES_SCAN = 20, 200, (64, 256, 1024)


def load_fp32() -> nn.Module:
    model = MnistCNN().cpu().eval()
    model.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "model_fp32.pth"), map_location="cpu", weights_only=True
        )
    )
    return model


def load_int8_static() -> nn.Module:
    m = torch.jit.load(os.path.join(MODEL_DIR, "model_int8_static.pth"), map_location="cpu")
    m.eval()
    return m


def build_models() -> dict[str, tuple[nn.Module, str]]:
    """返回 {名称: (模型, 保存文件名或 'script'/'derived')}。"""
    fp32 = load_fp32()
    bf16 = make_bf16_mixed(fp32)
    bf16._is_bf16 = True
    int8_dyn = make_int8_dynamic(fp32)
    int8_stat = load_int8_static()
    compiled = torch.compile(copy.deepcopy(fp32))
    return {
        "fp32": (fp32, "model_fp32.pth"),
        "fp32_compile": (compiled, "derived"),
        "bf16_mixed": (bf16, "model_bf16_mixed.pth"),
        "int8_dynamic": (int8_dyn, "model_int8.pth"),
        "int8_static": (int8_stat, "model_int8_static.pth"),
    }


def size_mb(fname: str, model: nn.Module) -> float:
    if fname == "derived":
        # compile 模型从源 fp32 权重计（权重相同，只多缓存代码）
        return os.path.getsize(os.path.join(MODEL_DIR, "model_fp32.pth")) / 1024 / 1024
    return os.path.getsize(os.path.join(MODEL_DIR, fname)) / 1024 / 1024


def forward_once(model, x, device, use_bf16):
    if use_bf16:
        with torch.autocast(device.type, dtype=torch.bfloat16):
            model(x)
    else:
        model(x)


@torch.no_grad()
def bench_latency(model, device, use_bf16: bool = False) -> dict:
    """batch=1 单条延迟，N_MEASURE 次取分位数（ms）。"""
    dummy = torch.randn(1, 1, 28, 28, device=device)
    for _ in range(WARMUP):
        forward_once(model, dummy, device, use_bf16)
    times = []
    for _ in range(N_MEASURE):
        t0 = time.perf_counter()
        forward_once(model, dummy, device, use_bf16)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "p50": statistics.median(times),
        "p95": times[int(N_MEASURE * 0.95)],
        "p99": times[int(N_MEASURE * 0.99)],
    }


@torch.no_grad()
def bench_throughput(model, device, batch: int, use_bf16: bool = False) -> float:
    """给定 batch 的吞吐（条/秒）。"""
    dummy = torch.randn(batch, 1, 28, 28, device=device)
    for _ in range(WARMUP):
        forward_once(model, dummy, device, use_bf16)
    ITERS = max(10, min(50, 25600 // batch))
    t0 = time.perf_counter()
    for _ in range(ITERS):
        forward_once(model, dummy, device, use_bf16)
    return ITERS * batch / (time.perf_counter() - t0)


@torch.no_grad()
def eval_acc(model, loader, device, use_bf16=False) -> float:
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if use_bf16:
            with torch.autocast(device.type, dtype=torch.bfloat16):
                preds = model(images).argmax(1)
        else:
            preds = model(images).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    device = torch.device("cpu")
    _, test_loader = get_loaders()
    models = build_models()

    print(f"=== PyTorch CPU 综合基准（延迟 {N_MEASURE} 次分位数）===\n")
    print(f"{'模型':14s} {'大小MB':>8s} {'P50':>8s} {'P95':>8s} {'P99':>8s} {'准确率':>8s}")
    for name, (model, fname) in models.items():
        use_bf16 = getattr(model, "_is_bf16", False)
        size = size_mb(fname, model)
        lat = bench_latency(model, device, use_bf16)
        acc = eval_acc(model, test_loader, device, use_bf16)
        print(f"{name:14s} {size:8.2f} {lat['p50']:8.3f} {lat['p95']:8.3f} {lat['p99']:8.3f} {acc:8.4f}")

    print(f"\n=== 吞吐扫描（CPU, 条/秒）===")
    header = f"{'模型':14s}" + "".join(f" batch={b:>5d}" for b in BATCHES_SCAN)
    print(header)
    for name, (model, _) in models.items():
        use_bf16 = getattr(model, "_is_bf16", False)
        row = f"{name:14s}"
        for b in BATCHES_SCAN:
            thr = bench_throughput(model, device, b, use_bf16)
            row += f" {thr:11,.0f}"
        print(row)


if __name__ == "__main__":
    main()
