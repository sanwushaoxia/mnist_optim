"""ResNet-18 普适性验证：小模型（MnistCNN 330 万参数）的结论在更大模型
（ResNet18MNIST 1100 万参数）上是否成立。

重复关键对比：fp32 训练 → bf16 混合 / int8 动态 / int8 静态量化 →
体积 / CPU 速度 / 精度。重点验证：
- int8 静态量化的加速比是否更高（大模型计算占比更大）
- torchao int4 weight-only 的速度是否反超（大矩阵 GEMM）
- 量化精度损失是否仍然很小
"""
import copy
import io
import os
import statistics
import time

import torch
import torch.nn as nn
from torchao.quantization import Int8WeightOnlyConfig, Int4WeightOnlyConfig, quantize_
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import build_model, MnistCNN
from quantize import make_bf16_mixed, make_int8 as make_int8_dynamic
from quantize_static import make_int8_static

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
EPOCHS = 3  # ResNet-18 收敛快，3 epoch 足够
BATCH = 256
LR = 1e-3
WARMUP, N_MEASURE = 10, 100


def get_loaders():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    return (
        DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(test_set, batch_size=512, shuffle=False, num_workers=2),
    )


def train_fp32(model, device) -> None:
    train_loader, test_loader = get_loaders()
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS + 1):
        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
        print(f"  epoch {epoch}/{EPOCHS} loss={running / len(train_loader.dataset):.4f}")
    model.cpu().eval()


@torch.no_grad()
def eval_acc(model, device) -> float:
    _, test_loader = get_loaders()
    model = model.to(device).eval()
    correct = total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def bench_gpu(model, device, batch: int = 1) -> float:
    """GPU P50 延迟（ms/条）。"""
    dummy = torch.randn(batch, 1, 28, 28, device=device)
    m = model.to(device).eval()
    for _ in range(WARMUP):
        m(dummy)
    torch.cuda.synchronize()
    times = []
    for _ in range(N_MEASURE):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m(dummy)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


@torch.no_grad()
def bench_cpu(model, batch: int = 1) -> float:
    """CPU P50 延迟（ms/条）。"""
    dummy = torch.randn(batch, 1, 28, 28)
    m = model.cpu().eval()
    for _ in range(WARMUP):
        m(dummy)
    times = []
    for _ in range(N_MEASURE):
        t0 = time.perf_counter()
        m(dummy)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


@torch.no_grad()
def bench_cpu_bf16(model, batch: int = 1) -> float:
    """bf16 混合模型的 CPU P50 延迟（autocast 下）。"""
    dummy = torch.randn(batch, 1, 28, 28)
    m = model.cpu().eval()
    for _ in range(WARMUP):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            m(dummy)
    times = []
    for _ in range(N_MEASURE):
        t0 = time.perf_counter()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            m(dummy)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def size_of(model) -> float:
    buf = io.BytesIO()
    torch.save(model, buf)
    return buf.tell() / 1024 / 1024


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = "resnet18"
    weights_path = os.path.join(MODEL_DIR, f"model_{tag}_fp32.pth")

    # 1) 训练（或复用已有权重）
    if os.path.exists(weights_path):
        model = build_model(tag).cpu().eval()
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        print(f"复用已训练权重: {weights_path}")
    else:
        print(f"=== 训练 {tag}（fp32, {EPOCHS} epochs, {device}）===")
        model = build_model(tag)
        train_fp32(model, device)
        torch.save(model.state_dict(), weights_path)
        print(f"已保存: {weights_path}")

    acc_fp32 = eval_acc(model, torch.device("cpu"))
    size_fp32 = size_of(model)
    lat_fp32 = bench_cpu(model)
    print(f"\nfp32: {size_fp32:.2f} MB | CPU P50 {lat_fp32:.3f} ms | acc {acc_fp32:.4f}")

    # 2) 各量化形态
    print(f"\n=== {tag} 量化对比（CPU P50 延迟）===")
    print(f"{'模型':<18s} {'大小MB':>8s} {'P50ms':>8s} {'加速比':>7s} {'准确率':>8s}")
    print(f"{'fp32':<18s} {size_fp32:8.2f} {lat_fp32:8.3f} {1.0:7.2f}x {acc_fp32:8.4f}")

    rows = []
    bf16 = make_bf16_mixed(model)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        acc = eval_acc(bf16, torch.device("cpu"))
    lat = bench_cpu_bf16(bf16)
    rows.append(("bf16 混合", size_of(bf16), lat, acc))

    int8_dyn = make_int8_dynamic(model)
    acc = eval_acc(int8_dyn, torch.device("cpu"))
    lat = bench_cpu(int8_dyn)
    rows.append(("int8 动态", size_of(int8_dyn), lat, acc))

    fp32_copy = MnistCNN() if False else build_model(tag)
    fp32_copy.load_state_dict(model.state_dict())
    int8_stat = make_int8_static(fp32_copy)
    acc = eval_acc(int8_stat, torch.device("cpu"))
    lat = bench_cpu(int8_stat)
    rows.append(("int8 静态", size_of(int8_stat), lat, acc))

    # torchao weight-only：tinygemm pack/前向算子仅 CUDA 有实现（int8/int4 都是），
    # 量化、延迟、评测全部在 GPU 上做；torchao 0.13 的 int4 打包 dtype bug
    # 用 bench_torchao.py 同款补丁解决
    import torchao.quantization.utils as _au
    _orig_pack = _au.pack_tinygemm_scales_and_zeros

    def _patched_pack(scales, zeros, dtype=torch.bfloat16):
        scales = scales.to(torch.bfloat16)
        zeros = zeros.to(torch.bfloat16)
        return _orig_pack(scales, zeros, torch.bfloat16)

    _au.pack_tinygemm_scales_and_zeros = _patched_pack
    for name, cfg in (("int8 weight-only", Int8WeightOnlyConfig()),
                      ("int4 weight-only", Int4WeightOnlyConfig())):
        m = copy.deepcopy(model).cuda()
        quantize_(m, cfg)
        lat = bench_gpu(m, device)
        acc = eval_acc(m, device)
        size = size_of(m.cpu())
        rows.append((name, size, lat, acc))
    _au.pack_tinygemm_scales_and_zeros = _orig_pack

    for name, size, lat, acc in rows:
        print(f"{name:<18s} {size:8.2f} {lat:8.3f} {lat_fp32/lat:6.2f}x {acc:8.4f}")


if __name__ == "__main__":
    main()
