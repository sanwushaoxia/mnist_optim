"""校准改进曲线实验：在 Fashion-MNIST（PTQ 掉 5.2% 的任务）上扫描
校准数据量 × 校准方法，回答"静态量化的校准到底怎么做好"。

扫描维度：
- 数据量：256 / 1024 / 4096 / 16384 张
- 方法（激活 observer）：
  * MinMaxObserver（现状，对异常值敏感）
  * MovingAverageMinMaxObserver（滑动平均，更稳）
  * HistogramObserver（分布直方图，最精细）

注：激活 per-channel observer 与 FX 算子融合不兼容（scales 通道数冲突），
不纳入扫描。
权重统一 per-channel MinMax（weight 不受校准数据量影响）。

注：torch 2.6 无 PercentileObserver，percentile 校准由 HistogramObserver 的
分布统计近似覆盖。
"""
import copy
import os

import torch
import torch.nn as nn
from torch.ao.quantization import (MinMaxObserver, MovingAverageMinMaxObserver,
                                   HistogramObserver, QConfig)
from torch.ao.quantization.observer import PerChannelMinMaxObserver
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MnistCNN
from verify_fmnist import evaluate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data_fmnist")
FMNIST_WEIGHTS = os.path.join(MODEL_DIR, "model_fmnist_fp32.pth")

CALIB_SIZES = (256, 1024, 4096, 16384)
BATCH = 256


def load_fmnist_fp32() -> MnistCNN:
    model = MnistCNN().cpu().eval()
    model.load_state_dict(torch.load(FMNIST_WEIGHTS, map_location="cpu", weights_only=True))
    return model


def get_fmnist_train_transform():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
    )


def calibrate_and_convert(model_fp32, qconfig: QConfig, n_samples: int, dataset: str = "fmnist") -> nn.Module:
    """用指定 qconfig 和数据量做静态量化（每次用独立副本）。

    dataset="mnist" 时故意用 MNIST 校准（归一化参数不同），模拟
    "校准数据分布与真实输入不一致"的部署事故，作对照。
    """
    m = copy.deepcopy(model_fp32).eval()
    example = (torch.randn(1, 1, 28, 28),)
    prepared = prepare_fx(m, {"": qconfig}, example)

    if dataset == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        ds = datasets.MNIST(os.path.join(BASE_DIR, "data"), train=True, download=False, transform=transform)
    else:
        ds = datasets.FashionMNIST(DATA_DIR, train=True, download=False,
                                   transform=get_fmnist_train_transform())
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=2)
    with torch.no_grad():
        seen = 0
        for images, _ in loader:
            if seen >= n_samples:
                break
            prepared(images)
            seen += images.size(0)
    return convert_fx(prepared)


def build_qconfigs() -> dict[str, QConfig]:
    # 权重 per-channel 用 PerChannelMinMaxObserver（MinMaxObserver 仅支持 per-tensor）
    wc = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    return {
        "MinMax（现状）": QConfig(activation=MinMaxObserver.with_args(dtype=torch.quint8), weight=wc),
        "MovingAvgMinMax": QConfig(activation=MovingAverageMinMaxObserver.with_args(dtype=torch.quint8), weight=wc),
        "Histogram": QConfig(activation=HistogramObserver.with_args(dtype=torch.quint8), weight=wc),
    }


def main():
    device = torch.device("cpu")
    model = load_fmnist_fp32()

    # fp32 基线与测试集
    transform = get_fmnist_train_transform()
    ts = datasets.FashionMNIST(DATA_DIR, train=False, download=False, transform=transform)
    test_loader = DataLoader(ts, batch_size=512, shuffle=False, num_workers=2)
    acc_fp32 = evaluate(model, test_loader, device)
    print(f"fp32 基线: {acc_fp32:.4f}\n")

    print(f"{'校准方法':<16s}" + "".join(f" {n:>6d}张" for n in CALIB_SIZES))
    results = {}
    for name, qc in build_qconfigs().items():
        row = []
        for n in CALIB_SIZES:
            q = calibrate_and_convert(model, qc, n)
            acc = evaluate(q, test_loader, device)
            row.append(acc)
        results[name] = row
        print(f"{name:<16s}" + "".join(f" {a:7.4f}" for a in row))

    # 对照：跨数据集校准（用 MNIST 校准 Fashion-MNIST 模型）——复现部署事故
    wc = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    qc = QConfig(activation=MinMaxObserver.with_args(dtype=torch.quint8), weight=wc)
    for n in (1024, 16384):
        q = calibrate_and_convert(model, qc, n, dataset="mnist")
        acc = evaluate(q, test_loader, device)
        print(f"{'⚠️ 跨数据集校准(MNIST)':<16s} {n}张: {acc:.4f}")

    print(f"\nfp32 基线: {acc_fp32:.4f}（量化损失 = 基线 - 量化后）")


if __name__ == "__main__":
    main()
