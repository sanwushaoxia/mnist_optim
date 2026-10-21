"""torchao weight-only 量化 GPU 对比：int8 / int4 weight-only vs fp32 基线。

weight-only 混合精度：权重存 int8/int4，激活保持 bf16（tinygemm kernel 原生 CUDA）。
- 体积：整模型 torch.save（subclass tensor 需整模型序列化）
- 速度：batch=1 延迟 + batch=256 吞吐（GPU）
- 准确率：MNIST 测试集
"""
import copy
import io
import os
import time

import torch
from torch.utils.data import DataLoader
from torchao.quantization import Int8WeightOnlyConfig, Int4WeightOnlyConfig, quantize_
from torchvision import datasets, transforms

from model import MnistCNN

# torchao 0.13 + torch 2.6 组合的 dtype 链路 bug：int4 的 zero_point 硬编码 bf16，
# 但打包时 guard 期望与 scale dtype 一致，导致非 (64,32) 形状的 Linear 量化报错。
# 补丁：强制 scales/zeros dtype 统一。
import torchao.quantization.utils as _au
_orig_pack = _au.pack_tinygemm_scales_and_zeros


def _patched_pack(scales, zeros, dtype=torch.bfloat16):
    # 统一为 bf16（tinygemm kernel 原生期望的 packed dtype；fp32 packed 会被 kernel 误读）
    scales = scales.to(torch.bfloat16)
    zeros = zeros.to(torch.bfloat16)
    return _orig_pack(scales, zeros, torch.bfloat16)


_au.pack_tinygemm_scales_and_zeros = _patched_pack

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
WARMUP, ITERS, BATCH = 10, 50, 256


def load_fp32() -> MnistCNN:
    model = MnistCNN().eval()
    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "model_fp32.pth"), map_location="cpu", weights_only=True)
    )
    return model


def save_whole(model, name: str) -> float:
    buf = io.BytesIO()
    torch.save(model, buf)
    path = os.path.join(MODEL_DIR, name)
    with open(path, "wb") as f:
        f.write(buf.getvalue())
    return buf.tell() / 1024 / 1024


@torch.no_grad()
def eval_acc(model, device) -> float:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_set = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform)
    loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=2)
    model = model.to(device).eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def bench(model, device, batch: int) -> tuple[float, float]:
    model = model.to(device).eval()
    dummy = torch.randn(batch, 1, 28, 28, device=device)
    for _ in range(WARMUP):
        model(dummy)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        model(dummy)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / ITERS / batch * 1000, ITERS * batch / elapsed


def main():
    device = torch.device("cuda")
    fp32 = load_fp32()

    # int4 的 _convert_weight_to_int4pack 仅 CUDA 有实现，量化需在 GPU 上进行
    m8 = copy.deepcopy(fp32).cuda()
    quantize_(m8, Int8WeightOnlyConfig())
    m4 = copy.deepcopy(fp32).cuda()
    quantize_(m4, Int4WeightOnlyConfig())

    size8 = save_whole(m8.cpu(), "model_torchao_int8wo.pth")
    size4 = save_whole(m4.cpu(), "model_torchao_int4wo.pth")

    print("=== torchao weight-only 量化 GPU 对比 ===\n")
    print(f"{'模型':16s} {'大小MB':>8s} {'延迟ms':>8s} {'吞吐/秒':>12s} {'准确率':>8s}")

    lat, thr = bench(fp32, device, 1)
    _, thr_b = bench(fp32, device, BATCH)
    acc = eval_acc(fp32, device)
    print(f"{'fp32':16s} {12.64:8.2f} {lat:8.3f} {thr_b:12,.0f} {acc:8.4f}")

    for name, m, size in (("int8 weight-only", m8, size8), ("int4 weight-only", m4, size4)):
        lat, thr = bench(m, device, 1)
        _, thr_b = bench(m, device, BATCH)
        acc = eval_acc(m, device)
        print(f"{name:16s} {size:8.2f} {lat:8.3f} {thr_b:12,.0f} {acc:8.4f}")


if __name__ == "__main__":
    main()
