"""QAT（量化感知训练）vs PTQ 对比：在 Fashion-MNIST（PTQ 掉 5.2% 的任务）上。

流程：加载 fp32 权重 → FX prepare_qat_fx（插入 fake-quant 节点）→
微调 3 epoch（lr 1e-4，让权重在量化约束下重新适应）→ convert → 评测。
对照：PTQ（87.45%）与 fp32 基线（92.68%）。
"""
import copy
import os
import time

import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MnistCNN
from verify_fmnist import evaluate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data_fmnist")
FMNIST_WEIGHTS = os.path.join(MODEL_DIR, "model_fmnist_fp32.pth")

QAT_EPOCHS = 3
QAT_LR = 1e-4
BATCH = 256


def load_fmnist_fp32() -> MnistCNN:
    model = MnistCNN().cpu()
    model.load_state_dict(torch.load(FMNIST_WEIGHTS, map_location="cpu", weights_only=True))
    return model


def get_loaders():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
    )
    train_set = datasets.FashionMNIST(DATA_DIR, train=True, download=False, transform=transform)
    test_set = datasets.FashionMNIST(DATA_DIR, train=False, download=False, transform=transform)
    return (
        DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(test_set, batch_size=512, shuffle=False, num_workers=2),
    )


def run_qat(model_fp32, train_loader, device) -> nn.Module:
    """QAT：fake-quant 插入 → 微调 → 转换为真量化模型。"""
    model_fp32.train()
    qconfig_mapping = get_default_qat_qconfig_mapping("x86")
    example = (torch.randn(1, 1, 28, 28),)
    prepared = prepare_qat_fx(model_fp32, qconfig_mapping, example).to(device)

    optimizer = torch.optim.Adam(prepared.parameters(), lr=QAT_LR)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, QAT_EPOCHS + 1):
        prepared.train()
        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(prepared(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
        print(f"QAT epoch {epoch}/{QAT_EPOCHS}  loss={running / len(train_loader.dataset):.4f}")

    prepared.eval().cpu()
    return convert_fx(copy.deepcopy(prepared))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_loaders()
    model = load_fmnist_fp32()

    acc_fp32 = evaluate(model, test_loader, torch.device("cpu"))
    print(f"fp32 基线: {acc_fp32:.4f}")

    t0 = time.perf_counter()
    qat_model = run_qat(model, train_loader, device)
    dt = time.perf_counter() - t0
    acc_qat = evaluate(qat_model, test_loader, torch.device("cpu"))
    print(f"QAT（微调 {QAT_EPOCHS} epoch, {dt:.0f}s）: {acc_qat:.4f}")
    print(f"\n三方对比：fp32 {acc_fp32:.4f} | QAT {acc_qat:.4f} | PTQ 0.8745")
    print(f"QAT 相比 PTQ 恢复: {acc_qat - 0.8745:+.4f}")


if __name__ == "__main__":
    main()
