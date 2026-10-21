"""Fashion-MNIST 鲁棒性验证：在更难的同尺寸任务上检验量化精度损失。

流程与 MNIST 完全一致：fp32 训练 → bf16 混合精度 / int8 动态 / int8 静态量化 → 对比。
目的：验证"MNIST 量化无损"的结论是任务简单导致的还是普遍规律。
Fashion-MNIST 为 10 类衣物图（28×28 灰度），fp32 基线通常 ~92%（比 MNIST 难得多）。
"""
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MnistCNN
from quantize import make_bf16_mixed, make_int8 as make_int8_dynamic
from quantize_static import make_int8_static

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
FMNIST_DIR = os.path.join(BASE_DIR, "data_fmnist")
EPOCHS = 8  # Fashion-MNIST 更难，多训几个 epoch
BATCH_SIZE = 256
LR = 1e-3


def get_fmnist_loaders():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
    )
    train_set = datasets.FashionMNIST(FMNIST_DIR, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(FMNIST_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=2)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device, use_bf16=False) -> float:
    model.eval()
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_fmnist_loaders()

    # 1) fp32 训练
    print(f"=== Fashion-MNIST fp32 训练（{EPOCHS} epochs, {device}）===")
    model = MnistCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{EPOCHS}  loss={running/len(train_loader.dataset):.4f}  acc={acc:.4f}")

    fp32 = model.cpu().eval()
    torch.save(fp32.state_dict(), os.path.join(MODEL_DIR, "model_fmnist_fp32.pth"))

    # 2) 三种量化
    print("\n=== Fashion-MNIST 量化对比（测试集 10000 张）===")
    acc_fp32 = evaluate(fp32, test_loader, torch.device("cpu"))
    print(f"fp32          : {acc_fp32:.4f}")

    bf16 = make_bf16_mixed(fp32)
    acc_bf16 = evaluate(bf16, test_loader, torch.device("cpu"), use_bf16=True)
    print(f"bf16 混合精度 : {acc_bf16:.4f}")

    int8_dyn = make_int8_dynamic(fp32)
    acc_dyn = evaluate(int8_dyn, test_loader, torch.device("cpu"))
    print(f"int8 动态     : {acc_dyn:.4f}")

    fp32_copy = MnistCNN().eval()
    fp32_copy.load_state_dict(fp32.state_dict())
    int8_stat = make_int8_static(fp32_copy)
    acc_stat = evaluate(int8_stat, test_loader, torch.device("cpu"))
    print(f"int8 静态     : {acc_stat:.4f}")

    print(f"\n量化损失：bf16 {acc_fp32-acc_bf16:+.4f}, int8动态 {acc_fp32-acc_dyn:+.4f}, int8静态 {acc_fp32-acc_stat:+.4f}")


if __name__ == "__main__":
    main()
