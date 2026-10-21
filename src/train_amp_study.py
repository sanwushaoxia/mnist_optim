"""训练侧精度对比：fp32 vs bf16 autocast vs fp16+GradScaler（GPU）。

对比三种精度训练：每 epoch 耗时、最终测试精度。
- bf16 不需要 GradScaler（指数位与 fp32 相同，梯度不易下溢）
- fp16 必须配 GradScaler（梯度下溢防护）
补全"训练 → 量化 → 部署"全链路中的训练侧数据。
"""
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MnistCNN
from train import get_loaders

EPOCHS = 5
BATCH = 256
LR = 1e-3
MODES = ("fp32", "bf16", "fp16")


def train_one_mode(mode: str, device) -> tuple[float, float]:
    """返回 (总耗时秒, 最终精度)。"""
    torch.manual_seed(42)
    train_loader, test_loader = get_loaders()
    model = MnistCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(mode == "fp16"))

    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if mode == "fp32":
                loss = criterion(model(images), labels)
            else:
                dtype = torch.bfloat16 if mode == "bf16" else torch.float16
                with torch.autocast("cuda", dtype=dtype):
                    loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    train_time = time.perf_counter() - t0

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return train_time, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 训练侧精度对比（{EPOCHS} epochs, {torch.cuda.get_device_name(0)}）===\n")
    print(f"{'模式':<8s} {'总耗时':>8s} {'最终精度':>8s}")
    for mode in MODES:
        t, acc = train_one_mode(mode, device)
        print(f"{mode:<8s} {t:7.1f}s {acc:8.4f}")


if __name__ == "__main__":
    main()
