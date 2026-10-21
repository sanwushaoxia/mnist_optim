"""int8 静态量化（新增模型，不替换动态量化版）。

流程：fusing → 插入 QConfig 观察器 → 校准（1000 张训练图）→ convert。
模型形态：Conv + Linear 全部 int8（权重 per-channel + 激活 per-tensor），
输出 models/model_int8_static.pth（TorchScript 序列化，静态量化模型需连同
量化配置一起保存才能正确加载）。
"""
import os

import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MnistCNN
from train import get_loaders, evaluate

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_PATH = os.path.join(MODEL_DIR, "model_int8_static.pth")
CALIB_BATCHES = 4  # 4 * 256 = 1024 张校准图


def make_int8_static(model_fp32: nn.Module) -> nn.Module:
    """FX 图模式静态量化：Conv+Linear 全 int8，x86 后端 qconfig。"""
    model_fp32.eval()
    qconfig_mapping = get_default_qconfig_mapping("x86")
    example_input = (torch.randn(1, 1, 28, 28),)

    prepared = prepare_fx(model_fp32, qconfig_mapping, example_input)
    # 校准：喂代表性数据，观察器统计激活范围
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST(
        os.path.join(BASE_DIR, "data"), train=True, download=False, transform=transform
    )
    calib_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=2)
    with torch.no_grad():
        for i, (images, _) in enumerate(calib_loader):
            if i >= CALIB_BATCHES:
                break
            prepared(images)

    quantized = convert_fx(prepared)
    return quantized


def main():
    fp32 = MnistCNN().cpu()
    fp32.load_state_dict(
        torch.load(
            os.path.join(MODEL_DIR, "model_fp32.pth"), map_location="cpu", weights_only=True
        )
    )
    fp32.eval()

    print("静态量化中（FX prepare → 校准 1024 张 → convert）...")
    static = make_int8_static(fp32)

    # 保存：静态量化模型用 TorchScript 序列化才能完整保留量化参数
    scripted = torch.jit.script(static)
    torch.jit.save(scripted, STATIC_PATH)
    print(f"静态量化模型已保存: {STATIC_PATH}")

    # 评测
    _, test_loader = get_loaders()
    acc = evaluate(static, test_loader, torch.device("cpu"))
    print(f"int8 静态量化准确率: {acc:.4f}")


if __name__ == "__main__":
    main()
