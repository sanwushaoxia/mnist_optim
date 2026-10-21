"""量化脚本：从 fp32 基线生成 bf16 混合精度与 int8 动态量化模型，并评测三者准确率。

- bf16 混合精度：权重 cast 成 bfloat16，但数值敏感层（首层 conv、最终分类层）保留 fp32。
- int8 混合精度：torch.ao.quantization.quantize_dynamic 动态量化，
  Linear 层权重 int8 + 激活动态量化，其余层保持 fp32。
"""
import copy
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MnistCNN
from train import get_loaders

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
FP32_PATH = os.path.join(MODEL_DIR, "model_fp32.pth")



def _sensitive_module(model: nn.Module, name: str, module: nn.Module) -> bool:
    """按模型拓扑判断敏感层（保留 fp32）：第一个 Conv2d、全部 BatchNorm、最后一个 Linear。

    不依赖具体层名，任意模型结构通用。
    """
    if isinstance(module, nn.BatchNorm2d):
        return True
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if convs and module is convs[0]:
        return True
    if linears and module is linears[-1]:
        return True
    return False


def make_bf16_mixed(model: nn.Module) -> nn.Module:
    """bfloat16 混合精度：敏感层（首个 conv/BN/末个 Linear）保留 fp32，其余 cast 成 bf16。

    推理需在 torch.autocast(bfloat16) 下进行，由 autocast 处理层间 dtype 转换。
    """
    mixed = copy.deepcopy(model).cpu()
    mixed.eval()
    for name, module in mixed.named_modules():
        if _sensitive_module(mixed, name, module):
            continue
        for param in module.parameters(recurse=False):
            param.data = param.data.bfloat16()
        for buf in module.buffers(recurse=False):
            if buf.is_floating_point():
                buf.data = buf.data.bfloat16()
    return mixed


def make_int8(model: nn.Module) -> nn.Module:
    """int8 混合精度：动态量化 Linear（权重 int8、激活动态量化），其余层保持 fp32。"""
    quantized = copy.deepcopy(model).cpu()
    quantized.eval()
    return torch.ao.quantization.quantize_dynamic(
        quantized, {nn.Linear}, dtype=torch.qint8
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_loaders()

    fp32 = MnistCNN().to("cpu")
    fp32.load_state_dict(torch.load(FP32_PATH, map_location="cpu", weights_only=True))
    fp32.eval()

    bf16_mixed = make_bf16_mixed(fp32)
    int8 = make_int8(fp32)

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(bf16_mixed.state_dict(), os.path.join(MODEL_DIR, "model_bf16_mixed.pth"))
    torch.save(int8.state_dict(), os.path.join(MODEL_DIR, "model_int8.pth"))

    print("=== 准确率评测（MNIST 测试集 10000 张）===")
    acc_fp32 = evaluate(fp32, test_loader, torch.device("cpu"))
    print(f"fp32          : {acc_fp32:.4f}")

    # bf16 混合精度：autocast 下推理
    with torch.autocast("cpu", dtype=torch.bfloat16):
        acc_bf16 = evaluate(bf16_mixed, test_loader, torch.device("cpu"))
    print(f"bf16 混合精度 : {acc_bf16:.4f}")

    # int8 动态量化仅支持 CPU 推理
    acc_int8 = evaluate(int8, test_loader, torch.device("cpu"))
    print(f"int8 混合精度 : {acc_int8:.4f}")


if __name__ == "__main__":
    main()
