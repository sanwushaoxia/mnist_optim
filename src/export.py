"""统一导出入口：从 fp32 基线一键导出全部模型形态到 models/。

用法：
    python src/export.py                # 导出全部（fp32/bf16/int8动态/int8静态/ONNX/TRT引擎）
    python src/export.py --skip-trt     # 跳过 TRT 引擎（构建慢）
    python src/export.py --list         # 只列出已有产物与大小
"""
import argparse
import os
import subprocess
import sys

import torch

from model import MnistCNN, build_model
from quantize import make_bf16_mixed, make_int8 as make_int8_dynamic
from quantize_static import make_int8_static

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
FP32_PATH = os.path.join(MODEL_DIR, "model_fp32.pth")


def load_fp32() -> MnistCNN:
    model = MnistCNN().cpu().eval()
    model.load_state_dict(torch.load(FP32_PATH, map_location="cpu", weights_only=True))
    return model


def export_pytorch(model: MnistCNN):
    import io

    artifacts = []

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model_fp32.pth"))
    artifacts.append(("model_fp32.pth", "state_dict"))

    bf16 = make_bf16_mixed(model)
    torch.save(bf16.state_dict(), os.path.join(MODEL_DIR, "model_bf16_mixed.pth"))
    artifacts.append(("model_bf16_mixed.pth", "state_dict"))

    int8_dyn = make_int8_dynamic(model)
    torch.save(int8_dyn.state_dict(), os.path.join(MODEL_DIR, "model_int8.pth"))
    artifacts.append(("model_int8.pth", "state_dict"))

    fp32_copy = MnistCNN().eval()
    fp32_copy.load_state_dict(model.state_dict())
    int8_stat = make_int8_static(fp32_copy)
    torch.jit.save(torch.jit.script(int8_stat), os.path.join(MODEL_DIR, "model_int8_static.pth"))
    artifacts.append(("model_int8_static.pth", "torchscript"))

    return artifacts


def export_onnx(model: MnistCNN):
    out = os.path.join(MODEL_DIR, "model.onnx")
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model, dummy, out,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


def export_trt():
    # TRT 引擎构建逻辑较长（QDQ/校准/四精度），直接复用 build_trt.py
    subprocess.run([sys.executable, os.path.join(BASE_DIR, "src", "build_trt.py")], check=True)


def list_artifacts():
    print(f"{'文件':<34s} {'大小':>10s}")
    for f in sorted(os.listdir(MODEL_DIR)):
        path = os.path.join(MODEL_DIR, f)
        size = os.path.getsize(path) / 1024 / 1024
        print(f"{f:<34s} {size:8.2f}MB")


def main():
    parser = argparse.ArgumentParser(description="统一导出入口")
    parser.add_argument("--skip-trt", action="store_true", help="跳过 TRT 引擎构建")
    parser.add_argument("--list", action="store_true", help="只列出已有产物")
    args = parser.parse_args()

    if args.list:
        list_artifacts()
        return

    assert os.path.exists(FP32_PATH), f"请先训练基线: python src/train.py（缺 {FP32_PATH}）"
    model = load_fp32()

    print("导出 PyTorch 形态（fp32/bf16/int8 动态/int8 静态）...")
    for fname, kind in export_pytorch(model):
        print(f"  {fname} ({kind})")

    print("导出 ONNX...")
    export_onnx(model)
    print("  model.onnx")

    if not args.skip_trt:
        print("构建 TRT 引擎（fp32/fp16/bf16/int8，含 QDQ 校准）...")
        export_trt()

    print("\n=== 全部产物 ===")
    list_artifacts()


if __name__ == "__main__":
    main()
