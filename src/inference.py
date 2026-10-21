"""端到端部署 Demo：加载任意一种模型形态做 MNIST 预测。

用法：
    python inference.py --model int8_static --images 5        # 随机 5 张测试图
    python inference.py --model fp32 --image-path /path/x.png # 自定义图片
    python inference.py --model trt_int8 --images 3           # TRT 引擎

支持的 --model：
    fp32 / bf16_mixed / int8_dynamic / int8_static / trt_fp32 / trt_fp16 / trt_bf16 / trt_int8
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
from torchvision import datasets, transforms

from model import MnistCNN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRT_ENGINES = {"trt_fp32", "trt_fp16", "trt_bf16", "trt_int8"}

CLASSES = [str(i) for i in range(10)]


def load_pytorch_model(name: str) -> torch.nn.Module:
    model = MnistCNN().cpu().eval()
    if name == "fp32":
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model_fp32.pth"), map_location="cpu", weights_only=True))
    elif name == "bf16_mixed":
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model_fp32.pth"), map_location="cpu", weights_only=True))
        from quantize import make_bf16_mixed
        model = make_bf16_mixed(model)
    elif name == "int8_dynamic":
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model_fp32.pth"), map_location="cpu", weights_only=True))
        from quantize import make_int8
        model = make_int8(model)
    elif name == "int8_static":
        model = torch.jit.load(os.path.join(MODEL_DIR, "model_int8_static.pth"), map_location="cpu")
        model.eval()
    else:
        raise ValueError(f"未知模型: {name}")
    return model


def load_trt_engine(name: str):
    import tensorrt as trt
    path = os.path.join(MODEL_DIR, f"model_{name}.engine")
    runtime = trt.Logger(trt.Logger.WARNING), trt.Runtime(trt.Logger(trt.Logger.WARNING))
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    assert engine is not None, f"引擎加载失败: {path}"
    return engine, trt


def predict_pytorch(model, x: torch.Tensor) -> int:
    with torch.no_grad():
        if getattr(model, "_is_bf16", False) or "bf16" in getattr(model, "_name", ""):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                logits = model(x)
        else:
            logits = model(x)
    return int(logits.argmax(1).item())


def predict_trt(engine_trt, x: torch.Tensor) -> int:
    engine, trt = engine_trt
    context = engine.create_execution_context()
    io_types = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16, trt.DataType.BF16: torch.bfloat16}
    inp_name = out_name = None
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            inp_name = n
        else:
            out_name = n
    inp_dtype = io_types[engine.get_tensor_dtype(inp_name)]
    out_dtype = io_types[engine.get_tensor_dtype(out_name)]
    context.set_input_shape(inp_name, x.shape)
    stream = torch.cuda.Stream()
    inp_t = x.to(inp_dtype).contiguous().cuda()
    out_t = torch.empty((x.shape[0], 10), dtype=out_dtype, device="cuda")
    context.set_tensor_address(inp_name, inp_t.data_ptr())
    context.set_tensor_address(out_name, out_t.data_ptr())
    context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()
    return int(out_t.argmax(1).cpu().item())


def get_test_image(idx: int) -> torch.Tensor:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    ts = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform)
    img, label = ts[idx]
    return img.unsqueeze(0), label


def load_custom_image(path: str) -> torch.Tensor:
    """加载 PNG/JPG 手写数字图（白底黑字或黑底白字均可，自动反色为 MNIST 规范）。"""
    from PIL import Image
    img = Image.open(path).convert("L").resize((28, 28))
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.mean() > 0.5:  # 白底黑字 → 反色
        arr = 1.0 - arr
    x = (arr - 0.1307) / 0.3081
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="MNIST 模型推理 Demo")
    parser.add_argument("--model", default="int8_static", choices=[
        "fp32", "bf16_mixed", "int8_dynamic", "int8_static", *TRT_ENGINES])
    parser.add_argument("--images", type=int, default=5, help="随机测试图数量")
    parser.add_argument("--image-path", type=str, default=None, help="自定义图片路径")
    args = parser.parse_args()

    if args.image_path:
        x = load_custom_image(args.image_path)
        label = None
    else:
        samples = [get_test_image(random.randint(0, 9999)) for _ in range(args.images)]

    if args.model in TRT_ENGINES:
        engine = load_trt_engine(args.model)
        predict = lambda t: predict_trt(engine, t)
    else:
        model = load_pytorch_model(args.model)
        model._name = args.model
        predict = lambda t: predict_pytorch(model, t)

    print(f"模型: {args.model}  设备: {'CUDA' if args.model in TRT_ENGINES or DEVICE.type == 'cuda' and args.model != 'int8_static' and args.model not in ('fp32', 'bf16_mixed', 'int8_dynamic') else 'CPU'}")
    if args.image_path:
        pred = predict(x)
        print(f"自定义图片 {args.image_path} → 预测: {pred}")
    else:
        correct = 0
        for x, label in samples:
            pred = predict(x)
            mark = "✓" if pred == label else "✗"
            correct += pred == label
            print(f"  真实={label}  预测={pred}  {mark}")
        print(f"命中 {correct}/{len(samples)}")


if __name__ == "__main__":
    main()
