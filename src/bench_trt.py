"""TensorRT 引擎基准测试：fp32 / fp16 / int8 三种引擎的推理速度与准确率。

- 速度：单条延迟（batch=1, ms/条）和吞吐（batch=256, 条/秒），
  预热 10 轮 + 50 轮计时，CUDA 同步。
- 准确率：MNIST 测试集 10000 张，验证量化是否无损。
- 注意：强类型引擎的 IO dtype 因精度而异（fp16 引擎为 HALF，其余 FLOAT），
  输入输出按引擎声明 dtype 动态适配。
"""
import os
import time

import numpy as np
import tensorrt as trt
import torch
from torchvision import datasets, transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
PRECISIONS = ("fp32", "fp16", "bf16", "int8")
WARMUP, ITERS, BATCH = 10, 50, 256
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.BF16: torch.bfloat16,
    trt.DataType.INT8: torch.int8,
}
TRT_TO_TORCH[trt.DataType.INT32] = torch.int32
TRT_TO_TORCH[trt.DataType.INT64] = torch.int64


def load_engine(path: str):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    assert engine is not None, f"引擎反序列化失败: {path}"
    return engine


def get_io(engine) -> tuple[str, str, torch.dtype, torch.dtype]:
    """返回 (输入名, 输出名, 输入 dtype, 输出 dtype)。"""
    inp_name = out_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inp_name = name
        else:
            out_name = name
    return (
        inp_name, out_name,
        TRT_TO_TORCH[engine.get_tensor_dtype(inp_name)],
        TRT_TO_TORCH[engine.get_tensor_dtype(out_name)],
    )


def get_test_images():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_set = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform)
    images = torch.stack([test_set[i][0] for i in range(len(test_set))])
    labels = torch.tensor([test_set[i][1] for i in range(len(test_set))])
    return images, labels


def run_batch(context, inp_name, out_name, inp_dtype, out_dtype,
              batch_t: torch.Tensor, out_shape: tuple, stream) -> torch.Tensor:
    inp_t = batch_t.to(inp_dtype).contiguous().cuda()
    out_t = torch.empty((batch_t.shape[0],) + out_shape, dtype=out_dtype, device="cuda")
    context.set_tensor_address(inp_name, inp_t.data_ptr())
    context.set_tensor_address(out_name, out_t.data_ptr())
    context.execute_async_v3(stream.cuda_stream)
    return out_t


@torch.no_grad()
def evaluate(engine, images: torch.Tensor, labels: torch.Tensor) -> float:
    context = engine.create_execution_context()
    inp_name, out_name, inp_dtype, out_dtype = get_io(engine)
    out_shape = tuple(engine.get_tensor_shape(out_name))[1:]
    stream = torch.cuda.Stream()

    correct = 0
    for start in range(0, len(images), BATCH):
        batch = images[start:start + BATCH]
        context.set_input_shape(inp_name, batch.shape)
        logits = run_batch(context, inp_name, out_name, inp_dtype, out_dtype, batch, out_shape, stream)
        torch.cuda.synchronize()
        correct += (logits.argmax(1).cpu() == labels[start:start + BATCH]).sum().item()
    return correct / len(images)


def bench(engine, batch: int) -> tuple[float, float]:
    context = engine.create_execution_context()
    inp_name, out_name, inp_dtype, out_dtype = get_io(engine)
    out_shape = tuple(engine.get_tensor_shape(out_name))[1:]
    context.set_input_shape(inp_name, (batch, 1, 28, 28))
    stream = torch.cuda.Stream()

    dummy = torch.randn(batch, 1, 28, 28)
    for _ in range(WARMUP):
        run_batch(context, inp_name, out_name, inp_dtype, out_dtype, dummy, out_shape, stream)
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(ITERS):
        run_batch(context, inp_name, out_name, inp_dtype, out_dtype, dummy, out_shape, stream)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    per_sample_ms = elapsed / ITERS / batch * 1000
    throughput = ITERS * batch / elapsed
    return per_sample_ms, throughput


def main():
    images, labels = get_test_images()
    print("=== TensorRT 引擎对比 ===\n")

    print(f"{'引擎':6s} {'大小':>10s} {'延迟 batch=1':>14s} {'吞吐 batch=256':>16s} {'准确率':>8s}")
    for p in PRECISIONS:
        path = os.path.join(MODEL_DIR, f"model_trt_{p}.engine")
        engine = load_engine(path)
        size = os.path.getsize(path) / 1024 / 1024
        lat, _ = bench(engine, 1)
        _, thr = bench(engine, BATCH)
        acc = evaluate(engine, images, labels)
        print(f"{p:6s} {size:8.2f}MB {lat:11.3f}ms {thr:13,.0f}/s {acc:8.4f}")


if __name__ == "__main__":
    main()
