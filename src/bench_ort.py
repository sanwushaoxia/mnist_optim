"""ONNX Runtime 三 provider 对比：CPU / CUDA / TensorRT EP。

用 fp32 ONNX 模型（model.onnx），分别用 CPUExecutionProvider、
CUDAExecutionProvider、TensorrtExecutionProvider 跑延迟/吞吐/准确率。
回答的问题：不装完整 TensorRT 时，ONNX Runtime 是不是好的替代部署引擎。
"""
import os
import time

# onnxruntime-gpu 的 CUDA/TensorRT EP 依赖 cudnn/cublas/TRT 动态库（torch pip 包自带），
# glibc 在进程启动时缓存 LD_LIBRARY_PATH，因此必须启动前设置：
#   LD_LIBRARY_PATH=<site>/nvidia/*/lib:<site>/tensorrt_libs python bench_ort.py
# 本脚本在 LD_LIBRARY_PATH 未设置时自动以子进程方式重启自身补上该变量。
_site = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__import__("torch").__file__))))
_needed = ["cudnn/lib", "cublas/lib", "cuda_runtime/lib", "cudnn"]
_pre = os.environ.get("LD_LIBRARY_PATH", "")
if not any("nvidia" in _pre for _ in [0]) or "cudnn" not in _pre:
    import glob, sys
    paths = glob.glob(os.path.join(_site, "nvidia", "*", "lib")) + \
            glob.glob(os.path.join(_site, "tensorrt_libs")) + \
            glob.glob(os.path.join(_site, "tensorrt_cu13_libs"))
    env = dict(os.environ, LD_LIBRARY_PATH=":".join(paths) + ":" + _pre)
    import subprocess
    sys.exit(subprocess.call([sys.executable] + sys.argv, env=env))

import numpy as np
import onnxruntime as ort
import torch
from torchvision import datasets, transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
ONNX_PATH = os.path.join(MODEL_DIR, "model.onnx")
WARMUP, ITERS, BATCH = 10, 50, 256

PROVIDERS = {
    "CPU": ["CPUExecutionProvider"],
    "CUDA": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    # 注意：ORT 的 TRT EP 编译对应 TRT 10（libnvinfer.so.10），与本机安装的
    # TRT 11.4（libnvinfer.so.11）二进制不兼容，会静默回退 CPU——原生 TRT
    # 对比见 bench_trt.py。此处保留以记录该兼容性结论。
    "TensorRT-EP": ["TensorrtExecutionProvider", "CPUExecutionProvider"],
}


def get_test_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_set = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform)
    images = np.stack([test_set[i][0].numpy() for i in range(len(test_set))])
    labels = np.array([test_set[i][1] for i in range(len(test_set))])
    return images, labels


def main():
    images, labels = get_test_data()
    print("=== ONNX Runtime 三 provider 对比（fp32 ONNX 模型）===\n")

    for name, provs in PROVIDERS.items():
        try:
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(ONNX_PATH, sess_options=sess_opts, providers=provs)
            actual = sess.get_providers()[0]
            inp = sess.get_inputs()[0].name

            # 吞吐（batch=256）
            dummy = images[:BATCH].astype(np.float32)
            for _ in range(WARMUP):
                sess.run(None, {inp: dummy})
            t0 = time.perf_counter()
            for _ in range(ITERS):
                sess.run(None, {inp: dummy})
            thr = ITERS * BATCH / (time.perf_counter() - t0)

            # 单条延迟 P50（batch=1）
            single = images[:1].astype(np.float32)
            for _ in range(WARMUP):
                sess.run(None, {inp: single})
            times = []
            for _ in range(100):
                t0 = time.perf_counter()
                sess.run(None, {inp: single})
                times.append((time.perf_counter() - t0) * 1000)
            times.sort()
            p50 = times[50]

            # 准确率（分批跑全测试集）
            preds = []
            for s in range(0, len(images), BATCH):
                logits = sess.run(None, {inp: images[s:s + BATCH].astype(np.float32)})[0]
                preds.append(logits.argmax(1))
            acc = (np.concatenate(preds) == labels).mean()

            print(f"{name:12s} 实际 provider: {actual}")
            print(f"             延迟P50 {p50:.3f} ms | 吞吐 {thr:,.0f} 条/秒 | 准确率 {acc:.4f}")
        except Exception as e:
            print(f"{name:12s} 失败: {str(e)[:150]}")


if __name__ == "__main__":
    main()
