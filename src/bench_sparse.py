"""2:4 稀疏引擎基准：sparse_fp32（无稀疏 kernel）vs sparse_fp32_sw（SPARSE_WEIGHTS）。

对比同权重下启用/不启用稀疏 Tensor Core 的速度与体积差异。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench_trt import load_engine, bench, evaluate, get_test_images

PRECISIONS = ("sparse_fp32", "sparse_fp32_sw")


def main():
    images, labels = get_test_images()
    print("=== 2:4 稀疏 TRT 引擎对比（同剪枝权重）===\n")
    print(f"{'引擎':16s} {'大小MB':>8s} {'延迟ms':>8s} {'吞吐/秒':>12s} {'准确率':>8s}")
    for p in PRECISIONS:
        path = os.path.join("models", f"model_trt_{p}.engine")
        engine = load_engine(path)
        size = os.path.getsize(path) / 1024 / 1024
        lat, _ = bench(engine, 1)
        _, thr = bench(engine, 256)
        acc = evaluate(engine, images, labels)
        print(f"{p:16s} {size:8.2f} {lat:8.3f} {thr:12,.0f} {acc:8.4f}")


if __name__ == "__main__":
    main()
