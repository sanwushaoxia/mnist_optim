"""2:4 结构化稀疏 TRT 实验：剪枝权重 → 导出 ONNX → SPARSE_WEIGHTS 构建引擎。

2:4 稀疏：每 4 个连续权重沿输入通道方向保留绝对值最大的 2 个，其余置 0
（NVIDIA 稀疏 Tensor Core 的硬件格式，Ampere 起支持，理论矩阵乘吞吐翻倍）。
强类型模式下 SPARSE_WEIGHTS 是结构性标志（非精度标志），配合剪枝后的权重生效。

对照组：同样剪枝的 fp32 引擎（无 SPARSE_WEIGHTS），隔离稀疏 kernel 的收益。
剪枝后需微调恢复精度，但为隔离纯稀疏加速效应，这里对比剪枝后立即量化的
两引擎（同一剪枝权重、是否启用稀疏 kernel），并同时报告剪枝本身的精度损失。
"""
import copy
import os

import numpy as np
import onnx
from onnx import numpy_helper
import tensorrt as trt
import torch

from model import MnistCNN
from build_trt import load_fp32, export_onnx, build_engine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
ONNX_SPARSE_PATH = os.path.join(MODEL_DIR, "model_sparse.onnx")


def prune_24(model: MnistCNN) -> MnistCNN:
    """对每个 Conv/Linear 权重沿最后维做 2:4 剪枝（每 4 个保留 abs 最大的 2 个）。"""
    m = copy.deepcopy(model)
    with torch.no_grad():
        for name, module in m.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                w = module.weight.data
                # 沿输出通道展平后按 4 个一组（Conv 权重 (out, in, kh, kw)，
                # 展平 (out, in*kh*kw) 后沿 in 维分组才是硬件 2:4 格式）
                out_c = w.shape[0]
                w2 = w.reshape(out_c, -1)
                k = w2.shape[1]
                if k % 4 != 0:
                    # 首层 conv 输入维 9（1 通道 3×3）不是 4 的倍数，跳过（保持稠密）
                    continue
                w2 = w2.reshape(out_c, k // 4, 4)
                threshold = w2.abs().topk(2, dim=-1).values.min(dim=-1, keepdim=True).values
                mask = w2.abs() >= threshold
                w2 = w2 * mask
                module.weight.data = w2.reshape(w.shape)
    return m


def check_24_sparsity(model: MnistCNN) -> float:
    """统计剪枝后实际稀疏度（%零）。"""
    total_zeros = total = 0
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            w = module.weight.data
            total_zeros += (w == 0).sum().item()
            total += w.numel()
    return total_zeros / total * 100


def add_sparse_flag(onnx_path: str, engine_tag: str, sparse: bool):
    """构建引擎；sparse=True 时加 SPARSE_WEIGHTS 标志（需在 build_engine 外单独建）。"""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        assert parser.parse(f.read()), f"ONNX 解析失败 ({engine_tag})"

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if sparse:
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 1, 28, 28), (256, 1, 28, 28), (256, 1, 28, 28))
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None
    out = os.path.join(MODEL_DIR, f"model_trt_{engine_tag}.engine")
    with open(out, "wb") as f:
        f.write(serialized)
    print(f"{engine_tag} 引擎已保存 ({os.path.getsize(out)/1024/1024:.2f} MB, sparse={sparse})")


def main():
    fp32 = load_fp32()
    pruned = prune_24(fp32)
    sp = check_24_sparsity(pruned)
    print(f"2:4 剪枝完成，实际稀疏度 {sp:.1f}%")

    # 剪枝本身（无微调）的精度损失
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    ts = datasets.MNIST(os.path.join(BASE_DIR, "data"), train=False, download=False, transform=transform)
    loader = DataLoader(ts, batch_size=512, num_workers=2)
    with torch.no_grad():
        correct = total = 0
        for imgs, lbls in loader:
            correct += (pruned(imgs).argmax(1) == lbls).sum().item()
            total += lbls.size(0)
    print(f"剪枝后精度（无微调）: {correct/total:.4f}（fp32 基线 0.9904）")

    # 导出剪枝模型 ONNX 并构建对照引擎
    export_onnx(pruned, torch.float32, ONNX_SPARSE_PATH)
    add_sparse_flag(ONNX_SPARSE_PATH, "sparse_fp32", sparse=False)
    add_sparse_flag(ONNX_SPARSE_PATH, "sparse_fp32_sw", sparse=True)


if __name__ == "__main__":
    main()
