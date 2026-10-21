"""TensorRT 部署：构建 fp32 / fp16 / int8 三种引擎（适配 TensorRT 11）。

TRT 11 移除了弱类型 API（BuilderFlag.INT8、IInt8Calibrator），int8 走官方推荐的
显式量化路径：在 ONNX 中插入 QDQ（QuantizeLinear/DequantizeLinear）节点 +
STRONGLY_TYPED 强类型网络构建。激活 scale 由校准集（训练集子集）统计（max-abs），
权重 per-channel。
fp16/fp32：ONNX 导出时直接 cast 权重 dtype，强类型网络按图执行。
"""
import copy
import os

import numpy as np
import onnx
from onnx import numpy_helper
import tensorrt as trt
import torch
from torchvision import datasets, transforms

from model import MnistCNN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
ONNX_PATH = os.path.join(MODEL_DIR, "model.onnx")
PRECISIONS = ("fp32", "fp16", "bf16", "int8")
CALIB_BATCHES = 32  # 32 * 256 = 8192 张校准图
BATCH = 256


def load_fp32() -> MnistCNN:
    model = MnistCNN().cpu().eval()
    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "model_fp32.pth"), map_location="cpu", weights_only=True)
    )
    return model


def _cast_model(model: MnistCNN, dtype: torch.dtype) -> MnistCNN:
    m = copy.deepcopy(model)
    for param in m.parameters():
        param.data = param.data.to(dtype)
    for buf in m.buffers():
        if buf.is_floating_point():
            buf.data = buf.data.to(dtype)
    return m


def export_onnx(model: MnistCNN, dtype: torch.dtype, out_path: str):
    m = model if dtype == torch.float32 else _cast_model(model, dtype)
    dummy = torch.randn(1, 1, 28, 28, dtype=dtype)
    torch.onnx.export(
        m, dummy, out_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX 已导出: {out_path}")


def collect_calibration_scales(model: MnistCNN) -> dict[str, float]:
    """训练集子集前向，hook 统计每个 Conv/Linear 数据输入的 per-tensor max-abs。

    返回 {层名: scale}，symmetric int8: scale = max_abs / 127。
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST(DATA_DIR, train=True, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH, shuffle=False, num_workers=2)

    ranges: dict[str, float] = {}

    def make_hook(name):
        def fn(module, inp, out):
            x = inp[0].detach()
            ranges[name] = max(ranges.get(name, 0.0), x.abs().max().item())
        return fn

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= CALIB_BATCHES:
                break
            model(images)
    for h in hooks:
        h.remove()

    return {name: r / 127.0 for name, r in ranges.items()}


def insert_qdq(onnx_path: str, scales: dict[str, float]):
    """在每个 Conv/Gemm 的数据输入和权重输入前插入 QDQ 对（TRT 显式量化）。

    torch.onnx.export 已把 Conv+BN 融合；ReLU 与 Q 可交换，TRT 会自动传播 QDQ。
    层名 -> 激活 scale 按模型层名匹配；权重 scale 为 per-channel max-abs。
    """
    model = onnx.load(onnx_path)
    g = model.graph

    # ONNX 节点名可能为空，先统一命名
    for i, node in enumerate(g.node):
        if not node.name:
            node.name = f"{node.op_type}_{i}"

    # 模型层名 -> ONNX 节点名匹配：按权重 initializer 名（如 features.0.weight）映射
    node_act_scale = {}
    for node in g.node:
        if node.op_type not in ("Conv", "Gemm"):
            continue
        weight_name = node.input[1]  # 形如 "features.0.weight"
        layer_name = weight_name.rsplit(".", 1)[0]
        if layer_name in scales:
            node_act_scale[node.name] = scales[layer_name]
        else:
            # 未匹配到校准统计（理论不应发生），用保守默认
            node_act_scale[node.name] = 0.01

    # 两遍法：第一遍为每个 Conv/Gemm 生成 QDQ 计划（不动原图），
    # 第二遍按拓扑顺序重建节点序列，把 QDQ 对插在消费节点之前
    plans = {}  # node.name -> (act_qdq, w_qdq, act_scale)
    for node in g.node:
        if node.op_type not in ("Conv", "Gemm"):
            continue

        # 激活 QDQ（per-tensor）
        act_scale = np.float32(max(node_act_scale[node.name], 1e-8))
        s_act, z_act = f"{node.name}/act_scale", f"{node.name}/act_zp"
        q_act, dq_act = f"{node.name}/act_q", f"{node.name}/act_dq"
        act_qdq = (
            numpy_helper.from_array(np.array([act_scale]), s_act),
            numpy_helper.from_array(np.zeros(1, dtype=np.int8), z_act),
            onnx.helper.make_node("QuantizeLinear", [node.input[0], s_act, z_act], [q_act]),
            onnx.helper.make_node("DequantizeLinear", [q_act, s_act, z_act], [dq_act]),
        )

        # 权重 QDQ（per-channel，沿输出通道轴）
        weight_name = node.input[1]
        weight_arr = numpy_helper.to_array(_find_initializer(g, weight_name))
        w_scales = np.abs(weight_arr.astype(np.float32)).reshape(weight_arr.shape[0], -1).max(axis=1) / 127.0
        w_scales = np.clip(w_scales, 1e-8, None).astype(np.float32)
        s_w, z_w = f"{node.name}/w_scale", f"{node.name}/w_zp"
        q_w, dq_w = f"{node.name}/w_q", f"{node.name}/w_dq"
        w_qdq = (
            numpy_helper.from_array(w_scales, s_w),
            numpy_helper.from_array(np.zeros_like(w_scales, dtype=np.int8), z_w),
            onnx.helper.make_node("QuantizeLinear", [weight_name, s_w, z_w], [q_w], axis=0),
            onnx.helper.make_node("DequantizeLinear", [q_w, s_w, z_w], [dq_w], axis=0),
        )

        plans[node.name] = (act_qdq, w_qdq, dq_act, dq_w)

    # 第二遍：重建节点序列
    new_inits: list = []
    rebuilt: list = []
    for node in g.node:
        if node.name in plans:
            act_qdq, w_qdq, dq_act, dq_w = plans[node.name]
            s_act_init, z_act_init, q_node, dq_node = act_qdq
            s_w_init, z_w_init, wq_node, wdq_node = w_qdq
            new_inits.extend([s_act_init, z_act_init, s_w_init, z_w_init])
            rebuilt.extend([q_node, dq_node, wq_node, wdq_node])
            node.input[0] = dq_act
            node.input[1] = dq_w
        rebuilt.append(node)

    del g.node[:]
    g.node.extend(rebuilt)
    g.initializer.extend(new_inits)

    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)
    print(f"QDQ 已插入: {onnx_path}")


def _find_initializer(g, name):
    for init in g.initializer:
        if init.name == name:
            return init
    raise KeyError(f"initializer not found: {name}")


def build_engine(onnx_path: str, precision: str):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"ONNX 解析失败 ({precision})")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 1, 28, 28), (BATCH, 1, 28, 28), (BATCH, 1, 28, 28))
    config.add_optimization_profile(profile)

    print(f"构建 {precision} 引擎中...")
    serialized = builder.build_serialized_network(network, config)
    assert serialized is not None, f"{precision} 引擎构建失败"
    out = os.path.join(MODEL_DIR, f"model_trt_{precision}.engine")
    with open(out, "wb") as f:
        f.write(serialized)
    print(f"{precision} 引擎已保存: {out} ({os.path.getsize(out)/1024/1024:.2f} MB)")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    model = load_fp32()

    export_onnx(model, torch.float32, ONNX_PATH)
    build_engine(ONNX_PATH, "fp32")

    export_onnx(model, torch.float16, ONNX_PATH)
    build_engine(ONNX_PATH, "fp16")

    # bf16：需 opset 20 导出（ONNX BF16 支持更全），TRT 强类型网络可保留 BF16
    BF16_PATH = os.path.join(MODEL_DIR, "model_bf16_o20.onnx")
    export_onnx(model, torch.bfloat16, BF16_PATH)
    build_engine(BF16_PATH, "bf16")

    export_onnx(model, torch.float32, ONNX_PATH)
    scales = collect_calibration_scales(model)
    print(f"校准统计完成，共 {len(scales)} 层激活范围")
    insert_qdq(ONNX_PATH, scales)
    build_engine(ONNX_PATH, "int8")


if __name__ == "__main__":
    main()
