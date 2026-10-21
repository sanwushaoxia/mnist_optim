"""FastAPI 推理服务 Demo：把量化模型包装成 HTTP 服务。

端点：
- GET /health          — 服务状态与可用模型列表
- POST /predict        — 上传图片（multipart form）+ 模型名，返回预测结果
- POST /predict_batch  — 多图批量预测

启动：uvicorn serve:app --host 0.0.0.0 --port 8000（在 src/ 目录下）
冒烟测试：python serve.py --selftest（用 FastAPI TestClient，不真正起服务）
"""
import argparse
import io
import os

import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import datasets, transforms

from model import MnistCNN
from quantize import make_bf16_mixed, make_int8 as make_int8_dynamic

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

app = FastAPI(title="MNIST 量化模型推理服务", description="fp32/bf16/int8 模型部署 Demo")

# 启动时加载全部 CPU 模型（TRT 引擎体积大、需 GPU，服务 Demo 不含）
_MODELS: dict[str, torch.nn.Module] = {}


def load_models():
    if _MODELS:
        return
    fp32 = MnistCNN().cpu().eval()
    fp32.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "model_fp32.pth"), map_location="cpu", weights_only=True)
    )
    _MODELS["fp32"] = fp32
    _MODELS["bf16_mixed"] = make_bf16_mixed(fp32)
    _MODELS["int8_dynamic"] = make_int8_dynamic(fp32)
    import contextlib
    with contextlib.suppress(Exception):
        _MODELS["int8_static"] = torch.jit.load(
            os.path.join(MODEL_DIR, "model_int8_static.pth"), map_location="cpu"
        )


TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


def preprocess(data: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(data)).convert("L").resize((28, 28))
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.mean() > 0.5:  # 白底黑字 → 反色为 MNIST 规范（黑底白字）
        arr = 1.0 - arr
    x = (arr - 0.1307) / 0.3081
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def predict_one(model: torch.nn.Module, name: str, x: torch.Tensor) -> tuple[int, float]:
    if name == "bf16_mixed":
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(x)
    else:
        logits = model(x)
    probs = torch.softmax(logits.float(), dim=1)
    conf, idx = probs.max(dim=1)
    return int(idx.item()), float(conf.item())


@app.on_event("startup")
def startup():
    load_models()


@app.get("/health")
def health():
    return {"status": "ok", "models": sorted(_MODELS.keys()), "device": "cpu"}


@app.post("/predict")
async def predict(image: UploadFile = File(...), model: str = Form("int8_static")):
    if model not in _MODELS:
        return JSONResponse(status_code=400, content={"error": f"未知模型 {model}", "available": sorted(_MODELS.keys())})
    x = preprocess(await image.read())
    pred, conf = predict_one(_MODELS[model], model, x)
    return {"model": model, "prediction": pred, "confidence": round(conf, 4)}


@app.post("/predict_batch")
async def predict_batch(images: list[UploadFile] = File(...), model: str = Form("int8_static")):
    if model not in _MODELS:
        return JSONResponse(status_code=400, content={"error": f"未知模型 {model}", "available": sorted(_MODELS.keys())})
    xs = torch.cat([preprocess(await f.read()) for f in images])
    m = _MODELS[model]
    results = []
    with torch.no_grad():
        if model == "bf16_mixed":
            with torch.autocast("cpu", dtype=torch.bfloat16):
                logits = m(xs)
        else:
            logits = m(xs)
        probs = torch.softmax(logits.float(), dim=1)
        for conf, idx in zip(*probs.max(dim=1)):
            results.append({"prediction": int(idx.item()), "confidence": round(float(conf.item()), 4)})
    return {"model": model, "count": len(results), "results": results}


def selftest():
    """TestClient 冒烟测试：随机测试图 + 全部模型。"""
    from fastapi.testclient import TestClient

    load_models()
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200, r.text
    print("health:", r.json())

    ts = datasets.MNIST(DATA_DIR, train=False, download=False)
    labels = []
    import PIL.Image as PILImage
    for i in (0, 1, 2):
        img, lbl = ts[i]
        labels.append(lbl)
        buf = io.BytesIO()
        PILImage.fromarray(255 - np.array(img)).save(buf, format="PNG")  # 白底黑字模拟真实扫描件
        buf.seek(0)
        for name in sorted(_MODELS.keys()):
            r = client.post("/predict", files={"image": ("x.png", buf.getvalue())}, data={"model": name})
            assert r.status_code == 200, r.text
            j = r.json()
            mark = "✓" if j["prediction"] == lbl else "✗"
            print(f"  样本{i}(label={lbl}) {name:12s} → {j['prediction']} conf={j['confidence']} {mark}")
    print("冒烟测试通过")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true", help="TestClient 冒烟测试")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
