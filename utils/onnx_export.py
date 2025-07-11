# utils/export_onnx.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.registry import get_model
from config import config

# ===== 加载模型 =====
model = get_model(
    config["model_name"],
    in_channels=config["in_channels"],
    out_channels=config["out_channels"]
).to(config["device"])
model.load_state_dict(torch.load(config["save_path"], map_location=config["device"]))
model.eval()

# ===== 注意：PyTorch 模型输入是 [N, C, H, W]，而 config["input_size"] 是 (W, H) =====
width, height = config["input_size"]
dummy_input = torch.randn(1, config["in_channels"], height, width).to(config["device"])

# ===== 导出 ONNX =====
onnx_path = "model.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print(f"✅ ONNX模型导出完成：{onnx_path}")


