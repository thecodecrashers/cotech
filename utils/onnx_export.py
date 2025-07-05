# utils/export_onnx.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.registry import get_model
from config import config

# 创建模型并加载权重
model = get_model(
    config["model_name"],
    in_channels=config["in_channels"],
    out_channels=config["out_channels"]
).to(config["device"])
model.load_state_dict(torch.load(config["save_path"], map_location=config["device"]))
model.eval()

# 构造一个示例输入
dummy_input = torch.randn(1, config["in_channels"], *config["input_size"]).to(config["device"])

# 导出 ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",                        # 导出路径
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print("✅ ONNX模型导出完成：model.onnx")

