# predict.py

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from models.registry import get_model
from config import config
import os

def predict_single_image(img_path, save_path="pred_vis.png"):
    model = get_model(config["model_name"],
                      in_channels=config["in_channels"],
                      out_channels=config["out_channels"]).to(config["device"])
    model.load_state_dict(torch.load(config["save_path"], map_location=config["device"]))
    model.eval()

    img = Image.open(img_path).convert("L")
    img = img.resize(config["input_size"])
    img_tensor = to_tensor(img).unsqueeze(0).to(config["device"])

    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))[0, 0]
        binary_mask = (pred > 0.5).float()

    # 保存可视化图像
    pred_img = to_pil_image(binary_mask)
    pred_img.save(save_path)
    print(f"✅ Prediction saved to: {save_path}")

if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test.png"
    save_path = sys.argv[2] if len(sys.argv) > 2 else "pred_vis.png"
    predict_single_image(img_path, save_path)
