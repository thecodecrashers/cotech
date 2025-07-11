import os
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import cv2

from models.registry import get_model
from config import config  # ✅ 用你自己的配置文件

# ==== 路径配置（自动从 config 中读取）====
IMAGE_DIR = r"C:\Users\86178\Desktop\小可智能\焊点 20250630\测试用图"
VIS_DIR = os.path.join(IMAGE_DIR, "vis")
os.makedirs(VIS_DIR, exist_ok=True)

device = config["device"]

# ==== 模型加载 ====
model = get_model(
    config["model_name"],
    config["in_channels"],
    config["out_channels"]
).to(device)
model.load_state_dict(torch.load(config["save_path"], map_location=device))
model.eval()

# ==== padding + 记录信息 ====
def pad_and_record(image: Image.Image, target_size):
    orig_w, orig_h = image.size
    target_w, target_h = target_size
    pad_w, pad_h = max(0, target_w - orig_w), max(0, target_h - orig_h)
    left, right = pad_w // 2, pad_w - pad_w // 2
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    padded = TF.pad(image, [left, top, right, bottom], fill=0)
    return padded, (left, top, orig_w, orig_h)

def crop_back(tensor: torch.Tensor, pad_info):
    left, top, orig_w, orig_h = pad_info
    return tensor[..., top:top+orig_h, left:left+orig_w]

# ==== 预测掩码 ====
def predict_mask(image: Image.Image):
    padded, pad_info = pad_and_record(image, config["input_size"])
    input_tensor = TF.to_tensor(padded).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)[0]
        mask = torch.argmax(logits, dim=0, keepdim=True)
        cropped = crop_back(mask, pad_info)
        return cropped.squeeze(0).cpu().numpy().astype(np.uint8)

# ==== 多边形转换 + 限制每个轮廓最多 N 个点（等间距采样） ====
def mask_to_shapes(mask: np.ndarray, max_points=10):
    shapes = []
    for class_id in range(1, int(mask.max()) + 1):
        binary = (mask == class_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if len(cnt) >= 3:
                cnt = cnt.squeeze(1)
                if len(cnt) > max_points:
                    indices = np.linspace(0, len(cnt) - 1, max_points, dtype=int)
                    cnt = cnt[indices]
                points = [[float(x), float(y)] for x, y in cnt]
                shapes.append({
                    "label": "welding_point",
                    "points": points,
                    "group_id": None,
                    "description": "",
                    "shape_type": "polygon",
                    "flags": {},
                    "mask": None
                })
    return shapes

# ==== 主处理函数 ====
def process_image(image_path):
    base = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(IMAGE_DIR, base + ".json")

    image = Image.open(image_path).convert("L")
    mask = predict_mask(image)
    shapes = mask_to_shapes(mask, max_points=10)

    if not shapes:
        print(f"❌ 没有有效轮廓：{base}")
        return

    json_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": image.height,
        "imageWidth": image.width
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON 生成：{json_path}")

# ==== 批量处理入口 ====
if __name__ == "__main__":
    for fname in os.listdir(IMAGE_DIR):
        if fname.lower().endswith((".png", ".bmp", ".jpg")):
            process_image(os.path.join(IMAGE_DIR, fname))





r"""
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
import cv2

from models.registry import get_model
from config import config

# ==== 配置路径 ====
IMAGE_DIR = r"C:\Users\86178\Desktop\小可智能\焊点 20250630\测试用图"
VIS_DIR = os.path.join(IMAGE_DIR, "vis")
os.makedirs(VIS_DIR, exist_ok=True)

device = config["device"]

# ==== 加载模型 ====
model = get_model(config["model_name"], config["in_channels"], config["out_channels"]).to(device)
model.load_state_dict(torch.load(config["save_path"], map_location=device))
model.eval()

# ==== padding + 记录信息 ====
def pad_and_record(image: Image.Image, target_size):
    orig_w, orig_h = image.size
    target_w, target_h = target_size
    pad_w, pad_h = max(0, target_w - orig_w), max(0, target_h - orig_h)
    left, right = pad_w // 2, pad_w - pad_w // 2
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    padded = TF.pad(image, [left, top, right, bottom], fill=0)
    return padded, (left, top, orig_w, orig_h)

def crop_back(tensor: torch.Tensor, pad_info):
    left, top, orig_w, orig_h = pad_info
    return tensor[..., top:top+orig_h, left:left+orig_w]

# ==== 预测掩码 ====
def predict_mask(image: Image.Image):
    padded, pad_info = pad_and_record(image, config["input_size"])
    input_tensor = TF.to_tensor(padded).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)[0]
        mask = torch.argmax(logits, dim=0, keepdim=True)
        cropped = crop_back(mask, pad_info)
        return cropped.squeeze(0).cpu().numpy().astype(np.uint8)

# ==== 生成 LabelMe shapes ====
def mask_to_shapes(mask: np.ndarray):
    shapes = []
    for class_id in range(1, int(mask.max()) + 1):  # 跳过背景
        binary = (mask == class_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) >= 3:
                points = [[float(x), float(y)] for [[x, y]] in cnt]
                shapes.append({
                    "label": "welding_point",
                    "points": points,
                    "group_id": None,
                    "description": "",
                    "shape_type": "polygon",
                    "flags": {},
                    "mask": None
                })
    return shapes

# ==== 可视化 ====
def visualize(image: Image.Image, mask: np.ndarray, outname: str):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Input Image")
    axs[1].imshow(mask, cmap='jet')
    axs[1].set_title("Predicted Mask")
    axs[2].imshow(image, cmap='gray')
    axs[2].imshow(mask, cmap='jet', alpha=0.5)
    axs[2].set_title("Overlay")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"{outname}_vis.png"))
    plt.close()

# ==== 主函数 ====
def process_image(image_path):
    base = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(IMAGE_DIR, base + ".json")
    if os.path.exists(json_path):
        return
    image = Image.open(image_path).convert("L")
    mask = predict_mask(image)
    shapes = mask_to_shapes(mask)
    #visualize(image, mask, base)

    if not shapes:
        print(f"❌ 没有有效轮廓：{base}")
        return

    json_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": image.height,
        "imageWidth": image.width
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON 生成：{json_path}")

# ==== 批量处理 ====
if __name__ == "__main__":
    for fname in os.listdir(IMAGE_DIR):
        if fname.lower().endswith((".png", ".bmp", ".jpg")):
            #process_image(os.path.join(IMAGE_DIR, fname))
    """
