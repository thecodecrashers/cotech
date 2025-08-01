import os, json, numpy as np, cv2
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

# ==== 加载配置 ====
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

from models.registry import get_model

# ==== 模型加载 ====
device = config["device"]
model = get_model(config["pretrain_model_name"], config["in_channels"], config["out_channels"]).to(device)
model.load_state_dict(torch.load(os.path.join(config["pretrain_save_dir"], config["pretrain_save_filename"]), map_location=device))
model.eval()

# ==== 输入尺寸和步长 ====
input_h, input_w = config["input_size"]
stride_h, stride_w = input_h // 2, input_w // 2  # 🧠 用整数除法

def sliding_window_prediction(image: Image.Image):
    w, h = image.size
    # 🛠️ 确保 pad_w / pad_h 是整数
    pad_w = int((input_w - w % stride_w) % stride_w)
    pad_h = int((input_h - h % stride_h) % stride_h)
    image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)

    w_p, h_p = image.size
    mask_total = np.zeros((h_p, w_p), dtype=np.uint16)
    count_map = np.zeros((h_p, w_p), dtype=np.uint8)

    for top in range(0, h_p - input_h + 1, stride_h):
        for left in range(0, w_p - input_w + 1, stride_w):
            patch = image.crop((left, top, left + input_w, top + input_h))
            tensor = TF.to_tensor(patch).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(tensor)[0].argmax(0).cpu().numpy().astype(np.uint16)
            mask_total[top:top+input_h, left:left+input_w] += pred

            count_map[top:top+input_h, left:left+input_w] += 1

    count_map[count_map == 0] = 1  # 防止除以0
    final_mask = (mask_total / count_map).astype(np.uint8)
    return final_mask[:h, :w]  # 裁回原图尺寸

def mask_to_shapes(mask: np.ndarray, max_points=10):
    shapes = []
    for class_id in range(1, int(mask.max()) + 1):
        binary = (mask == class_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if len(cnt) >= 3:
                cnt = cnt.squeeze(1)
                if len(cnt) > max_points:
                    idx = np.linspace(0, len(cnt) - 1, max_points, dtype=int)
                    cnt = cnt[idx]
                points = [[float(x), float(y)] for x, y in cnt]
                shapes.append({
                    "label": str(class_id),
                    "points": points,
                    "group_id": None,
                    "description": "",
                    "shape_type": "polygon",
                    "flags": {}
                })
    return shapes

def annotate_one(image_path: str):
    base = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(os.path.dirname(image_path), base + ".json")
    image = Image.open(image_path).convert("L")
    mask = sliding_window_prediction(image)
    shapes = mask_to_shapes(mask, max_points=config["max_points"])
    if not shapes:
        print(f"⚠️ 无有效轮廓: {base}")
        return
    data = {
        #"version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": image.height,
        "imageWidth": image.width
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ 标注完成: {json_path}")

def main():
    files = [f for f in os.listdir(config["annotate_dir"]) if f.lower().endswith((".png", ".jpg", ".bmp"))]
    pbar = tqdm(files, desc="自动标注中")
    for f in pbar:
        annotate_one(os.path.join(config["annotate_dir"], f))

if __name__ == "__main__":
    main()









"""import os
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from models.registry import get_model
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)  # ✅ 自行准备 config 字典或模块

# ==== 路径配置 ====
IMAGE_DIR = config["annotate_img_dir"]
VIS_DIR = os.path.join(IMAGE_DIR, "vis")
CORRECT_DIR = os.path.join(IMAGE_DIR, "correct")
WRONG_DIR = os.path.join(IMAGE_DIR, "wrong")
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(CORRECT_DIR, exist_ok=True)
os.makedirs(WRONG_DIR, exist_ok=True)

device = config["device"]

# ==== 加载模型 ====
model = get_model(config["model_name"], config["in_channels"], config["out_channels"]).to(device)
model.load_state_dict(torch.load(os.path.join(config["save_dir"],config["save_filename"]), map_location=device))
model.eval()

# ==== padding + crop ====
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

# ==== mask 预测 ====
def predict_mask(image: Image.Image):
    padded, pad_info = pad_and_record(image, config["input_size"])
    input_tensor = TF.to_tensor(padded).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)[0]
        mask = torch.argmax(logits, dim=0, keepdim=True)
        cropped = crop_back(mask, pad_info)
        return cropped.squeeze(0).cpu().numpy().astype(np.uint8)

# ==== mask → polygon ====
import cv2
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

# ==== 可视化保存 ====
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
    vis_path = os.path.join(VIS_DIR, f"{outname}_vis.png")
    plt.savefig(vis_path, bbox_inches='tight')
    plt.close()
    return vis_path

# ==== 人工判断界面 ====
def show_for_review(vis_path):
    img = Image.open(vis_path)
    while True:
        plt.imshow(img)
        plt.axis("off")
        plt.title("Prediction Review\n[y] 正确 | [n] 错误 | [q] 退出")
        plt.show(block=False)
        key = input("👉 请输入标注结果 (y/n/q): ").strip().lower()
        plt.close()

        if key == 'y':
            return 'correct'
        elif key == 'n':
            return 'wrong'
        elif key == 'q':
            print("👋 已退出人工标注")
            exit()
        else:
            print("⚠️ 无效输入，请重新输入：y（正确）/n（错误）/q（退出）")

# ==== 单图处理 ====
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
    vis_path = visualize(image, mask, base)

    # 人工分类
    result = show_for_review(vis_path)
    dest_folder = CORRECT_DIR if result == 'correct' else WRONG_DIR

    for ext in [".png", ".jpg", ".bmp", ".json"]:
        src = os.path.join(IMAGE_DIR, base + ext)
        if os.path.exists(src):
            dst = os.path.join(dest_folder, base + ext)
            os.rename(src, dst)
            print(f"➡️ 移动至: {dst}")

# ==== 批量入口 ====
if __name__ == "__main__":
    for fname in os.listdir(IMAGE_DIR):
        if fname.lower().endswith((".png", ".bmp", ".jpg")):
            process_image(os.path.join(IMAGE_DIR, fname))
"""







"""import os
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import cv2

from models.registry import get_model
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)  # ✅ 用你自己的配置文件

# ==== 路径配置（自动从 config 中读取）====
IMAGE_DIR = config["annotate_img_dir"]
VIS_DIR = os.path.join(IMAGE_DIR, "vis")
os.makedirs(VIS_DIR, exist_ok=True)

device = config["device"]

# ==== 模型加载 ====
model = get_model(
    config["model_name"],
    config["in_channels"],
    config["out_channels"]
).to(device)
model.load_state_dict(torch.load(os.path.join(config["save_dir"],config["save_filename"]), map_location=device))
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
            process_image(os.path.join(IMAGE_DIR, fname))"""





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
