import os
import shutil
import json
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from models.registry import get_model

# ======= 路径与参数 =======
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

IMG_DIR = config["human_filter_dir"]         # 需要预测的图片文件夹
BAD_DIR = config["hum_filter_bad_picture_dir"]              # 差的图片复制到这个目录
os.makedirs(BAD_DIR, exist_ok=True)

device = config["device"]
model = get_model(config["model_name"], config["in_channels"], config["out_channels"]).to(device)
model.load_state_dict(torch.load(os.path.join(config["save_dir"], config["save_filename"]), map_location=device))
model.eval()

input_size = config.get("input_size", (512, 512))

# ======= 推理函数 =======
def preprocess(img_path):
    img = Image.open(img_path).convert("L")
    orig = img.copy()
    img = img.resize(input_size, Image.BILINEAR)
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    return img_tensor, orig

def predict(img_tensor):
    with torch.no_grad():
        out = model(img_tensor)
        pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()
    return pred

def mask_to_pil(mask):
    if mask.max() <= 1:  # 0/1二分类
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = (mask * (255 // mask.max())).astype(np.uint8)
    return Image.fromarray(mask)

def overlay_mask(img, mask, alpha=0.5):
    mask_img = mask_to_pil(mask).convert("RGBA")
    color_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)
    color_mask[..., 0] = 255  # Red channel, 可改成你想要的色彩
    color_mask[..., 3] = (mask > 0) * int(255 * alpha)
    mask_img = Image.fromarray(color_mask)
    overlay = Image.alpha_composite(img.convert("RGBA"), mask_img)
    return overlay.convert("RGB")

# ======= 遍历预测并人工筛选 =======
img_list = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))]

for img_name in img_list:
    img_path = os.path.join(IMG_DIR, img_name)
    img_tensor, orig_img = preprocess(img_path)
    pred_mask = predict(img_tensor)
    mask_img = mask_to_pil(pred_mask).resize(orig_img.size, Image.NEAREST)
    overlay_img = overlay_mask(orig_img, np.array(mask_img))

    # 显示三联图
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    axs[0].imshow(orig_img)
    axs[0].set_title("原图")
    axs[1].imshow(mask_img, cmap='gray')
    axs[1].set_title("预测掩码")
    axs[2].imshow(overlay_img)
    axs[2].set_title("叠加效果")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # 人工判别
    res = input(f"【{img_name}】效果如何？(y=通过/n=差，复制到BAD_DIR)：").strip().lower()
    if res == "n":
        shutil.copy(img_path, os.path.join(BAD_DIR, img_name))
        print(f"→ {img_name} 已复制到 {BAD_DIR}")
    else:
        print(f"→ {img_name} 跳过")

print("全部图片已处理完毕！")
