import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
from models.registry import get_model
from config import config

# ====== 图像路径（可以放多个） ======
IMG_PATHS = [
    r"C:\Users\86178\Desktop\小可智能\裂纹\my_patches\1_3304_1131.png",
]

def predict_single_image(img_path):
    # ===== 模型加载 =====
    model = get_model(
        config["model_name"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"]
    ).to(config["device"])
    
    model.load_state_dict(torch.load(config["save_path"], map_location=config["device"]))
    model.eval()

    # ===== 图像预处理 =====
    img = Image.open(img_path).convert("L")
    img_resized = img.resize(config["input_size"])
    img_tensor = to_tensor(img_resized).unsqueeze(0).to(config["device"])

    # ===== 推理预测 =====
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))[0, 0]
        binary_mask = (pred > 0.5).float().cpu().numpy()

    # ===== 可视化显示 =====
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(os.path.basename(img_path))

    axs[0].imshow(img_resized, cmap='gray')
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    axs[1].imshow(binary_mask, cmap='jet')  # 彩色掩码
    axs[1].set_title("Predicted Mask (jet)")
    axs[1].axis("off")

    axs[2].imshow(img_resized, cmap='gray')
    axs[2].imshow(binary_mask, cmap='jet', alpha=0.5)  # 叠加图像
    axs[2].set_title("Overlay")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    for path in IMG_PATHS:
        if os.path.exists(path):
            predict_single_image(path)
        else:
            print(f"❌ 文件不存在：{path}")

