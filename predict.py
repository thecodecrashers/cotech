import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
from models.registry import get_model
from config import config

# ====== 图像路径列表（支持多个） ======
IMG_PATHS = [
    r"C:\Users\86178\Desktop\小可智能\焊点 20250630\测试用图\_焊点图_光源1_0b152950-bfc9-4ad2-8a10-6b324f0bcff2.png",
]

# 是否保存预测结果图像
SAVE_RESULT = False
SAVE_DIR = "pred_vis"
os.makedirs(SAVE_DIR, exist_ok=True)

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
    img = Image.open(img_path).convert("L")  # 灰度图
    img_resized = img.resize(config["input_size"])
    img_tensor = to_tensor(img_resized).unsqueeze(0).to(config["device"])  # shape: [1, 1, H, W]

    # ===== 推理（多分类） =====
    with torch.no_grad():
        logits = model(img_tensor)[0]  # shape: [C, H, W]
        probs = torch.softmax(logits, dim=0)  # softmax over channels
        pred_mask = torch.argmax(probs, dim=0).cpu().numpy()  # shape: [H, W]，值为类别索引（0,1,...）

    # ===== 可视化显示 =====
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(os.path.basename(img_path))

    axs[0].imshow(img_resized, cmap='gray')
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    axs[1].imshow(pred_mask, cmap='jet')
    axs[1].set_title("Predicted Mask (Class Index)")
    axs[1].axis("off")

    axs[2].imshow(img_resized, cmap='gray')
    axs[2].imshow(pred_mask, cmap='jet', alpha=0.5)
    axs[2].set_title("Overlay")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()

    # ===== 可选保存结果 =====
    if SAVE_RESULT:
        save_name = os.path.splitext(os.path.basename(img_path))[0] + "_pred.png"
        save_path = os.path.join(SAVE_DIR, save_name)
        overlay = Image.fromarray((pred_mask * 127).astype('uint8'))  # 简单上色保存
        overlay.save(save_path)
        print(f"✅ 预测图已保存：{save_path}")

if __name__ == "__main__":
    for path in IMG_PATHS:
        if os.path.exists(path):
            predict_single_image(path)
        else:
            print(f"❌ 文件不存在：{path}")

