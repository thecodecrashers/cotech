import os
import torch
from torch.utils.data import DataLoader
from models.registry import get_model
from utils.dataset import SegmentationDataset
from config import config
import torchvision.transforms.functional as TF
from PIL import Image

def evaluate(model, dataloader, device, save_vis=False, vis_dir=None):
    model.eval()
    total_iou, total_dice = 0, 0

    # ✅ 1️⃣ 可视化保存路径从 config 中读取
    vis_dir = vis_dir or config["val_vis_dir"]
    os.makedirs(vis_dir, exist_ok=True)

    with torch.no_grad():
        for i, (imgs, masks) in enumerate(dataloader):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = torch.sigmoid(model(imgs))
            preds_bin = (preds > 0.5).float()

            intersection = (preds_bin * masks).sum()
            union = preds_bin.sum() + masks.sum()
            iou = (intersection + 1e-6) / (union - intersection + 1e-6)
            dice = (2 * intersection + 1e-6) / (union + 1e-6)

            total_iou += iou.item()
            total_dice += dice.item()

            if save_vis:
                vis = TF.to_pil_image(preds_bin[0])
                vis.save(os.path.join(vis_dir, f"val_{i:03}.png"))

    avg_iou = total_iou / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    print(f"✅ Evaluation Results - mIoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}")

def main():
    # ✅ 2️⃣ 模型从 config 中读取参数和路径
    model = get_model(config["model_name"],
                      in_channels=config["in_channels"],
                      out_channels=config["out_channels"]).to(config["device"])
    model.load_state_dict(torch.load(config["save_path"], map_location=config["device"]))

    # ✅ 3️⃣ 验证集路径从 config 中读取
    if not os.path.exists(config["val_img_dir"]) or len(os.listdir(config["val_img_dir"])) == 0:
        print("⚠️ 没有发现验证集图片，跳过验证")
        return

    val_set = SegmentationDataset(config["val_img_dir"], config["val_mask_dir"],
                                   image_size=config["input_size"], augment=False)
    val_loader = DataLoader(val_set, batch_size=1)

    # ✅ 4️⃣ 启动评估时也使用 config 中的 vis_dir
    evaluate(model, val_loader, config["device"], save_vis=True)

if __name__ == "__main__":
    main()
