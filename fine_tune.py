import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
from models.registry import get_model
from utils.dataset import SegmentationDataset
from config import config
from losses.combo_loss import build_loss_fn
from tqdm import tqdm

# ==== 可配置参数 ====
fine_tune_img_dir = "fix_data/images"
fine_tune_mask_dir = "fix_data/masks"
fine_tune_save_path = "checkpoints/fine_tuned.pth"
fine_tune_epochs = 5
fine_tune_lr = 1e-6
freeze_encoder = True
use_mixed_training = True  # ✅ 是否与原训练集混合训练
# ====================

def freeze_backbone_params(model):
    model_name = config["model_name"].lower()
    if "unet" in model_name:
        for name, param in model.named_parameters():
            if "down" in name or "encoder" in name:
                param.requires_grad = False
    elif "segformer" in model_name:
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
    else:
        print("⚠️ 未识别模型结构，未进行参数冻结")
    print("🧊 已冻结 Encoder/Backbone 部分参数")

def train_finetune():
    device = config["device"]

    # === 加载模型 ===
    model = get_model(config["model_name"],
                      in_channels=config["in_channels"],
                      out_channels=config["out_channels"]).to(device)
    model.load_state_dict(torch.load(config["save_path"], map_location=device))
    print(f"📦 已加载模型参数: {config['save_path']}")

    if freeze_encoder:
        freeze_backbone_params(model)

    # === 构建微调数据集 ===
    fix_dataset = SegmentationDataset(
        fine_tune_img_dir, fine_tune_mask_dir,
        image_size=config["input_size"], augment=True
    )

    if use_mixed_training:
        print("🔀 使用混合训练模式（原训练集 + 错误样本）")
        train_dataset = SegmentationDataset(
            config["train_img_dir"], config["train_mask_dir"],
            image_size=config["input_size"], augment=True
        )
        total_dataset = ConcatDataset([train_dataset, fix_dataset])
    else:
        print("🧪 仅使用错误样本进行微调")
        total_dataset = fix_dataset

    loader = DataLoader(total_dataset, batch_size=config["batch_size"], shuffle=True)

    # === 损失 & 优化器 ===
    criterion = build_loss_fn(config)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=fine_tune_lr)

    # === 开始微调训练 ===
    for epoch in range(fine_tune_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Fine-tune Epoch {epoch+1}/{fine_tune_epochs}")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        print(f"✅ Epoch {epoch+1} 微调平均 Loss: {avg_loss:.4f}")

    # === 保存微调后的模型 ===
    torch.save(model.state_dict(), fine_tune_save_path)
    print(f"✅ 微调模型保存至: {fine_tune_save_path}")

if __name__ == "__main__":
    train_finetune()
