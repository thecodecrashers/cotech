import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.registry import get_model
from utils.dataset import SegmentationDataset
from config import config
from losses.combo_loss import build_loss_fn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from utils.metrics import pixel_accuracy, mean_iou


def align_prediction_size(preds, masks):
    if preds.ndim == 4 and masks.ndim == 3:
        if preds.shape[2:] != masks.shape[1:]:
            preds = nn.functional.interpolate(preds, size=masks.shape[1:], mode="bilinear", align_corners=False)
    else:
        raise ValueError(f"Unsupported shape: preds={preds.shape}, masks={masks.shape}")
    return preds


def freeze_layers(model):
    if config.get("freeze_encoder", False):
        print("🧊 冻结编码器层")
        if hasattr(model, "encoder"):
            for param in model.encoder.parameters():
                param.requires_grad = False
        else:
            print("⚠️ 未找到 'encoder' 属性，无法冻结")


def fine_tune():
    assert os.path.exists(config["fine_tune_img_dir"]), "❌ 未找到 fine_tune_img_dir"
    assert os.path.exists(config["fine_tune_mask_dir"]), "❌ 未找到 fine_tune_mask_dir"
    os.makedirs(os.path.dirname(config["fine_tune_save_path"]), exist_ok=True)

    # === 准备模型 ===
    model = get_model(config["model_name"], config["in_channels"], config["out_channels"]).to(config["device"])
    model.load_state_dict(torch.load(config["save_path"], map_location=config["device"]))
    freeze_layers(model)
    model.train()

    # === 微调数据集 ===
    fine_tune_set = SegmentationDataset(
        config["fine_tune_img_dir"],
        config["fine_tune_mask_dir"],
        config["input_size"],
        augment=True
    )
    loader = DataLoader(fine_tune_set, batch_size=config["fine_tune_batch_size"], shuffle=True)

    # === 优化器和损失函数 ===
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["fine_tune_lr"])
    criterion = build_loss_fn(config)

    # === 混合精度和累积梯度 ===
    use_amp = config.get("use_amp", False)
    accum_iter = config.get("accum_iter", 1)
    scaler = GradScaler(enabled=use_amp)

    # === 微调训练 ===
    for epoch in range(config["fine_tune_epochs"]):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        bar = tqdm(enumerate(loader), total=len(loader), desc=f"[微调] Epoch {epoch+1}/{config['fine_tune_epochs']}")

        for step, (imgs, masks) in bar:
            imgs, masks = imgs.to(config["device"]), masks.to(config["device"])

            with autocast(enabled=use_amp):
                preds = model(imgs)
                loss = criterion(preds, masks) / accum_iter

            scaler.scale(loss).backward()

            if (step + 1) % accum_iter == 0 or (step + 1) == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_iter
            bar.set_postfix(loss=loss.item() * accum_iter)

        print(f"📉 Epoch {epoch+1}: Fine-tune loss: {total_loss / len(loader):.4f}")

    # === 保存微调模型 ===
    torch.save(model.state_dict(), config["fine_tune_save_path"])
    print(f"✅ 微调完成，模型保存至: {config['fine_tune_save_path']}")


if __name__ == "__main__":
    fine_tune()
