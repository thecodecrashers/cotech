import os
import csv
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
        raise ValueError(f"Unsupported tensor shape: preds={preds.shape}, masks={masks.shape}")
    return preds


def save_checkpoint(model, optimizer, epoch, best_loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
    }, path)


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location=config["device"])
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]
    print(f"📦 恢复训练：从 Epoch {epoch+1} 开始")
    return epoch, best_loss


def append_loss_log(epoch, train_loss, val_loss=None, path=config["log_csv"]):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "train_loss", "val_loss"])
        writer.writerow([epoch+1, train_loss, val_loss if val_loss is not None else ""])


def test():
    print("🧪 正在进行测试集评估...")
    model = get_model(config["model_name"], config["in_channels"], config["out_channels"]).to(config["device"])
    model.load_state_dict(torch.load(config["save_path"], map_location=config["device"]))
    model.eval()

    test_set = SegmentationDataset(config["test_img_dir"], config["test_mask_dir"], config["input_size"], augment=False)
    test_loader = DataLoader(test_set, batch_size=1)

    acc_total, miou_total = 0, 0
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Testing", leave=False)
        for imgs, masks in test_bar:
            imgs, masks = imgs.to(config["device"]), masks.to(config["device"])
            preds = model(imgs)
            preds = align_prediction_size(preds, masks)

            acc_total += pixel_accuracy(preds, masks).item()
            miou_total += mean_iou(preds, masks, num_classes=config["out_channels"]).item()

    n = len(test_loader)
    print(f"📈 测试结果: Pixel Acc: {acc_total/n:.4f} | mIoU: {miou_total/n:.4f}")


def train():
    os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["log_csv"]), exist_ok=True)

    model = get_model(config["model_name"], config["in_channels"], config["out_channels"]).to(config["device"])
    train_set = SegmentationDataset(config["train_img_dir"], config["train_mask_dir"], config["input_size"], augment=False)
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)

    do_validation = os.path.exists(config["val_img_dir"]) and len(os.listdir(config["val_img_dir"])) > 0
    if do_validation:
        val_set = SegmentationDataset(config["val_img_dir"], config["val_mask_dir"], config["input_size"], augment=False)
        val_loader = DataLoader(val_set, batch_size=1)
    else:
        print("⚠️ 没有找到验证集，跳过验证阶段")

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = build_loss_fn(config)

    use_amp = config.get("use_amp", False)
    accum_iter = config.get("accum_iter", 1)
    scaler = GradScaler(enabled=use_amp)

    warmup_epochs = int(config["epochs"] * 0.2)
    warmup_factor = config.get("warmup_factor", 0.1)

    start_epoch = 0
    best_loss = float("inf")
    if os.path.exists(config["checkpoint_path"]):
        start_epoch, best_loss = load_checkpoint(model, optimizer, config["checkpoint_path"])
    else:
        print("⚙️ 未检测到断点文件，使用随机初始化参数")

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        if epoch < warmup_epochs:
            alpha = epoch / warmup_epochs
            warmup_lr = config["lr"] * (warmup_factor + (1 - warmup_factor) * alpha)
            optimizer.param_groups[0]["lr"] = warmup_lr
            print(f"🚀 Warmup Epoch {epoch+1}/{warmup_epochs} | lr={warmup_lr:.6f}")
        else:
            optimizer.param_groups[0]["lr"] = config["lr"]

        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config['epochs']} - Train", leave=False)
        for step, (imgs, masks) in train_bar:
            imgs, masks = imgs.to(config["device"]), masks.to(config["device"])

            with autocast(enabled=use_amp):
                preds = model(imgs)
                loss = criterion(preds, masks) / accum_iter

            scaler.scale(loss).backward()

            if (step + 1) % accum_iter == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accum_iter
            train_bar.set_postfix(loss=loss.item() * accum_iter)

        avg_train = train_loss / len(train_loader)

        # ===== 验证阶段 =====
        avg_val = None
        if do_validation:
            model.eval()
            val_loss = 0
            acc_total, miou_total = 0, 0

            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} - Val", leave=False)
            with torch.no_grad():
                for imgs, masks in val_bar:
                    imgs, masks = imgs.to(config["device"]), masks.to(config["device"])
                    with autocast(enabled=use_amp):
                        preds = model(imgs)
                        loss = criterion(preds, masks)

                    val_loss += loss.item()
                    val_bar.set_postfix(loss=loss.item())

                    preds = align_prediction_size(preds, masks)
                    acc_total += pixel_accuracy(preds, masks).item()
                    miou_total += mean_iou(preds, masks, num_classes=config["out_channels"]).item()

            avg_val = val_loss / len(val_loader)
            n = len(val_loader)
            print(f"[Epoch {epoch+1}] Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
            print(f"📊 Pixel Acc: {acc_total/n:.4f} | mIoU: {miou_total/n:.4f}")

            if avg_val < best_loss:
                torch.save(model.state_dict(), config["save_path"])
                best_loss = avg_val
                print(f"✅ Best model updated: {config['save_path']}")
        else:
            print(f"[Epoch {epoch+1}] Train Loss: {avg_train:.4f}")

        save_checkpoint(model, optimizer, epoch, best_loss, path=config["checkpoint_path"])
        append_loss_log(epoch, avg_train, avg_val, path=config["log_csv"])


if __name__ == "__main__":
    train()
    if os.path.exists(config["test_img_dir"]) and os.path.exists(config["test_mask_dir"]):
        test()
    else:
        print("⚠️ 未检测到测试集目录，跳过测试阶段")

