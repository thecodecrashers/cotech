import os
import csv
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.registry import get_model
from utils.dataset import SegmentationDataset
from config import config
from losses.combo_loss import build_loss_fn
from tqdm import tqdm  # 进度条

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

def train():
    os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["log_csv"]), exist_ok=True)

    # ===== 模型加载 =====
    model = get_model(
        config["model_name"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"]
    ).to(config["device"])

    # ===== 数据加载 =====
    train_set = SegmentationDataset(
        config["train_img_dir"], config["train_mask_dir"],
        image_size=config["input_size"], augment=True
    )
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)

    do_validation = os.path.exists(config["val_img_dir"]) and len(os.listdir(config["val_img_dir"])) > 0
    if do_validation:
        val_set = SegmentationDataset(
            config["val_img_dir"], config["val_mask_dir"],
            image_size=config["input_size"], augment=False
        )
        val_loader = DataLoader(val_set, batch_size=1)
    else:
        print("⚠️ 没有找到验证集，跳过验证阶段")

    # ===== 优化器 & 损失函数 =====
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = build_loss_fn(config)

    # ===== 判断是否恢复训练 =====
    start_epoch = 0
    best_loss = float("inf")
    if os.path.exists(config["checkpoint_path"]):
        start_epoch, best_loss = load_checkpoint(model, optimizer, config["checkpoint_path"])
    else:
        print("⚙️ 未检测到断点文件，使用随机初始化参数")

    # ===== 正式训练 =====
    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} - Train", leave=False)
        for imgs, masks in train_bar:
            imgs, masks = imgs.to(config["device"]), masks.to(config["device"])
            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train = train_loss / len(train_loader)

        # ===== 验证阶段 =====
        avg_val = None
        if do_validation:
            model.eval()
            val_loss = 0
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} - Val", leave=False)
            with torch.no_grad():
                for imgs, masks in val_bar:
                    imgs, masks = imgs.to(config["device"]), masks.to(config["device"])
                    preds = model(imgs)
                    loss = criterion(preds, masks)
                    val_loss += loss.item()
                    val_bar.set_postfix(loss=loss.item())
            avg_val = val_loss / len(val_loader)
            print(f"[Epoch {epoch+1}/{config['epochs']}] Train Loss: {avg_train:.4f}  Val Loss: {avg_val:.4f}")

            if avg_val < best_loss:
                torch.save(model.state_dict(), config["save_path"])
                best_loss = avg_val
                print(f"✅ Best model updated: {config['save_path']}")
        else:
            print(f"[Epoch {epoch+1}/{config['epochs']}] Train Loss: {avg_train:.4f}")

        # ===== 保存 checkpoint 和记录 loss =====
        save_checkpoint(model, optimizer, epoch, best_loss, path=config["checkpoint_path"])
        append_loss_log(epoch, avg_train, avg_val,path=config["log_csv"])

if __name__ == "__main__":
    train()
