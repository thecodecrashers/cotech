
import os
import json
import random
from PIL import Image, ImageDraw, ImageEnhance, ImageOps, ImageFilter
import numpy as np

# ---------- 读取配置 ----------
with open("config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

DATA_DIR         = cfg["annotate_img_dir"]
OUTPUT_DIR       = os.path.join(DATA_DIR, "pretrain_split_data")
AUGMENT_TIMES    = cfg["pretrain_augment_times"]   # 名字有点长但保持一致
INCLUDE_ORIGINAL = True

JSON_EXT           = ".json"
SUPPORTED_IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 1, 0, 0   # 一键预训练：全部进 train

# ---------- 工具函数 ----------
def make_dirs(base):
    for sp in ["train", "val", "test"]:
        os.makedirs(os.path.join(base, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(base, sp, "masks"),  exist_ok=True)

def load_labelme_json(jpath):
    with open(jpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    w, h = data["imageWidth"], data["imageHeight"]
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    label_map, next_id = {"_background_": 0}, 1
    for s in data["shapes"]:
        label = s.get("label", "_background_")
        if label not in label_map:
            label_map[label], next_id = next_id, next_id + 1
        draw.polygon(s["points"], fill=label_map[label])
    return mask

def add_gaussian_noise(img, mean=0, std=10):
    arr = np.array(img, np.float32)
    noise = np.random.normal(mean, std, arr.shape).astype(np.float32)
    out   = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def rand_aug(img, msk):
    if random.random() < .5:
        img, msk = ImageOps.mirror(img), ImageOps.mirror(msk)
    if random.random() < .5:
        img, msk = ImageOps.flip(img),   ImageOps.flip(msk)
    if random.random() < .3:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(.8, 1.2))
    if random.random() < .3:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(.8, 1.2))
    if random.random() < .3:
        img = ImageEnhance.Color(img).enhance(random.uniform(.8, 1.2))
    if random.random() < .3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(.5, 1.5)))
    if random.random() < .3:
        img = add_gaussian_noise(img, std=random.uniform(5, 20))
    return img, msk

def save_pair(img, msk, base, idx, split, ext):
    img.save(os.path.join(OUTPUT_DIR, split, "images", f"{base}_aug{idx}{ext}"))
    msk.save(os.path.join(OUTPUT_DIR, split, "masks",  f"{base}_aug{idx}{ext}"))

def process_one(img_file, split):
    base, ext = os.path.splitext(img_file)
    img_path  = os.path.join(DATA_DIR, img_file)
    json_path = os.path.join(DATA_DIR, base + JSON_EXT)
    if not os.path.exists(json_path):
        return
    img  = Image.open(img_path).convert("RGB")
    mask = load_labelme_json(json_path)

    idx = 0
    if INCLUDE_ORIGINAL:
        save_pair(img, mask, base, idx, split, ext); idx += 1
    for _ in range(AUGMENT_TIMES):
        new_img, new_msk = rand_aug(img.copy(), mask.copy())
        save_pair(new_img, new_msk, base, idx, split, ext); idx += 1

def main_augment():
    imgs = [f for f in os.listdir(DATA_DIR)
            if os.path.splitext(f)[1].lower() in SUPPORTED_IMG_EXTS
            and os.path.exists(os.path.join(DATA_DIR, os.path.splitext(f)[0] + JSON_EXT))]
    random.shuffle(imgs)
    total = len(imgs)
    n_train = int(total * TRAIN_RATIO)
    n_val   = int(total * VAL_RATIO)

    splits = {"train": imgs[:n_train],
              "val":   imgs[n_train:n_train+n_val],
              "test":  imgs[n_train+n_val:]}

    print(f"共 {total} 张 | 每张扩增 {AUGMENT_TIMES} 次 | Train/Val/Test = "
          f"{len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")

    make_dirs(OUTPUT_DIR)
    for sp, subset in splits.items():
        for f in subset:
            process_one(f, sp)
    print("✅ 数据增强与划分完成 →", OUTPUT_DIR)

# =========================================================
#  Part 2  —  一键训练 + 验证 + 测试
# =========================================================
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.registry import get_model
from utils.dataset  import SegmentationDataset
from losses.combo_loss import build_loss_fn
from utils.metrics import pixel_accuracy, mean_iou
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# 立即输出（避免 PyQt 缓冲）
print = lambda *a, **k: __import__("builtins").print(*a, flush=True, **k)

def resize_preds(preds, masks):
    if preds.ndim==4 and masks.ndim==3 and preds.shape[2:]!=masks.shape[1:]:
        preds = nn.functional.interpolate(preds, size=masks.shape[1:], mode="bilinear", align_corners=False)
    return preds

def save_ckpt(model, opt, epoch, best, path):
    torch.save({"epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "best_loss": best}, path)

def load_ckpt(model, opt, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    opt.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"📦 恢复至 Epoch {ckpt['epoch']+1}")
    return ckpt["epoch"], ckpt["best_loss"]


def evaluate(model, loader, criterion, cfg):
    model.eval()
    total_loss, acc_sum, miou_sum = 0, 0, 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(cfg["pretrain_device"]), masks.to(cfg["pretrain_device"])
            with autocast(enabled=cfg.get("use_amp", False)):
                preds = model(imgs)
                loss  = criterion(preds, masks)
            total_loss += loss.item()
            preds = resize_preds(preds, masks)
            acc_sum += pixel_accuracy(preds, masks).item()
            miou_sum += mean_iou(preds, masks, num_classes=cfg["out_channels"]).item()
    n = len(loader)
    return total_loss/n, acc_sum/n, miou_sum/n

def train():
    with open("config.json","r",encoding="utf-8") as f:
        cfg = json.load(f)

    device = cfg["pretrain_device"]
    model  = get_model(cfg["pretrain_model_name"],
                       cfg["in_channels"],
                       cfg["out_channels"]).to(device)
    split_base = os.path.join(cfg["annotate_img_dir"], "pretrain_split_data")
    train_loader = DataLoader(
        SegmentationDataset(os.path.join(split_base, "train", "images"),
                            os.path.join(split_base, "train", "masks"),                                                                                                                
                            cfg["input_size"],
                            augment=False),
        batch_size=cfg["pretrain_batch_size"], shuffle=True)

    opt  = optim.Adam(model.parameters(), lr=cfg["pretrain_lr"])
    crit = build_loss_fn(cfg)
    scaler = GradScaler(enabled=cfg.get("use_amp", False))
    accum = cfg.get("accum_iter",1)

    warm_epochs  = int(cfg["pretrain_epochs"]*0.2)
    warm_factor  = cfg.get("pretrain_warmup_factor",0.1)

    start_ep, best_loss = 0, float("inf")
    ckpt_path = os.path.join(cfg["pretrain_checkpoint_dir"], cfg["pretrain_checkpoint_filename"])
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    if os.path.exists(ckpt_path):
        start_ep, best_loss = load_ckpt(model, opt, ckpt_path, device)
    else:
        print("⚙️ 随机初始化权重")

    for epoch in range(start_ep, cfg["pretrain_epochs"]):
        model.train()
        running = 0
        opt.zero_grad()

        # 学习率 warm‑up
        if epoch < warm_epochs:
            lr = cfg["pretrain_lr"] * (warm_factor + (1-warm_factor)*epoch/warm_epochs)
            opt.param_groups[0]["lr"] = lr
            print(f"🚀 Warmup {epoch+1}/{warm_epochs} lr={lr:.6f}")
        else:
            opt.param_groups[0]["lr"] = cfg["pretrain_lr"]

        bar = tqdm(enumerate(train_loader), total=len(train_loader),
                   desc=f"Epoch {epoch+1}/{cfg['pretrain_epochs']} - Train", ascii=True, ncols=90)
        for step,(imgs,masks) in bar:
            imgs,masks = imgs.to(device), masks.to(device)
            with autocast(enabled=cfg.get("use_amp", False)):
                preds = model(imgs)
                loss  = crit(preds, masks)/accum
            scaler.scale(loss).backward()
            if (step+1)%accum==0 or (step+1)==len(train_loader):
                scaler.step(opt); scaler.update(); opt.zero_grad()
            running += loss.item()*accum
            bar.set_postfix(loss=loss.item()*accum)
        torch.save(model.state_dict(), os.path.join(cfg["pretrain_save_dir"], cfg["pretrain_save_filename"]))
        save_ckpt(model,opt,epoch,best_loss,ckpt_path)
# ---------------- 入口 ----------------
if __name__ == "__main__":
    main_augment()   # ① 生成 / 划分 / 增强
    train()          # ② 训练（自动包含验证和测试）
