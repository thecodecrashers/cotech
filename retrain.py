# retrain.py

import os
import shutil
from train import train
from config import config

def prepare_from_hard_cases():
    """自动将 hard_cases 图像加入训练集"""
    hard_img_dir = "data/hard_cases/images"
    hard_mask_dir = "data/hard_cases/masks"
    train_img_dir = config["train_img_dir"]
    train_mask_dir = config["train_mask_dir"]

    if not os.path.exists(hard_img_dir) or not os.path.exists(hard_mask_dir):
        print("⚠️ 未找到 hard_cases 图像与mask，跳过回流处理")
        return

    for fname in os.listdir(hard_img_dir):
        if fname.endswith((".png", ".jpg", ".bmp")):
            shutil.copy(os.path.join(hard_img_dir, fname), os.path.join(train_img_dir, "hc_" + fname))
            shutil.copy(os.path.join(hard_mask_dir, fname), os.path.join(train_mask_dir, "hc_" + fname))

    print(f"✅ 已将 hard_cases 增量样本复制到训练集，准备再次训练")

def main():
    prepare_from_hard_cases()
    train()

if __name__ == "__main__":
    main()
