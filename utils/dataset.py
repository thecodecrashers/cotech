import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random
import torch

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ico")

def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)

def pad_or_crop(img, target_size, is_mask=False):
    """将图像强制处理为指定大小（先中心裁切，再 padding）"""
    target_w, target_h = target_size
    w, h = img.size

    # === Step 1: 中心裁切（如果图像尺寸过大） ===
    left = max((w - target_w) // 2, 0)
    top = max((h - target_h) // 2, 0)
    right = min(left + target_w, w)
    bottom = min(top + target_h, h)

    img = img.crop((left, top, right, bottom))

    # === Step 2: padding（如果图像尺寸不足） ===
    new_w, new_h = img.size
    pad_left = max((target_w - new_w) // 2, 0)
    pad_top = max((target_h - new_h) // 2, 0)
    pad_right = target_w - new_w - pad_left
    pad_bottom = target_h - new_h - pad_top

    fill = 0 if is_mask else 0  # 背景填充为 0
    img = TF.pad(img, padding=(pad_left, pad_top, pad_right, pad_bottom), fill=fill)

    return img

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(512, 512), augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment

        self.image_names = sorted([
            f for f in os.listdir(image_dir)
            if is_image_file(f)
        ])
        self.mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(mask_dir)}

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_file = self.image_names[idx]
        basename = os.path.splitext(image_file)[0]

        img_path = os.path.join(self.image_dir, image_file)

        if basename not in self.mask_files:
            raise FileNotFoundError(f"❌ 找不到对应掩码：{basename}.* in {self.mask_dir}")

        mask_path = os.path.join(self.mask_dir, self.mask_files[basename])

        # 加载图像和掩码
        img = Image.open(img_path).convert("L")       # 灰度图像
        mask = Image.open(mask_path).convert("L")     # 掩码图（类别ID）

        # ⚠️ 使用裁切 + padding 处理尺寸
        img = pad_or_crop(img, self.image_size, is_mask=False)
        mask = pad_or_crop(mask, self.image_size, is_mask=True)

        # 数据增强（随机水平翻转）
        if self.augment and random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # 转为 Tensor
        img_tensor = TF.to_tensor(img)  # float32: [1, H, W]
        mask_tensor = torch.as_tensor(TF.pil_to_tensor(mask).squeeze(0), dtype=torch.long)  # int64: [H, W]

        return img_tensor, mask_tensor



