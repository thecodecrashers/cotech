import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random
import torch

# 支持的图像扩展名（大写、小写都支持）
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ico")

def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(512, 512), augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment

        # 支持所有合法图像文件
        self.image_names = sorted([
            f for f in os.listdir(image_dir)
            if is_image_file(f)
        ])

        # 用于查找掩码路径（同名不同后缀也支持）
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
        mask = Image.open(mask_path).convert("L")     # 像素值为类别ID

        # 调整尺寸
        img = img.resize(self.image_size, resample=Image.BILINEAR)
        mask = mask.resize(self.image_size, resample=Image.NEAREST)  # 保证整数类别不变

        # 数据增强（可选）
        if self.augment and random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # 转为 Tensor
        img_tensor = TF.to_tensor(img)  # shape: [1, H, W]
        mask_tensor = torch.as_tensor(TF.pil_to_tensor(mask).squeeze(0), dtype=torch.long)  # shape: [H, W]

        return img_tensor, mask_tensor


