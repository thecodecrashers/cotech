import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random
import torch

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(512, 512), augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment

        self.image_names = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith((".jpg", ".png", ".bmp"))
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.image_names[idx])

        # 加载图像和mask
        img = Image.open(img_path).convert("L")      # 输入灰度图
        mask = Image.open(mask_path).convert("L")    # 假设像素值 = 类别id

        # 统一尺寸
        img = img.resize(self.image_size, resample=Image.BILINEAR)
        mask = mask.resize(self.image_size, resample=Image.NEAREST)  # 注意！类别图必须用最近邻

        # 数据增强（随机水平翻转）
        if self.augment and random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # 转为 Tensor
        img = TF.to_tensor(img)                      # shape: [1, H, W], float32
        mask = torch.from_numpy(
            (TF.pil_to_tensor(mask)).numpy().squeeze(0).astype("int64")
        )  # shape: [H, W], long

        return img, mask

