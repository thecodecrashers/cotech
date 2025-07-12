import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ico")

def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)

def pad_image(img, pad_left, pad_top, pad_right, pad_bottom, fill=0):
    return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)

def sliding_window_pad(img, mask, window_size, stride):
    """先padding图像，再滑窗提取patch"""
    w, h = img.size
    win_w, win_h = window_size
    stride_w, stride_h = stride

    # === padding 使图像可被滑窗整除 ===
    pad_w = (-(w - win_w) % stride_w) if w > win_w else max(0, win_w - w)
    pad_h = (-(h - win_h) % stride_h) if h > win_h else max(0, win_h - h)

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    img = pad_image(img, pad_left, pad_top, pad_right, pad_bottom, fill=0)
    mask = pad_image(mask, pad_left, pad_top, pad_right, pad_bottom, fill=0)

    W, H = img.size
    patches = []

    for top in range(0, H - win_h + 1, stride_h):
        for left in range(0, W - win_w + 1, stride_w):
            box = (left, top, left + win_w, top + win_h)
            img_patch = img.crop(box)
            mask_patch = mask.crop(box)
            patches.append((img_patch, mask_patch))

    return patches

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(512, 512), augment=False, use_sliding=True, stride=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment
        self.use_sliding = use_sliding
        self.stride = stride or (image_size[0] // 2, image_size[1] // 2)

        self.image_names = sorted([f for f in os.listdir(image_dir) if is_image_file(f)])
        self.mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(mask_dir)}

        self.sliding_index = []  # list of (img_idx, patch_idx)
        self.patch_dict = {}     # (img_idx) -> [(img, mask), ...]

        for img_idx, image_file in enumerate(self.image_names):
            basename = os.path.splitext(image_file)[0]
            img_path = os.path.join(self.image_dir, image_file)
            mask_path = os.path.join(self.mask_dir, self.mask_files.get(basename, ""))

            if not os.path.exists(mask_path):
                continue

            img = Image.open(img_path).convert("L")
            mask = Image.open(mask_path).convert("L")
            w, h = img.size

            if self.use_sliding and (w > self.image_size[0] or h > self.image_size[1]):
                patches = sliding_window_pad(img, mask, self.image_size, self.stride)
                self.patch_dict[img_idx] = patches
                for p_idx in range(len(patches)):
                    self.sliding_index.append((img_idx, p_idx))
            else:
                img = pad_image(img, 0, 0, max(0, self.image_size[0] - w), max(0, self.image_size[1] - h))
                mask = pad_image(mask, 0, 0, max(0, self.image_size[0] - w), max(0, self.image_size[1] - h))
                self.patch_dict[img_idx] = [(img, mask)]
                self.sliding_index.append((img_idx, 0))

    def __len__(self):
        return len(self.sliding_index)

    def __getitem__(self, index):
        img_idx, patch_idx = self.sliding_index[index]
        img, mask = self.patch_dict[img_idx][patch_idx]

        # 数据增强
        if self.augment and random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        img_tensor = TF.to_tensor(img)  # float32: [1, H, W]
        mask_tensor = torch.as_tensor(TF.pil_to_tensor(mask).squeeze(0), dtype=torch.long)  # int64: [H, W]

        return img_tensor, mask_tensor
