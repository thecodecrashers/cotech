import os
import json
import shutil
import random
from PIL import Image, ImageDraw, ImageEnhance, ImageOps, ImageFilter
import numpy as np

# ====== 加载配置 ======
config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

DATA_DIR            = config["annotate_img_dir"]
OUTPUT_DIR          = os.path.join(DATA_DIR, "split_data")
JSON_EXT            = ".json"
TRAIN_RATIO         = config["preprocess_train_ratio"]
VAL_RATIO           = config["preprocess_val_ratio"]
TEST_RATIO          = config["preprocess_test_ratio"]
AUGMENT_TIMES       = config["preprocess_augment_times"]
INCLUDE_ORIGINAL    = config["preprocess_include_original"]

SUPPORTED_IMG_EXTS  = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def make_dirs(base_dir):
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(base_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, "masks"), exist_ok=True)

def load_labelme_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    w, h = data["imageWidth"], data["imageHeight"]
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    label_map = {"_background_": 0}
    label_id_counter = 1

    for shape in data["shapes"]:
        label = shape.get("label", "_background_")
        if label not in label_map:
            label_map[label] = label_id_counter
            label_id_counter += 1
        class_id = label_map[label]
        draw.polygon(shape["points"], fill=class_id)

    return mask



def add_gaussian_noise(image, mean=0, std=10):
    np_img = np.array(image).astype(np.float32)
    noise = np.random.normal(mean, std, np_img.shape).astype(np.float32)
    noisy = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_random_augment(img: Image.Image, mask: Image.Image):
    """图像增强函数：图像和掩码同步"""
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
        mask = ImageOps.mirror(mask)
    if random.random() < 0.5:
        img = ImageOps.flip(img)
        mask = ImageOps.flip(mask)
    if random.random() < 0.3:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.3:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.3:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    if random.random() < 0.3:
        img = add_gaussian_noise(img, std=random.uniform(5, 20))

    return img, mask


def save_pair(image, mask, base_name, index, split, ext):
    """保存图像 + 掩码"""
    img_dst = os.path.join(OUTPUT_DIR, split, "images", f"{base_name}_aug{index}{ext}")
    mask_dst = os.path.join(OUTPUT_DIR, split, "masks", f"{base_name}_aug{index}{ext}")
    image.save(img_dst)
    mask.save(mask_dst)


def process_sample(img_file, split):
    base_name, ext = os.path.splitext(img_file)
    img_path = os.path.join(DATA_DIR, img_file)
    json_path = os.path.join(DATA_DIR, base_name + JSON_EXT)
    if not os.path.exists(json_path):
        return

    image = Image.open(img_path).convert("RGB")
    mask = load_labelme_json(json_path)

    idx = 0
    if INCLUDE_ORIGINAL:
        save_pair(image, mask, base_name, idx, split, ext)
        idx += 1

    for i in range(AUGMENT_TIMES):
        img_aug, mask_aug = apply_random_augment(image.copy(), mask.copy())
        save_pair(img_aug, mask_aug, base_name, idx, split, ext)
        idx += 1


def main():
    all_imgs = [f for f in os.listdir(DATA_DIR)
                if os.path.splitext(f)[1].lower() in SUPPORTED_IMG_EXTS]
    all_bases = [f for f in all_imgs
                 if os.path.exists(os.path.join(DATA_DIR, os.path.splitext(f)[0] + JSON_EXT))]

    random.shuffle(all_bases)
    total = len(all_bases)
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)

    train_set = all_bases[:n_train]
    val_set = all_bases[n_train:n_train + n_val]
    test_set = all_bases[n_train + n_val:]

    print(f"共找到 {total} 张原始图片")
    print(f"每图生成 {AUGMENT_TIMES} 个增强样本（含原图：{INCLUDE_ORIGINAL}）")
    print(f"划分：Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

    make_dirs(OUTPUT_DIR)

    for split, subset in zip(["train", "val", "test"], [train_set, val_set, test_set]):
        for img_file in subset:
            process_sample(img_file, split)

    print(" 数据增强与划分完成，保存在：", OUTPUT_DIR)


if __name__ == "__main__":
    main()






# import os
# import json
# import shutil
# import random
# from PIL import Image, ImageDraw, ImageEnhance, ImageOps, ImageFilter
# import numpy as np

# config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
# with open(config_path, "r", encoding="utf-8") as f:
#     config = json.load(f)

# DATA_DIR            = config["annotate_img_dir"]
# OUTPUT_DIR = os.path.join(DATA_DIR, "split_data")
# IMG_EXT             = config["preprocess_img_ext"]
# JSON_EXT            = ".json"
# TRAIN_RATIO         = config["preprocess_train_ratio"]
# VAL_RATIO           = config["preprocess_val_ratio"]
# TEST_RATIO          = config["preprocess_test_ratio"]
# AUGMENT_TIMES       = config["preprocess_augment_times"]
# INCLUDE_ORIGINAL    = config["preprocess_include_original"]


# def make_dirs(base_dir):
#     for split in ["train", "val", "test"]:
#         os.makedirs(os.path.join(base_dir, split, "images"), exist_ok=True)
#         os.makedirs(os.path.join(base_dir, split, "masks"), exist_ok=True)


# def load_labelme_json(json_path):
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     w, h = data["imageWidth"], data["imageHeight"]
#     mask = Image.new("L", (w, h), 0)
#     draw = ImageDraw.Draw(mask)
#     for shape in data["shapes"]:
#         draw.polygon(shape["points"], fill=255)
#     return mask


# def add_gaussian_noise(image, mean=0, std=10):
#     np_img = np.array(image).astype(np.float32)
#     noise = np.random.normal(mean, std, np_img.shape).astype(np.float32)
#     noisy = np.clip(np_img + noise, 0, 255).astype(np.uint8)
#     return Image.fromarray(noisy)


# def apply_random_augment(img: Image.Image, mask: Image.Image):
#     """图像增强函数：图像和掩码同步"""
#     if random.random() < 0.5:
#         img = ImageOps.mirror(img)
#         mask = ImageOps.mirror(mask)
#     if random.random() < 0.5:
#         img = ImageOps.flip(img)
#         mask = ImageOps.flip(mask)
#     if random.random() < 0.3:
#         enhancer = ImageEnhance.Brightness(img)
#         img = enhancer.enhance(random.uniform(0.8, 1.2))
#     if random.random() < 0.3:
#         enhancer = ImageEnhance.Contrast(img)
#         img = enhancer.enhance(random.uniform(0.8, 1.2))
#     if random.random() < 0.3:
#         enhancer = ImageEnhance.Color(img)
#         img = enhancer.enhance(random.uniform(0.8, 1.2))
#     if random.random() < 0.3:
#         img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
#     if random.random() < 0.3:
#         img = add_gaussian_noise(img, std=random.uniform(5, 20))

#     return img, mask


# def save_pair(image, mask, base_name, index, split):
#     """保存图像 + 掩码"""
#     img_dst = os.path.join(OUTPUT_DIR, split, "images", f"{base_name}_aug{index}{IMG_EXT}")
#     mask_dst = os.path.join(OUTPUT_DIR, split, "masks", f"{base_name}_aug{index}{IMG_EXT}")
#     image.save(img_dst)
#     mask.save(mask_dst)


# def process_sample(base_name, split):
#     img_path = os.path.join(DATA_DIR, base_name + IMG_EXT)
#     json_path = os.path.join(DATA_DIR, base_name + JSON_EXT)

#     image = Image.open(img_path).convert("RGB")
#     mask = load_labelme_json(json_path)

#     idx = 0
#     if INCLUDE_ORIGINAL:
#         save_pair(image, mask, base_name, idx, split)
#         idx += 1

#     for i in range(AUGMENT_TIMES):
#         img_aug, mask_aug = apply_random_augment(image.copy(), mask.copy())
#         save_pair(img_aug, mask_aug, base_name, idx, split)
#         idx += 1


# def main():
#     all_imgs = [f for f in os.listdir(DATA_DIR) if f.endswith(IMG_EXT)]
#     all_bases = [os.path.splitext(f)[0] for f in all_imgs if os.path.exists(os.path.join(DATA_DIR, os.path.splitext(f)[0] + JSON_EXT))]

#     random.shuffle(all_bases)
#     total = len(all_bases)
#     n_train = int(total * TRAIN_RATIO)
#     n_val = int(total * VAL_RATIO)

#     train_set = all_bases[:n_train]
#     val_set = all_bases[n_train:n_train + n_val]
#     test_set = all_bases[n_train + n_val:]

#     print(f"共找到 {total} 张原始图片")
#     print(f"每图生成 {AUGMENT_TIMES} 个增强样本（含原图：{INCLUDE_ORIGINAL}）")
#     print(f"划分：Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

#     make_dirs(OUTPUT_DIR)

#     for split, subset in zip(["train", "val", "test"], [train_set, val_set, test_set]):
#         for base_name in subset:
#             process_sample(base_name, split)

#     print("✅ 数据增强与划分完成，保存在：", OUTPUT_DIR)


# if __name__ == "__main__":
#     main()






"""import os
import json
import shutil
import random
from PIL import Image, ImageDraw

# ================= 参数设置 =================
DATA_DIR = r"C:\YourDataset\all"           # 原始图片+json 所在目录
OUTPUT_DIR = r"C:\YourDataset\split_data"  # 输出路径

IMG_EXT = ".png"
JSON_EXT = ".json"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

MASK_SIZE = None  # 如果需要固定大小，可设为如 (512, 512)，否则为 None
# ===========================================

def make_dirs(base_dir):
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(base_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, "masks"), exist_ok=True)

def load_labelme_json(json_path, mask_size):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_shape = data.get("imageHeight"), data.get("imageWidth")
    if mask_size:
        mask = Image.new("L", mask_size, 0)
    else:
        mask = Image.new("L", (image_shape[1], image_shape[0]), 0)

    draw = ImageDraw.Draw(mask)
    for shape in data["shapes"]:
        points = shape["points"]
        draw.polygon(points, fill=255)

    return mask

def copy_and_generate_mask(base_name, split):
    img_src = os.path.join(DATA_DIR, base_name + IMG_EXT)
    json_src = os.path.join(DATA_DIR, base_name + JSON_EXT)

    img_dst = os.path.join(OUTPUT_DIR, split, "images", base_name + IMG_EXT)
    mask_dst = os.path.join(OUTPUT_DIR, split, "masks", base_name + IMG_EXT)  # 用相同文件名

    # 拷贝原图
    shutil.copy2(img_src, img_dst)

    # 生成并保存掩码图
    mask = load_labelme_json(json_src, MASK_SIZE)
    mask.save(mask_dst)

def main():
    all_imgs = [f for f in os.listdir(DATA_DIR) if f.endswith(IMG_EXT)]
    all_pairs = [os.path.splitext(f)[0] for f in all_imgs if os.path.exists(os.path.join(DATA_DIR, os.path.splitext(f)[0] + JSON_EXT))]

    random.shuffle(all_pairs)
    total = len(all_pairs)
    train_n = int(total * TRAIN_RATIO)
    val_n = int(total * VAL_RATIO)

    train_set = all_pairs[:train_n]
    val_set = all_pairs[train_n:train_n + val_n]
    test_set = all_pairs[train_n + val_n:]

    print(f"共找到 {total} 个样本")
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    make_dirs(OUTPUT_DIR)

    for split, subset in zip(["train", "val", "test"], [train_set, val_set, test_set]):
        for base_name in subset:
            copy_and_generate_mask(base_name, split)

    print("✅ 所有图片和掩码已划分完成，结果保存在：", OUTPUT_DIR)

if __name__ == "__main__":
    main()
"""