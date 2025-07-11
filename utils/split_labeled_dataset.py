import os
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
