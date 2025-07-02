# config.py

num_classes = 2  # 举例：背景+3类（改成你实际类别数量）

config = {
    "model_name": "unet",
    "in_channels": 1,
    "out_channels": num_classes,
    "input_size": (512, 512),
    "epochs": 3,
    "batch_size": 1,
    "lr": 1e-4,
    "device": "cuda" if __import__('torch').cuda.is_available() else "cpu",

    "train_img_dir": r"C:\Users\86178\Desktop\小可智能\裂纹\my_patches",
    "train_mask_dir": r"C:\Users\86178\Desktop\小可智能\裂纹\my_masks",
    "val_img_dir": r"C:\Users\86178\Desktop\小可智能\裂纹\my_patches",
    "val_mask_dir": r"C:\Users\86178\Desktop\小可智能\裂纹\my_masks",

    "save_path": r"C:\Users\86178\Desktop\小可智能\项目\checkpoint/best.pth",
    "checkpoint_path": r"C:\Users\86178\Desktop\小可智能\项目\checkpoint/checkpoint.pth",
    "log_csv": r"C:\Users\86178\Desktop\小可智能\项目\logs",

    "num_classes": num_classes,

    # 改为多类 loss，禁用 BCE/Dice/Focal
    "loss": {
        "use_ce": True,          # ✅ 启用交叉熵
        "use_bce": False,
        "use_dice": False,
        "use_focal": False
    }
}

