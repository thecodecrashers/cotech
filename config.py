# ====================🔧 模型与训练基础配置（供所有脚本使用） ====================
num_classes = 2  # 类别数：例如背景 + 裂纹

config = {
    "model_name": "unet",            # 模型名称（供 registry.py 使用：可选 "unet", "segformer" 等）
    "in_channels": 1,                # 输入通道数（灰度图用1）
    "out_channels": num_classes,     # 输出类别数（2表示二分类掩码）
    "input_size": (512, 512),        # 输入图像大小（供 dataset.py 使用）
    "num_classes": num_classes,      # 类别总数（部分模型可能用到）

    "device": "cuda" if __import__('torch').cuda.is_available() else "cpu",  # 所有模型/训练通用
    "batch_size": 1,                 # DataLoader 的 batch_size（train.py, fine_tune.py 使用）
    "accum_iter": 2,     # 每累计4个batch更新一次权重（适用于显存不够的情况）
    "use_amp": True,     # 启用混合精度训练（FP16）

    # ====================📂 路径设置（各文件夹中调用） ====================
    "train_img_dir": r"C:\Users\86178\Desktop\小可智能\裂纹\my_patches",  # 训练图像路径（train.py, fine_tune.py）
    "train_mask_dir": r"C:\Users\86178\Desktop\小可智能\裂纹\my_masks",   # 训练掩码路径

    "val_img_dir": r"C:\Users\86178\Desktop\小可智能\裂纹\my_patches",    # 验证图像路径（train.py）
    "val_mask_dir": r"C:\Users\86178\Desktop\小可智能\裂纹\my_masks",     # 验证掩码路径

    "save_path": r"C:\Users\86178\Desktop\小可智能\项目\checkpoint/best.pth",         # 最优模型保存路径（train.py）
    "checkpoint_path": r"C:\Users\86178\Desktop\小可智能\项目\checkpoint/checkpoint.pth", # 中断点模型保存路径（train.py）
    "log_csv": r"C:\Users\86178\Desktop\小可智能\项目\logs",                         # 训练损失日志路径（train.py）
    "val_vis_dir": "val_vis",     # 可视化输出路径（validate.py）

    # ====================🧠 学习率与 epoch 配置（train.py / fine_tune.py） ====================
    "epochs": 2,                  # 训练 epoch（train.py 使用）
    "lr": 1e-4,                   # 初始学习率（train.py / fine_tune.py 都用到）

    # ====================📉 损失函数设置（build_loss_fn 使用） ====================
    "loss": {
        "use_ce": True,           # ✅ 多分类任务使用 CrossEntropyLoss
        "use_bce": False,         # ❌ 禁用 BCE（用于二分类）
        "use_dice": False,        # ❌ 禁用 Dice Loss
        "use_focal": False        # ❌ 禁用 Focal Loss
    }
}

