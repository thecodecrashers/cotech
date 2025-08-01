import os
import json
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from models.registry import get_model
from utils.dataset import SegmentationDataset

# ==== 加载配置 ====
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("✅ 使用设备:", device)

# ==== 加载模型 ====
model = get_model(
    config['model_name'],
    config['in_channels'],
    config['out_channels'],
    config.get("freeze_mode", "none")
).to(device)

model.load_state_dict(torch.load(
    os.path.join(config["save_dir"], config["save_filename"]),
    map_location=device
))

# ==== 构建数据集 ====
finetune_dataset = SegmentationDataset(
    config['fine_tune_img_dir'],
    config['fine_tune_mask_dir'],
    augment=False
)

train_loader = DataLoader(
    finetune_dataset,
    batch_size=config['fine_tune_batch_size'],
    shuffle=True,
    drop_last=False,
    num_workers=4,
    pin_memory=True
)

# ==== 优化器与损失函数 ====
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=config['fine_tune_lr']
)

criterion = torch.nn.CrossEntropyLoss()

# ==== AMP 和梯度累积 ====
scaler = GradScaler()
accum_steps = 2  # 梯度累积步数，设置为2表示每2个batch更新一次梯度

# ==== 训练循环 ====
for epoch in range(config['fine_tune_epochs']):
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['fine_tune_epochs']}", unit="batch")
    optimizer.zero_grad()

    for step, (imgs, masks) in enumerate(pbar):
        imgs, masks = imgs.to(device), masks.to(device)

        with autocast():  # ✅ 混合精度 forward
            outputs = model(imgs)
            loss = criterion(outputs, masks) / accum_steps  # ✅ 梯度累积

        scaler.scale(loss).backward()

        # ✅ 梯度更新控制
        if (step + 1) % accum_steps == 0 or (step + 1 == len(train_loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # ✅ tqdm 显示
        running_loss += loss.item() * accum_steps  # 累积时除掉了，这里乘回来
        pbar.set_postfix(loss=loss.item() * accum_steps)

    avg_loss = running_loss / len(train_loader)
    print(f"📘 Epoch [{epoch+1}/{config['fine_tune_epochs']}], Avg Loss: {avg_loss:.4f}")

# ==== 保存模型 ====
torch.save(model.state_dict(), config['fine_tune_model_save_path'])
print("✅ 微调完成，模型已保存至:", config['fine_tune_model_save_path'])


# import torch
# from torch.utils.data import DataLoader
# from models.registry import get_model
# from utils.dataset import SegmentationDataset
# import json
# import os
# from tqdm import tqdm  # ✅ 加载进度条库

# # ==== 加载配置 ====
# with open('config.json', 'r', encoding='utf-8') as f:
#     config = json.load(f)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("设备:", device)

# # ==== 加载模型 ====
# model = get_model(config['model_name'], config['in_channels'], config['out_channels'],config["freeze_mode"]).to(device)
# model.load_state_dict(torch.load(os.path.join(config["save_dir"], config["save_filename"]), map_location=device))

# # ==== 构建数据集 ====
# finetune_dataset = SegmentationDataset(config['fine_tune_img_dir'], config['fine_tune_mask_dir'], augment=True)
# #train_loader = DataLoader(finetune_dataset, batch_size=config['fine_tune_batch_size'], shuffle=True, drop_last=True)
# train_loader = DataLoader(
#     finetune_dataset,
#     batch_size=config['fine_tune_batch_size'],
#     shuffle=True,
#     drop_last=False,
#     num_workers=4,           # ✅ 多线程加速
#     pin_memory=True          # ✅ 提升 GPU 传输效率
# )


# # ==== 优化器与损失函数 ====
# # optimizer = torch.optim.Adam(model.parameters(), lr=config['fine_tune_lr'])
# optimizer = torch.optim.Adam(
#     filter(lambda p: p.requires_grad, model.parameters()),
#     lr=config['fine_tune_lr']
# )

# criterion = torch.nn.CrossEntropyLoss()

# # ==== 训练循环 ====
# for epoch in range(config['fine_tune_epochs']):
#     model.train()
#     running_loss = 0.0

#     # ✅ tqdm 包裹 train_loader，显示进度条
#     pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['fine_tune_epochs']}", unit="batch")

#     for imgs, masks in pbar:
#         imgs, masks = imgs.to(device), masks.to(device)
#         optimizer.zero_grad()
#         outputs = model(imgs)
#         loss = criterion(outputs, masks)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()

#         # ✅ 实时更新 tqdm 显示当前 loss
#         pbar.set_postfix(loss=loss.item())

#     avg_loss = running_loss / len(train_loader)
#     print(f"✅ Epoch [{epoch+1}/{config['fine_tune_epochs']}], Avg Loss: {avg_loss:.4f}")

# # ==== 保存模型 ====
# torch.save(model.state_dict(), config['fine_tune_model_save_path'])
# print("✅ Finetune done, model saved.")






# import os
# import torch
# from torch import nn, optim
# from torch.utils.data import DataLoader
# from models.registry import get_model
# from utils.dataset import SegmentationDataset
# from config import config
# from losses.combo_loss import build_loss_fn
# from tqdm import tqdm
# from torch.cuda.amp import autocast, GradScaler
# from utils.metrics import pixel_accuracy, mean_iou


# def align_prediction_size(preds, masks):
#     if preds.ndim == 4 and masks.ndim == 3:
#         if preds.shape[2:] != masks.shape[1:]:
#             preds = nn.functional.interpolate(preds, size=masks.shape[1:], mode="bilinear", align_corners=False)
#     else:
#         raise ValueError(f"Unsupported shape: preds={preds.shape}, masks={masks.shape}")
#     return preds


# def freeze_layers(model):
#     if config.get("freeze_encoder", False):
#         print("🧊 冻结编码器层")
#         if hasattr(model, "encoder"):
#             for param in model.encoder.parameters():
#                 param.requires_grad = False
#         else:
#             print("⚠️ 未找到 'encoder' 属性，无法冻结")


# def fine_tune():
#     assert os.path.exists(config["fine_tune_img_dir"]), "❌ 未找到 fine_tune_img_dir"
#     assert os.path.exists(config["fine_tune_mask_dir"]), "❌ 未找到 fine_tune_mask_dir"
#     os.makedirs(os.path.dirname(config["fine_tune_save_path"]), exist_ok=True)

#     # === 准备模型 ===
#     model = get_model(config["model_name"], config["in_channels"], config["out_channels"]).to(config["device"])
#     model.load_state_dict(torch.load(config["save_path"], map_location=config["device"]))
#     freeze_layers(model)
#     model.train()

#     # === 微调数据集 ===
#     fine_tune_set = SegmentationDataset(
#         config["fine_tune_img_dir"],
#         config["fine_tune_mask_dir"],
#         config["input_size"],
#         augment=True
#     )
#     loader = DataLoader(fine_tune_set, batch_size=config["fine_tune_batch_size"], shuffle=True)

#     # === 优化器和损失函数 ===
#     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["fine_tune_lr"])
#     criterion = build_loss_fn(config)

#     # === 混合精度和累积梯度 ===
#     use_amp = config.get("use_amp", False)
#     accum_iter = config.get("accum_iter", 1)
#     scaler = GradScaler(enabled=use_amp)

#     # === 微调训练 ===
#     for epoch in range(config["fine_tune_epochs"]):
#         model.train()
#         total_loss = 0
#         optimizer.zero_grad()
#         bar = tqdm(enumerate(loader), total=len(loader), desc=f"[微调] Epoch {epoch+1}/{config['fine_tune_epochs']}")

#         for step, (imgs, masks) in bar:
#             imgs, masks = imgs.to(config["device"]), masks.to(config["device"])

#             with autocast(enabled=use_amp):
#                 preds = model(imgs)
#                 loss = criterion(preds, masks) / accum_iter

#             scaler.scale(loss).backward()

#             if (step + 1) % accum_iter == 0 or (step + 1) == len(loader):
#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad()

#             total_loss += loss.item() * accum_iter
#             bar.set_postfix(loss=loss.item() * accum_iter)

#         print(f"📉 Epoch {epoch+1}: Fine-tune loss: {total_loss / len(loader):.4f}")

#     # === 保存微调模型 ===
#     torch.save(model.state_dict(), config["fine_tune_save_path"])
#     print(f"✅ 微调完成，模型保存至: {config['fine_tune_save_path']}")


# if __name__ == "__main__":
#     fine_tune()
