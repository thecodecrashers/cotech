# dice_loss.py
# losses/dice_loss.py

import torch

def dice_loss(pred, target, smooth=1.):
    """
    Dice Loss，适用于二值分割任务。
    pred: (B, 1, H, W) - logits（未sigmoid）
    target: (B, 1, H, W) - 0/1 标签
    """
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2 * intersection + smooth) / (union + smooth)
    loss = 1 - dice.mean()
    return loss
