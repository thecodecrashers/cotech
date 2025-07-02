# focal_loss.py
# losses/focal_loss.py

import torch
import torch.nn.functional as F

def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    """
    Focal Loss，用于处理类别不平衡问题
    logits: (B, 1, H, W) - raw output（未sigmoid）
    targets: (B, 1, H, W) - 0/1 标签
    """
    probs = torch.sigmoid(logits)
    targets = targets.float()

    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = probs * targets + (1 - probs) * (1 - targets)
    modulating_factor = (1 - p_t) ** gamma
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)

    focal_loss = alpha_factor * modulating_factor * ce_loss

    if reduction == "mean":
        return focal_loss.mean()
    elif reduction == "sum":
        return focal_loss.sum()
    else:
        return focal_loss
