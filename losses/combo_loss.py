# losses/combo_loss.py

import torch.nn as nn
from .dice_loss import dice_loss
from .focal_loss import focal_loss

def build_loss_fn(config):
    """构建适用于多类或多标签的组合损失"""

    loss_cfg = config.get("loss", {})
    num_classes = config.get("num_classes", 1)

    if loss_cfg.get("use_ce", False):  # 多类交叉熵模式
        return nn.CrossEntropyLoss()

    # 默认回退为多标签组合损失
    def total_loss(pred, target):
        loss_val = 0.0
        if loss_cfg.get("use_bce", True):
            loss_val += nn.BCEWithLogitsLoss()(pred, target)
        if loss_cfg.get("use_dice", False):
            loss_val += dice_loss(pred, target)
        if loss_cfg.get("use_focal", False):
            loss_val += focal_loss(pred, target,
                                   alpha=loss_cfg.get("focal_alpha", 0.25),
                                   gamma=loss_cfg.get("focal_gamma", 2.0))
        return loss_val

    return total_loss
