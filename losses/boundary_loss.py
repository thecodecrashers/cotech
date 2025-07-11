import torch
import torch.nn.functional as F
import torch.nn as nn

def edge_map(tensor):
    """生成边界图（基于 Sobel）"""
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                           dtype=torch.float32, device=tensor.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                           dtype=torch.float32, device=tensor.device).unsqueeze(0).unsqueeze(0)

    edge_x = F.conv2d(tensor, sobel_x, padding=1)
    edge_y = F.conv2d(tensor, sobel_y, padding=1)
    edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)

    return (edge > 0).float()

def boundary_loss(pred, target):
    """
    边界损失：
    - pred: logits (B, 1, H, W)
    - target: mask (B, 1, H, W)
    """
    if pred.shape[1] > 1:
        pred = torch.sigmoid(pred[:, 1:2])  # 只提取前景类别通道
    else:
        pred = torch.sigmoid(pred)

    target = target.float()

    pred_edge = edge_map(pred)
    target_edge = edge_map(target)

    loss = F.binary_cross_entropy(pred_edge, target_edge)
    return loss
