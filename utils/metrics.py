import torch

def flatten_probs(preds, targets, threshold=0.5):
    """
    将预测与标签 flatten，转为二值化，便于后续指标计算。
    """
    probs = torch.sigmoid(preds)
    preds_bin = (probs > threshold).float()
    return preds_bin.view(-1), targets.view(-1)

def dice_coef(preds, targets, threshold=0.5, smooth=1e-6):
    pred, target = flatten_probs(preds, targets, threshold)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(preds, targets, threshold=0.5, smooth=1e-6):
    pred, target = flatten_probs(preds, targets, threshold)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def precision(preds, targets, threshold=0.5, smooth=1e-6):
    pred, target = flatten_probs(preds, targets, threshold)
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return (tp + smooth) / (tp + fp + smooth)

def recall(preds, targets, threshold=0.5, smooth=1e-6):
    pred, target = flatten_probs(preds, targets, threshold)
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return (tp + smooth) / (tp + fn + smooth)

def f1_score(preds, targets, threshold=0.5, smooth=1e-6):
    p = precision(preds, targets, threshold, smooth)
    r = recall(preds, targets, threshold, smooth)
    return 2 * p * r / (p + r + smooth)

def accuracy(preds, targets, threshold=0.5):
    pred, target = flatten_probs(preds, targets, threshold)
    correct = (pred == target).sum()
    return correct.float() / len(target)

def specificity(preds, targets, threshold=0.5, smooth=1e-6):
    pred, target = flatten_probs(preds, targets, threshold)
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    return (tn + smooth) / (tn + fp + smooth)
