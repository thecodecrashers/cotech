import torch
import time

# === 二分类指标 ===
def flatten_probs(preds, targets, threshold=0.5):
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

# === 多分类语义分割指标 ===
def pixel_accuracy(preds, targets):
    preds = torch.argmax(preds, dim=1)
    correct = (preds == targets).float()
    return correct.sum() / correct.numel()

def mean_iou(preds, targets, num_classes, ignore_index=None):
    preds = torch.argmax(preds, dim=1)
    ious = []
    for cls in range(num_classes):
        if ignore_index is not None and cls == ignore_index:
            continue
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou = torch.tensor(1.0)
        else:
            iou = intersection / union
        ious.append(iou)
    return torch.stack(ious).mean()

def mean_iou_per_class(conf_matrix):
    class_iou = []
    for i in range(conf_matrix.shape[0]):
        TP = conf_matrix[i, i]
        FP = conf_matrix[:, i].sum() - TP
        FN = conf_matrix[i, :].sum() - TP
        denom = TP + FP + FN
        iou = TP / denom if denom != 0 else 0.0
        class_iou.append(iou)
    return sum(class_iou) / len(class_iou)

def pixel_accuracy_confmat(conf_matrix):
    return torch.diag(conf_matrix).sum() / conf_matrix.sum()

def class_accuracy(conf_matrix):
    recall_list = []
    for i in range(conf_matrix.shape[0]):
        TP = conf_matrix[i, i]
        FN = conf_matrix[i, :].sum() - TP
        denom = TP + FN
        recall = TP / denom if denom != 0 else 0.0
        recall_list.append(recall)
    return recall_list

# === 模型推理性能指标 ===
def measure_inference_speed(model, input_size=(1, 1, 512, 512), device="cuda", warmup=10, runs=50):
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(runs):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        end = time.time()
    avg_time = (end - start) / runs * 1000
    return avg_time

def measure_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb
