# ✅ DeepLabV3（ResNet50 + ASPP 巨大化版）
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[1, 6, 12, 18, 24, 36]):
        super().__init__()
        self.branches = nn.ModuleList([
            ASPPConv(in_channels, out_channels, d) for d in dilations
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations)+1), out_channels * 2, 1, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[2:]
        res = [branch(x) for branch in self.branches]
        gp = F.interpolate(self.global_pool(x), size=size, mode='bilinear', align_corners=False)
        res.append(gp)
        x = torch.cat(res, dim=1)
        return self.project(x)

class ResNetBackbone(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        base_model = models.resnet101(weights=None)
        if in_channels != 3:
            base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4
        )

    def forward(self, x):
        return self.backbone(x)

class DeepLabV3(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.backbone = ResNetBackbone(in_channels)
        self.aspp = ASPP(2048, 1024)
        self.classifier = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, out_channels, 1)
        )

    def forward(self, x):
        size = x.shape[2:]
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.classifier(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)



"""# ✅ DeepLabV3（增强版：更多通道 + 更深特征 + ASPP更强）
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[1, 6, 12, 18, 24]):
        super().__init__()
        self.branches = nn.ModuleList([
            ASPPConv(in_channels, out_channels, d) for d in dilations
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations)+1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)  # 更强正则
        )

    def forward(self, x):
        size = x.shape[2:]
        res = [branch(x) for branch in self.branches]
        gp = F.interpolate(self.global_pool(x), size=size, mode='bilinear', align_corners=False)
        res.append(gp)
        x = torch.cat(res, dim=1)
        return self.project(x)

class SimpleBackbone(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.features(x)

class DeepLabV3(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.backbone = SimpleBackbone(in_channels)
        self.aspp = ASPP(1024, 512)
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 1)
        )

    def forward(self, x):
        size = x.shape[2:]
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.classifier(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)"""