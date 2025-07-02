import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """两个卷积 + BN + ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    """下采样模块（MaxPool + DoubleConv）"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    """上采样模块（上采 + 拼接 + DoubleConv）"""
    def __init__(self, x1_ch, x2_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(x1_ch, x1_ch, 2, stride=2)

        self.conv = DoubleConv(x1_ch + x2_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 补齐尺寸差异（适配奇偶尺寸）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """输出卷积（1×1）"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """UNet 模型：支持多标签（二分类时 out_channels=1）"""
    def __init__(self, in_channels=1, out_channels=1, bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # 编码器
        self.inc    = DoubleConv(in_channels, 64)
        self.down1  = Down(64, 128)
        self.down2  = Down(128, 256)
        self.down3  = Down(256, 512)
        self.down4  = Down(512, 1024)

        # 解码器
        self.up1    = Up(1024, 512, 512, bilinear)
        self.up2    = Up(512,  256, 256, bilinear)
        self.up3    = Up(256,  128, 128, bilinear)
        self.up4    = Up(128,  64,  64,  bilinear)

        # 输出头
        self.outc   = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)  # ⚠️ 注意：未加 sigmoid，推理时需自行处理
        return logits
