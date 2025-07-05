import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNetPP(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_ch=64, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision

        ch = base_ch
        self.conv00 = DoubleConv(in_channels, ch)
        self.conv10 = DoubleConv(ch, ch * 2)
        self.conv20 = DoubleConv(ch * 2, ch * 4)
        self.conv30 = DoubleConv(ch * 4, ch * 8)
        self.conv40 = DoubleConv(ch * 8, ch * 16)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        # Nested connections
        self.conv01 = DoubleConv(ch + ch * 2, ch)
        self.conv11 = DoubleConv(ch * 2 + ch * 4, ch * 2)
        self.conv21 = DoubleConv(ch * 4 + ch * 8, ch * 4)
        self.conv31 = DoubleConv(ch * 8 + ch * 16, ch * 8)

        self.conv02 = DoubleConv(ch * 2 + ch, ch)
        self.conv12 = DoubleConv(ch * 4 + ch * 2, ch * 2)
        self.conv22 = DoubleConv(ch * 8 + ch * 4, ch * 4)

        self.conv03 = DoubleConv(ch * 2 + ch, ch)
        self.conv13 = DoubleConv(ch * 4 + ch * 2, ch * 2)

        self.conv04 = DoubleConv(ch * 2 + ch, ch)

        # 输出头
        if deep_supervision:
            self.final1 = nn.Conv2d(ch, out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(ch, out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(ch, out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(ch, out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(ch, out_channels, kernel_size=1)

    def forward(self, x):
        x00 = self.conv00(x)
        x10 = self.conv10(self.maxpool(x00))
        x20 = self.conv20(self.maxpool(x10))
        x30 = self.conv30(self.maxpool(x20))
        x40 = self.conv40(self.maxpool(x30))

        x01 = self.conv01(torch.cat([x00, self.upsample(x10)], 1))
        x11 = self.conv11(torch.cat([x10, self.upsample(x20)], 1))
        x21 = self.conv21(torch.cat([x20, self.upsample(x30)], 1))
        x31 = self.conv31(torch.cat([x30, self.upsample(x40)], 1))

        x02 = self.conv02(torch.cat([x01, self.upsample(x11)], 1))
        x12 = self.conv12(torch.cat([x11, self.upsample(x21)], 1))
        x22 = self.conv22(torch.cat([x21, self.upsample(x31)], 1))

        x03 = self.conv03(torch.cat([x02, self.upsample(x12)], 1))
        x13 = self.conv13(torch.cat([x12, self.upsample(x22)], 1))

        x04 = self.conv04(torch.cat([x03, self.upsample(x13)], 1))

        if self.deep_supervision:
            return [
                self.final1(x01),
                self.final2(x02),
                self.final3(x03),
                self.final4(x04)
            ]
        else:
            return self.final(x04)
