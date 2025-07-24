import torch
import torch.nn as nn
import torch.nn.functional as F

def upsample_to(x, target):
    return F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)

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
    """
    UNet++ 支持 freeze_mode：
    - "none": 全部训练
    - "partial": 冻结 conv00~conv40 + maxpool，只训练嵌套路径 conv01+ 后续
    - "full_backbone": 冻结所有 convXX，只训练输出 head
    - "all": 全部冻结
    """
    def __init__(self, in_channels=1, out_channels=1, base_ch=64, deep_supervision=False, freeze_mode="none"):
        super().__init__()
        self.deep_supervision = deep_supervision

        ch = base_ch
        self.conv00 = DoubleConv(in_channels, ch)
        self.conv10 = DoubleConv(ch, ch * 2)
        self.conv20 = DoubleConv(ch * 2, ch * 4)
        self.conv30 = DoubleConv(ch * 4, ch * 8)
        self.conv40 = DoubleConv(ch * 8, ch * 16)

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

        if deep_supervision:
            self.final1 = nn.Conv2d(ch, out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(ch, out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(ch, out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(ch, out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(ch, out_channels, kernel_size=1)

        self.maxpool = nn.MaxPool2d(2)

        # 应用 freeze_mode
        self.apply_freeze(freeze_mode)

    def apply_freeze(self, mode="none"):
        assert mode in ["none", "partial", "full_backbone", "all"], f"Invalid freeze_mode: {mode}"

        def freeze(module, do_freeze=True):
            for param in module.parameters():
                param.requires_grad = not do_freeze

        if mode == "none":
            freeze(self, False)

        elif mode == "partial":
            # 冻结主干下采路径
            freeze(self.conv00, True)
            freeze(self.conv10, True)
            freeze(self.conv20, True)
            freeze(self.conv30, True)
            freeze(self.conv40, True)
            freeze(self.maxpool, True)
            # 解冻其余部分
            for name, module in self.named_modules():
                if name.startswith("conv") and not name.startswith(("conv00", "conv10", "conv20", "conv30", "conv40")):
                    freeze(module, False)
            if self.deep_supervision:
                freeze(self.final1, False)
                freeze(self.final2, False)
                freeze(self.final3, False)
                freeze(self.final4, False)
            else:
                freeze(self.final, False)

        elif mode == "full_backbone":
            # 冻结所有 conv 模块
            for name, module in self.named_modules():
                if name.startswith("conv"):
                    freeze(module, True)
            freeze(self.maxpool, True)
            # 只解冻输出头
            if self.deep_supervision:
                freeze(self.final1, False)
                freeze(self.final2, False)
                freeze(self.final3, False)
                freeze(self.final4, False)
            else:
                freeze(self.final, False)

        elif mode == "all":
            freeze(self, True)

    def forward(self, x):
        x00 = self.conv00(x)
        x10 = self.conv10(self.maxpool(x00))
        x20 = self.conv20(self.maxpool(x10))
        x30 = self.conv30(self.maxpool(x20))
        x40 = self.conv40(self.maxpool(x30))

        x01 = self.conv01(torch.cat([x00, upsample_to(x10, x00)], dim=1))
        x11 = self.conv11(torch.cat([x10, upsample_to(x20, x10)], dim=1))
        x21 = self.conv21(torch.cat([x20, upsample_to(x30, x20)], dim=1))
        x31 = self.conv31(torch.cat([x30, upsample_to(x40, x30)], dim=1))

        x02 = self.conv02(torch.cat([x01, upsample_to(x11, x01)], dim=1))
        x12 = self.conv12(torch.cat([x11, upsample_to(x21, x11)], dim=1))
        x22 = self.conv22(torch.cat([x21, upsample_to(x31, x21)], dim=1))

        x03 = self.conv03(torch.cat([x02, upsample_to(x12, x02)], dim=1))
        x13 = self.conv13(torch.cat([x12, upsample_to(x22, x12)], dim=1))

        x04 = self.conv04(torch.cat([x03, upsample_to(x13, x03)], dim=1))

        if self.deep_supervision:
            return [
                self.final1(x01),
                self.final2(x02),
                self.final3(x03),
                self.final4(x04)
            ]
        else:
            return self.final(x04)







# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# def upsample_to(x, target):
#     """上采样到指定特征图尺寸"""
#     return F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=True)


# class DoubleConv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.block(x)


# class UNetPP(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, base_ch=64, deep_supervision=False):
#         super().__init__()
#         self.deep_supervision = deep_supervision

#         ch = base_ch
#         self.conv00 = DoubleConv(in_channels, ch)
#         self.conv10 = DoubleConv(ch, ch * 2)
#         self.conv20 = DoubleConv(ch * 2, ch * 4)
#         self.conv30 = DoubleConv(ch * 4, ch * 8)
#         self.conv40 = DoubleConv(ch * 8, ch * 16)

#         # Nested convs
#         self.conv01 = DoubleConv(ch + ch * 2, ch)
#         self.conv11 = DoubleConv(ch * 2 + ch * 4, ch * 2)
#         self.conv21 = DoubleConv(ch * 4 + ch * 8, ch * 4)
#         self.conv31 = DoubleConv(ch * 8 + ch * 16, ch * 8)

#         self.conv02 = DoubleConv(ch * 2 + ch, ch)
#         self.conv12 = DoubleConv(ch * 4 + ch * 2, ch * 2)
#         self.conv22 = DoubleConv(ch * 8 + ch * 4, ch * 4)

#         self.conv03 = DoubleConv(ch * 2 + ch, ch)
#         self.conv13 = DoubleConv(ch * 4 + ch * 2, ch * 2)

#         self.conv04 = DoubleConv(ch * 2 + ch, ch)

#         # 输出头
#         if deep_supervision:
#             self.final1 = nn.Conv2d(ch, out_channels, kernel_size=1)
#             self.final2 = nn.Conv2d(ch, out_channels, kernel_size=1)
#             self.final3 = nn.Conv2d(ch, out_channels, kernel_size=1)
#             self.final4 = nn.Conv2d(ch, out_channels, kernel_size=1)
#         else:
#             self.final = nn.Conv2d(ch, out_channels, kernel_size=1)

#         self.maxpool = nn.MaxPool2d(2)

#     def forward(self, x):
#         x00 = self.conv00(x)
#         x10 = self.conv10(self.maxpool(x00))
#         x20 = self.conv20(self.maxpool(x10))
#         x30 = self.conv30(self.maxpool(x20))
#         x40 = self.conv40(self.maxpool(x30))

#         x01 = self.conv01(torch.cat([x00, upsample_to(x10, x00)], dim=1))
#         x11 = self.conv11(torch.cat([x10, upsample_to(x20, x10)], dim=1))
#         x21 = self.conv21(torch.cat([x20, upsample_to(x30, x20)], dim=1))
#         x31 = self.conv31(torch.cat([x30, upsample_to(x40, x30)], dim=1))

#         x02 = self.conv02(torch.cat([x01, upsample_to(x11, x01)], dim=1))
#         x12 = self.conv12(torch.cat([x11, upsample_to(x21, x11)], dim=1))
#         x22 = self.conv22(torch.cat([x21, upsample_to(x31, x21)], dim=1))

#         x03 = self.conv03(torch.cat([x02, upsample_to(x12, x02)], dim=1))
#         x13 = self.conv13(torch.cat([x12, upsample_to(x22, x12)], dim=1))

#         x04 = self.conv04(torch.cat([x03, upsample_to(x13, x03)], dim=1))

#         if self.deep_supervision:
#             return [
#                 self.final1(x01),
#                 self.final2(x02),
#                 self.final3(x03),
#                 self.final4(x04)
#             ]
#         else:
#             return self.final(x04)
