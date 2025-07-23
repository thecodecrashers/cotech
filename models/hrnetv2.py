import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return self.relu(out + identity)


class HRModule(nn.Module):
    def __init__(self, num_branches, num_blocks, num_channels):
        super(HRModule, self).__init__()
        self.num_branches = num_branches
        self.blocks = nn.ModuleList()
        for i in range(num_branches):
            branch = []
            for _ in range(num_blocks):
                branch.append(BasicBlock(num_channels[i], num_channels[i]))
            self.blocks.append(nn.Sequential(*branch))

        self.fuse_layers = nn.ModuleList()
        for i in range(num_branches):
            fuse_layer = nn.ModuleList()
            for j in range(num_branches):
                if i == j:
                    fuse_layer.append(nn.Identity())
                elif i < j:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_channels[j], num_channels[i], 1, bias=False),
                        nn.BatchNorm2d(num_channels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='bilinear', align_corners=True)
                    ))
                else:  # i > j
                    ops = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            ops.append(nn.Conv2d(num_channels[j], num_channels[i], 3, 2, 1, bias=False))
                            ops.append(nn.BatchNorm2d(num_channels[i]))
                        else:
                            ops.append(nn.Conv2d(num_channels[j], num_channels[j], 3, 2, 1, bias=False))
                            ops.append(nn.BatchNorm2d(num_channels[j]))
                            ops.append(nn.ReLU(inplace=True))
                    fuse_layer.append(nn.Sequential(*ops))
            self.fuse_layers.append(fuse_layer)

    def forward(self, x):
        x = [branch(xi) for branch, xi in zip(self.blocks, x)]

        fused = []
        for i in range(self.num_branches):
            y = self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                y += self.fuse_layers[i][j](x[j])
            fused.append(F.relu(y))
        return fused


class HRNetV2(nn.Module):
    def __init__(self, num_classes=1):
        super(HRNetV2, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 1
        self.layer1 = nn.Sequential(*[BasicBlock(64, 64) for _ in range(4)])

        # Transition to stage 2
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 32, 3, 1, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 3, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        ])

        self.stage2 = HRModule(num_branches=2, num_blocks=2, num_channels=[32, 64])

        # Fuse and output head
        self.last_layer = nn.Sequential(
            nn.Conv2d(32 + 64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)

        x_list = [trans(x) for trans in self.transition1]
        x_list = self.stage2(x_list)

        # 上采样所有分支并拼接
        x0 = x_list[0]
        x1 = F.interpolate(x_list[1], size=x0.shape[2:], mode='bilinear', align_corners=True)
        out = torch.cat([x0, x1], dim=1)

        out = self.last_layer(out)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)  # 恢复原图分辨率
        return torch.sigmoid(out)  # 如果是多类，用 softmax
