import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size=7, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size//2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class MixTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim)
        )

    def forward(self, x):
        B,C,H,W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x1 = self.norm1(x)
        attn_out,_ = self.attn(x1, x1, x1)
        x = x + attn_out
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)


class MiT_B3(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.stage1 = nn.Sequential(
            OverlapPatchEmbed(in_ch, 64, 7, 4),
            *[MixTransformerBlock(64, 1) for _ in range(3)]
        )
        self.stage2 = nn.Sequential(
            OverlapPatchEmbed(64, 128, 3, 2),
            *[MixTransformerBlock(128, 2) for _ in range(6)]
        )
        self.stage3 = nn.Sequential(
            OverlapPatchEmbed(128, 320, 3, 2),
            *[MixTransformerBlock(320, 5) for _ in range(18)]
        )
        self.stage4 = nn.Sequential(
            OverlapPatchEmbed(320, 512, 3, 2),
            *[MixTransformerBlock(512, 8) for _ in range(3)]
        )

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return x1, x2, x3, x4


class SegFormerHead(nn.Module):
    def __init__(self, dims=[64,128,320,512], embed_dim=256, num_classes=1):
        super().__init__()
        self.linear_c = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, embed_dim, 1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            ) for dim in dims
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim*4, embed_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, num_classes, 1)
        )

    def forward(self, feats, target_size):
        upsampled = []
        for i, feat in enumerate(feats):
            x = self.linear_c[i](feat)
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            upsampled.append(x)
        x = torch.cat(upsampled, dim=1)
        return self.fuse(x)


class SegFormer(nn.Module):
    """
    SegFormer-B3 支持 freeze_mode：
    - "none"：全部训练
    - "partial"：冻结 stage1, stage2
    - "full_backbone"：冻结 backbone
    - "all"：全部冻结（推理/蒸馏）
    """
    def __init__(self, in_channels=1, out_channels=1, freeze_mode="none"):
        super().__init__()
        self.backbone = MiT_B3(in_channels)
        self.decode_head = SegFormerHead([64,128,320,512], embed_dim=256, num_classes=out_channels)
        self.apply_freeze(freeze_mode)

    def apply_freeze(self, mode="none"):
        assert mode in ["none", "partial", "full_backbone", "all"], f"Invalid freeze_mode: {mode}"

        def freeze(m, do_freeze=True):
            for param in m.parameters():
                param.requires_grad = not do_freeze

        if mode == "none":
            freeze(self, False)
        elif mode == "partial":
            freeze(self.backbone.stage1, True)
            freeze(self.backbone.stage2, True)
            freeze(self.backbone.stage3, False)
            freeze(self.backbone.stage4, False)
            freeze(self.decode_head, False)
        elif mode == "full_backbone":
            freeze(self.backbone, True)
            freeze(self.decode_head, False)
        elif mode == "all":
            freeze(self, True)

    def forward(self, x):
        H,W = x.shape[2:]
        feats = self.backbone(x)
        out = self.decode_head(feats, (H, W))
        return out







# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange

# class OverlapPatchEmbed(nn.Module):
#     def __init__(self, in_ch, embed_dim, patch_size=7, stride=4):
#         super().__init__()
#         self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size//2)
#         self.norm = nn.LayerNorm(embed_dim)

#     def forward(self, x):
#         x = self.proj(x)
#         B, C, H, W = x.shape
#         x = rearrange(x, 'b c h w -> b (h w) c')
#         x = self.norm(x)
#         x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
#         return x

# class MixTransformerBlock(nn.Module):
#     def __init__(self, dim, num_heads=1, mlp_ratio=4.0):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, int(dim*mlp_ratio)),
#             nn.GELU(),
#             nn.Linear(int(dim*mlp_ratio), dim)
#         )

#     def forward(self, x):
#         B,C,H,W = x.shape
#         x = rearrange(x, 'b c h w -> b (h w) c')
#         x1 = self.norm1(x)
#         attn_out,_ = self.attn(x1, x1, x1)
#         x = x + attn_out
#         x2 = self.norm2(x)
#         x = x + self.mlp(x2)
#         return rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

# class MiT_B3(nn.Module):
#     def __init__(self, in_ch):
#         super().__init__()
#         self.stage1 = nn.Sequential(
#             OverlapPatchEmbed(in_ch, 64, 7, 4),
#             *[MixTransformerBlock(64, 1) for _ in range(3)]
#         )
#         self.stage2 = nn.Sequential(
#             OverlapPatchEmbed(64, 128, 3, 2),
#             *[MixTransformerBlock(128, 2) for _ in range(6)]
#         )
#         self.stage3 = nn.Sequential(
#             OverlapPatchEmbed(128, 320, 3, 2),
#             *[MixTransformerBlock(320, 5) for _ in range(18)]
#         )
#         self.stage4 = nn.Sequential(
#             OverlapPatchEmbed(320, 512, 3, 2),
#             *[MixTransformerBlock(512, 8) for _ in range(3)]
#         )

#     def forward(self, x):
#         x1 = self.stage1(x)
#         x2 = self.stage2(x1)
#         x3 = self.stage3(x2)
#         x4 = self.stage4(x3)
#         return x1, x2, x3, x4

# class SegFormerHead(nn.Module):
#     def __init__(self, dims=[64,128,320,512], embed_dim=256, num_classes=1):
#         super().__init__()
#         self.linear_c = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(dim, embed_dim, 1),
#                 nn.BatchNorm2d(embed_dim),
#                 nn.ReLU(inplace=True)
#             ) for dim in dims
#         ])
#         self.fuse = nn.Sequential(
#             nn.Conv2d(embed_dim*4, embed_dim, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(embed_dim, num_classes, 1)
#         )

#     def forward(self, feats, target_size):
#         upsampled = []
#         for i, feat in enumerate(feats):
#             x = self.linear_c[i](feat)
#             x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
#             upsampled.append(x)
#         x = torch.cat(upsampled, dim=1)
#         return self.fuse(x)

# class SegFormer(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1):
#         super().__init__()
#         self.backbone = MiT_B3(in_channels)
#         self.decode_head = SegFormerHead([64,128,320,512], embed_dim=256, num_classes=out_channels)

#     def forward(self, x):
#         H,W = x.shape[2:]
#         feats = self.backbone(x)
#         out = self.decode_head(feats, (H, W))
#         return out





#B2版本SegFormer这个东西
"""# ✅ 升级版 SegFormer-B2（更深更宽 + 拼接融合）
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size=7, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size//2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x

class MixTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim)
        )

    def forward(self, x):
        B,C,H,W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x1 = self.norm1(x)
        attn_out,_ = self.attn(x1, x1, x1)
        x = x + attn_out
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

class MiT_B2(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.stage1 = nn.Sequential(
            OverlapPatchEmbed(in_ch, 64, 7, 4),
            *[MixTransformerBlock(64, 1) for _ in range(3)]
        )
        self.stage2 = nn.Sequential(
            OverlapPatchEmbed(64, 128, 3, 2),
            *[MixTransformerBlock(128, 2) for _ in range(4)]
        )
        self.stage3 = nn.Sequential(
            OverlapPatchEmbed(128, 320, 3, 2),
            *[MixTransformerBlock(320, 5) for _ in range(6)]
        )
        self.stage4 = nn.Sequential(
            OverlapPatchEmbed(320, 512, 3, 2),
            *[MixTransformerBlock(512, 8) for _ in range(3)]
        )

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return x1, x2, x3, x4

class SegFormerHead(nn.Module):
    def __init__(self, dims=[64,128,320,512], embed_dim=256, num_classes=1):
        super().__init__()
        self.linear_c = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, embed_dim, 1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            ) for dim in dims
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim*4, embed_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, num_classes, 1)
        )

    def forward(self, feats, target_size):
        upsampled = []
        for i, feat in enumerate(feats):
            x = self.linear_c[i](feat)
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            upsampled.append(x)
        x = torch.cat(upsampled, dim=1)
        return self.fuse(x)

class SegFormer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.backbone = MiT_B2(in_channels)
        self.decode_head = SegFormerHead([64,128,320,512], embed_dim=256, num_classes=out_channels)

    def forward(self, x):
        H,W = x.shape[2:]
        feats = self.backbone(x)
        out = self.decode_head(feats, (H, W))
        return out
"""