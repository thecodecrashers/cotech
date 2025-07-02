import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# === OverlapPatchEmbed ===
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim=32, patch_size=7, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride, patch_size//2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B,C,H,W = x.shape
        x = self.proj(x)
        _, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x

# === Transformer Encoder Block ===
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

# === MiT-B0 Backbone ===
class MiT_B0(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.stage1 = OverlapPatchEmbed(in_ch, 32, 7, 4)
        self.block1 = nn.Sequential(*[MixTransformerBlock(32, num_heads=1) for _ in range(2)])
        # [可扩展更多stage，MiT-B0 默认4stage略简略]

    def forward(self, x):
        x = self.stage1(x)
        x = self.block1(x)
        return x

# === Decode Head ===
class SegFormerHead(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.linear = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(self, x, target_size):
        x = self.linear(x)
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

# === 完整 SegFormer-B0 ===
class SegFormer(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.backbone = MiT_B0(in_channels)
        self.decode_head = SegFormerHead(32, out_channels)

    def forward(self, x):
        H,W = x.shape[2:]
        x = self.backbone(x)
        x = self.decode_head(x, (H, W))
        return x
