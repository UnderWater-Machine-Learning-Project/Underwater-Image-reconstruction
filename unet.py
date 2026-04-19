"""
U-Net with Swin Transformer bottleneck — underwater image enhancement.

Structure:
    Encoder     4 levels:  3 → 64 → 128 → 256 → 512
    Bottleneck  + Swin:    512 → 1024 → Swin blocks → 1024
    Decoder     4 levels:  1024 → 512 → 256 → 128 → 64 → 3

Input  : 3 × H × W  float tensor, values 0-1  (RGB)
Output : 3 × H × W  float tensor, values 0-1  (enhanced RGB)
H and W must be multiples of 16 — enhance.py pads automatically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Basic building block ───────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ── Swin Transformer helpers ───────────────────────────────────────────────────

def window_partition(x, window_size):
    """(B, H, W, C) → (num_windows*B, window_size, window_size, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    """(num_windows*B, window_size, window_size, C) → (B, H, W, C)"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with relative position bias.
    Used for both W-MSA (no shift) and SW-MSA (with cyclic shift).
    Relative position bias lets the model know WHERE tokens are inside
    each window — critical for spatial understanding of haze patterns.
    """
    def __init__(self, dim, window_size, num_heads, dropout=0.0):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size   # (Wh, Ww)
        self.num_heads   = num_heads
        self.scale       = (dim // num_heads) ** -0.5

        # Relative position bias table
        self.rel_pos_bias = nn.Parameter(
            torch.zeros((2*window_size-1) * (2*window_size-1), num_heads))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        # Build relative position index
        coords   = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing='ij'))
        flat     = torch.flatten(coords, 1)
        rel      = flat[:, :, None] - flat[:, None, :]
        rel      = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer('rel_pos_index', rel.sum(-1))

        self.qkv      = nn.Linear(dim, dim * 3, bias=True)
        self.proj     = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = (self.qkv(x)
                   .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
                   .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        # Add relative position bias
        bias = self.rel_pos_bias[self.rel_pos_index.view(-1)]
        bias = bias.view(N, N, self.num_heads).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        # Apply shifted-window mask if provided
        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x    = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinBlock(nn.Module):
    """
    One Swin Transformer block.
    shift_size=0        → regular window attention (W-MSA)
    shift_size=win//2   → shifted window attention (SW-MSA)

    Shifted windows are what make Swin powerful: the shift connects
    adjacent windows so information flows across the whole feature map,
    not just within isolated squares. This is how it captures global
    haze while remaining computationally cheap.
    """
    def __init__(self, dim, num_heads, window_size=4, shift_size=0,
                 mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.shift_size  = shift_size
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(dim, window_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
        self.attn_mask = None   # computed once on first forward

    def _make_mask(self, H, W, device):
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask    = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x):
        B, H, W, C = x.shape
        res = x

        x = self.norm1(x)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if self.attn_mask is None or self.attn_mask.shape[0] != (H // self.window_size) * (W // self.window_size):
                self.attn_mask = self._make_mask(H, W, x.device)
        else:
            self.attn_mask = None

        # Window partition → attention → reverse
        wins  = window_partition(x, self.window_size)
        wins  = wins.view(-1, self.window_size * self.window_size, C)
        wins  = self.attn(wins, mask=self.attn_mask)
        wins  = wins.view(-1, self.window_size, self.window_size, C)
        x     = window_reverse(wins, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = res + x
        x = x + self.ffn(self.norm2(x))
        return x


class SwinBottleneck(nn.Module):
    """
    Two Swin blocks (W-MSA + SW-MSA) replacing the plain ViT at the U-Net bottleneck.

    Why two blocks:
        Block 1 (W-MSA):  attends within non-overlapping windows — local haze patterns
        Block 2 (SW-MSA): shifts windows by half — cross-window connections — global scene

    Alpha gate (starts at 0):
        output = CNN_features + alpha × Swin_features
        Safe initialization: model trains as plain U-Net until Swin learns to contribute.

    window_size=4 at bottleneck:
        For 256×256 input, bottleneck is 16×16.
        window_size=4 → 4×4 = 16 tokens per window → lightweight attention.
    """
    def __init__(self, channels, num_heads=8, window_size=4, dropout=0.1):
        super().__init__()

        # Project CNN channels to Swin dim and back
        self.in_proj  = nn.Linear(channels, channels)
        self.block_w  = SwinBlock(channels, num_heads, window_size,
                                  shift_size=0,            dropout=dropout)
        self.block_sw = SwinBlock(channels, num_heads, window_size,
                                  shift_size=window_size//2, dropout=dropout)
        self.out_proj = nn.Linear(channels, channels)
        self.alpha    = nn.Parameter(torch.zeros(1))   # off at init

    def forward(self, x):
        B, C, H, W = x.shape

        # CNN → (B, H, W, C) for Swin
        out = x.permute(0, 2, 3, 1).contiguous()
        out = self.in_proj(out)
        out = self.block_w(out)    # W-MSA
        out = self.block_sw(out)   # SW-MSA (shifted)
        out = self.out_proj(out)

        # Back to (B, C, H, W)
        out = out.permute(0, 3, 1, 2).contiguous()

        return x + self.alpha * out   # alpha gate


# ── Full U-Net + Swin ──────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    U-Net with Swin Transformer bottleneck for underwater image enhancement.
    base=64: encoder 3→64→128→256→512, bottleneck 1024.
    """
    def __init__(self, in_ch=3, out_ch=3, base=64):
        super().__init__()
        b = base

        # Encoder
        self.enc1 = _ConvBlock(in_ch, b)
        self.enc2 = _ConvBlock(b,     b*2)
        self.enc3 = _ConvBlock(b*2,   b*4)
        self.enc4 = _ConvBlock(b*4,   b*8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck + Swin
        self.bottleneck = _ConvBlock(b*8, b*16)
        self.swin = SwinBottleneck(
            channels    = b*16,   # 1024
            num_heads   = 8,
            window_size = 4,
            dropout     = 0.1,
        )

        # Decoder
        self.up4  = nn.ConvTranspose2d(b*16, b*8, 2, stride=2)
        self.dec4 = _ConvBlock(b*16, b*8)

        self.up3  = nn.ConvTranspose2d(b*8, b*4, 2, stride=2)
        self.dec3 = _ConvBlock(b*8,  b*4)

        self.up2  = nn.ConvTranspose2d(b*4, b*2, 2, stride=2)
        self.dec2 = _ConvBlock(b*4,  b*2)

        self.up1  = nn.ConvTranspose2d(b*2, b,   2, stride=2)
        self.dec1 = _ConvBlock(b*2,  b)

        self.final   = nn.Conv2d(b, out_ch, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b  = self.bottleneck(self.pool(e4))
        b  = self.swin(b)                         # ← Swin here

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.sigmoid(self.final(d1))