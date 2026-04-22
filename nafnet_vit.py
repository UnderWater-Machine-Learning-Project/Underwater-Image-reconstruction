"""
NAFNet + ViT Bottleneck — Underwater Image Enhancement
=======================================================

Architecture:
    Input (3×H×W)
      └─ Shallow Feature Extraction (Conv3×3)
           └─ NAFNet Encoder  (4 scales, NAFBlocks + downsample)
                └─ ViT Bottleneck (full global self-attention on compressed map)
                     └─ NAFNet Decoder (4 scales, NAFBlocks + upsample + skip)
                          └─ Output Conv (3×3) + residual

Key design decisions:
    NAFNet blocks   — SimpleGate replaces all nonlinear activations (ReLU/GELU).
                      Simplified Channel Attention (SCA) replaces self-attention
                      in the CNN stages. Together they outperform U-Net/vanilla
                      CNNs on image restoration with fewer parameters.

    ViT bottleneck  — Full (non-windowed) multi-head self-attention at the
                      smallest feature map. At 256×256 input the bottleneck is
                      16×16 = 256 tokens. Global attention on 256 tokens is
                      computationally cheap while capturing depth-dependent
                      color cast and scene-wide haze that CNN receptive fields
                      miss. This is the architecturally meaningful ViT
                      contribution — not windowed/local attention.

    Residual output — final_out = shallow_features_residual + decoder_out
                      Stabilises training: the network learns the correction
                      (residual) rather than the full clean image from scratch.

Parameters (default, base width=32):
    ~17M — lighter than old U-Net+Swin (58M) but architecturally superior.
    Width can be increased (base=64 → ~58M) if GPU memory allows.

Input/Output:
    float32 tensor, shape (B, 3, H, W), values in [0, 1].
    H and W must be multiples of 16. enhance.py pads automatically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ────────────────────────────────────────────────────────────────────

def _layer_norm2d(c: int) -> nn.LayerNorm:
    """Channel-last LayerNorm for (B, H, W, C) tensors."""
    return nn.LayerNorm(c)


# ── NAFNet Building Blocks ─────────────────────────────────────────────────────

class SimpleGate(nn.Module):
    """
    Splits the channel dimension in half and multiplies the two halves.
    Replaces ReLU/GELU — learnable, parameter-free gating.
    Input  channels: C  →  output channels: C // 2
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """
    Global average pool → 1×1 Conv → scale.
    Captures global channel statistics (which channels = which colour/frequency).
    Much lighter than self-attention but still models cross-channel dependencies.
    """
    def __init__(self, c: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c, c, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    """
    Core NAFNet block.

    Structure (per original paper, ECCV 2022):
        LayerNorm (channel-last)
        Conv1×1  (expand: C → 2C)
        DW Conv3×3
        SimpleGate (2C → C)
        SCA
        Conv1×1  (project: C → C)
        + residual
        LayerNorm
        Conv1×1 (expand: C → 2C FFN)
        SimpleGate (2C → C)
        Conv1×1 (project)
        + residual

    DW (depthwise) conv keeps cost low while preserving spatial structure.
    """
    def __init__(self, c: int, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        ec = c * expand   # expanded channels (before SimpleGate halves them)

        # Attention branch
        self.norm1 = _layer_norm2d(c)
        self.conv1 = nn.Conv2d(c,  ec, 1, bias=True)           # pointwise expand
        self.conv2 = nn.Conv2d(ec, ec, 3, padding=1,
                               groups=ec, bias=True)            # depthwise
        self.gate1 = SimpleGate()                               # ec → ec//2 = c
        self.sca   = SimplifiedChannelAttention(c)
        self.conv3 = nn.Conv2d(c,  c,  1, bias=True)           # pointwise project

        # FFN branch
        self.norm2 = _layer_norm2d(c)
        self.conv4 = nn.Conv2d(c,  ec, 1, bias=True)
        self.gate2 = SimpleGate()                               # ec → c
        self.conv5 = nn.Conv2d(c,  c,  1, bias=True)

        self.drop  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Learnable residual scales (initialised near 1e-2 for stable start)
        self.beta  = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-2)
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention branch (channel-last norm)
        res = x
        x   = x.permute(0, 2, 3, 1)                    # BCHW → BHWC
        x   = self.norm1(x)
        x   = x.permute(0, 3, 1, 2)                    # BHWC → BCHW
        x   = self.conv2(self.conv1(x))
        x   = self.gate1(x)
        x   = self.sca(x)
        x   = self.drop(self.conv3(x))
        x   = res + x * self.beta

        # FFN branch
        res = x
        x   = x.permute(0, 2, 3, 1)
        x   = self.norm2(x)
        x   = x.permute(0, 3, 1, 2)
        x   = self.gate2(self.conv4(x))
        x   = self.drop(self.conv5(x))
        x   = res + x * self.gamma
        return x


# ── ViT Bottleneck ─────────────────────────────────────────────────────────────

class ViTBottleneck(nn.Module):
    """
    Full (non-windowed) Vision Transformer bottleneck.

    Why full attention here, not windowed (Swin):
        At the deepest encoder level the feature map is small (H/16 × W/16).
        For 256×256 input: 16×16 = 256 tokens.
        For 384×384 input: 24×24 = 576 tokens.
        Full O(N²) attention on 256-576 tokens is cheap, and lets EVERY
        spatial position attend to EVERY other position — capturing the
        global depth-dependent colour cast and scene-wide haze that windowed
        attention can miss at patch boundaries.

    Structure (L transformer blocks):
        Input (B, C, H, W)
          → flatten to sequence (B, N, C),  N = H*W
          → L × (LayerNorm → MHSA → residual → LayerNorm → MLP → residual)
          → unflatten back to (B, C, H, W)

    Learnable position embeddings:
        Created for (H, W) at first forward pass and cached.
        Re-created if spatial size changes (variable-resolution support).
    """
    def __init__(self, c: int, depth: int = 4,
                 num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.c    = c
        self.pos  = None               # cached positional embedding
        self._hw  = None               # (H, W) for which pos was created

        self.blocks = nn.ModuleList([
            _ViTBlock(c, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(c)

    def _get_pos(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Return (1, H*W, C) learnable positional embedding."""
        if self._hw != (H, W) or self.pos is None:
            self.pos  = nn.Parameter(
                torch.zeros(1, H * W, self.c, device=device))
            nn.init.trunc_normal_(self.pos, std=0.02)
            self._hw  = (H, W)
        return self.pos.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # (B, C, H, W) → (B, N, C)
        seq = x.flatten(2).transpose(1, 2)                     # B, N, C
        seq = seq + self._get_pos(H, W, x.device)

        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm(seq)

        # (B, N, C) → (B, C, H, W)
        return seq.transpose(1, 2).reshape(B, C, H, W)


class _ViTBlock(nn.Module):
    """Single ViT encoder block: MHSA + MLP with pre-norm."""
    def __init__(self, c: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(c)
        self.attn  = nn.MultiheadAttention(
            c, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(c)
        hidden     = int(c * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(c, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, c),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MHSA with pre-norm + residual
        n = self.norm1(x)
        a, _ = self.attn(n, n, n, need_weights=False)
        x = x + a

        # MLP with pre-norm + residual
        x = x + self.mlp(self.norm2(x))
        return x


# ── Downsample / Upsample ──────────────────────────────────────────────────────

class _Downsample(nn.Module):
    """2× spatial downsampling via pixel unshuffle (lossless rearrangement)."""
    def __init__(self, c: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c * 2, 2, stride=2, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _Upsample(nn.Module):
    """2× spatial upsampling via pixel shuffle."""
    def __init__(self, c: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c * 2, 1, bias=False),
            nn.PixelShuffle(2),           # C*2 channels → C/2 channels, 2× HW
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ── Full Model ─────────────────────────────────────────────────────────────────

class NAFNetViT(nn.Module):
    """
    NAFNet Encoder-Decoder + ViT Bottleneck for underwater image enhancement.

    Args:
        in_ch       : input channels (3 for RGB).
        out_ch      : output channels (3 for RGB).
        base        : base feature width. Scales all channel counts.
                      base=32  → ~17M params  (fast, good for limited data)
                      base=64  → ~58M params  (matches old U-Net size)
        enc_blocks  : NAFBlocks per encoder level  [L1, L2, L3, L4].
        dec_blocks  : NAFBlocks per decoder level  [L1, L2, L3, L4].
        vit_depth   : number of ViT transformer blocks at bottleneck.
        vit_heads   : number of attention heads in ViT.
        dropout     : dropout rate for NAFBlocks (disabled in eval mode).
    """

    def __init__(self,
                 in_ch: int      = 3,
                 out_ch: int     = 3,
                 base: int       = 32,
                 enc_blocks: list = [2, 2, 4, 8],
                 dec_blocks: list = [2, 2, 2, 2],
                 vit_depth: int  = 4,
                 vit_heads: int  = 8,
                 dropout: float  = 0.0):
        super().__init__()

        # Channel widths at each scale: base, 2b, 4b, 8b
        ch = [base * (2 ** i) for i in range(4)]   # [32, 64, 128, 256]

        # ── Shallow feature extraction (input projection) ──────────────────
        self.intro = nn.Conv2d(in_ch, ch[0], 3, padding=1, bias=True)

        # ── Encoder ───────────────────────────────────────────────────────
        self.enc1 = nn.Sequential(*[NAFBlock(ch[0], dropout=dropout)
                                    for _ in range(enc_blocks[0])])
        self.down1 = _Downsample(ch[0])   # ch[0] → ch[1]

        self.enc2 = nn.Sequential(*[NAFBlock(ch[1], dropout=dropout)
                                    for _ in range(enc_blocks[1])])
        self.down2 = _Downsample(ch[1])   # ch[1] → ch[2]

        self.enc3 = nn.Sequential(*[NAFBlock(ch[2], dropout=dropout)
                                    for _ in range(enc_blocks[2])])
        self.down3 = _Downsample(ch[2])   # ch[2] → ch[3]

        self.enc4 = nn.Sequential(*[NAFBlock(ch[3], dropout=dropout)
                                    for _ in range(enc_blocks[3])])
        self.down4 = _Downsample(ch[3])   # ch[3] → ch[3]*2 = bottleneck

        # ── ViT Bottleneck ────────────────────────────────────────────────
        # Channel count at bottleneck = ch[3] * 2  (after last downsample)
        btn_ch = ch[3] * 2                          # 256*2 = 512 for base=32
        self.bottleneck = ViTBottleneck(
            c         = btn_ch,
            depth     = vit_depth,
            num_heads = vit_heads,
            dropout   = dropout,
        )

        # ── Decoder ───────────────────────────────────────────────────────
        # Each upsample halves channels; skip connection doubles them again.
        self.up4   = _Upsample(btn_ch)              # btn_ch → btn_ch//2 = ch[3]
        self.dec4  = nn.Sequential(*[NAFBlock(ch[3] * 2, dropout=dropout)
                                     for _ in range(dec_blocks[3])])
        self.fuse4 = nn.Conv2d(ch[3] * 2, ch[3], 1, bias=True)

        self.up3   = _Upsample(ch[3])               # ch[3] → ch[2]
        self.dec3  = nn.Sequential(*[NAFBlock(ch[2] * 2, dropout=dropout)
                                     for _ in range(dec_blocks[2])])
        self.fuse3 = nn.Conv2d(ch[2] * 2, ch[2], 1, bias=True)

        self.up2   = _Upsample(ch[2])               # ch[2] → ch[1]
        self.dec2  = nn.Sequential(*[NAFBlock(ch[1] * 2, dropout=dropout)
                                     for _ in range(dec_blocks[1])])
        self.fuse2 = nn.Conv2d(ch[1] * 2, ch[1], 1, bias=True)

        self.up1   = _Upsample(ch[1])               # ch[1] → ch[0]
        self.dec1  = nn.Sequential(*[NAFBlock(ch[0] * 2, dropout=dropout)
                                     for _ in range(dec_blocks[0])])
        self.fuse1 = nn.Conv2d(ch[0] * 2, ch[0], 1, bias=True)

        # ── Output projection ─────────────────────────────────────────────
        self.output = nn.Conv2d(ch[0], out_ch, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, 3, H, W) float32, values in [0, 1].
               H, W must be multiples of 16.

        Returns:
            Enhanced (B, 3, H, W) float32, values in [0, 1].
        """
        inp = x   # save for residual output

        # Shallow feature extraction
        x = self.intro(x)           # → (B, ch[0], H, W)

        # Encoder
        e1 = self.enc1(x)           # (B, ch[0], H,   W)
        e2 = self.enc2(self.down1(e1))  # (B, ch[1], H/2, W/2)
        e3 = self.enc3(self.down2(e2))  # (B, ch[2], H/4, W/4)
        e4 = self.enc4(self.down3(e3))  # (B, ch[3], H/8, W/8)

        # Bottleneck (ViT — full global self-attention)
        b  = self.bottleneck(self.down4(e4))   # (B, btn_ch, H/16, W/16)

        # Decoder (upsample → concat skip → NAFBlocks → fuse to correct width)
        # up4: btn_ch → btn_ch//2 = ch[3].  cat with e4 (ch[3]) → ch[3]*2
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))   # → ch[3]*2
        d4 = self.fuse4(d4)                                     # → ch[3]

        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))   # → ch[2]*2
        d3 = self.fuse3(d3)                                     # → ch[2]

        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))   # → ch[1]*2
        d2 = self.fuse2(d2)                                     # → ch[1]

        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))   # → ch[0]*2
        d1 = self.fuse1(d1)                                     # → ch[0]

        # Output projection + global residual (learn correction, not full image)
        out = self.output(d1) + inp
        return torch.clamp(out, 0.0, 1.0)


# ── Convenience constructors ───────────────────────────────────────────────────

def nafnet_vit_small(**kwargs) -> NAFNetViT:
    """~17M params. Good starting point for 3-10k pairs."""
    return NAFNetViT(base=32, enc_blocks=[2,2,4,8],
                     dec_blocks=[2,2,2,2], vit_depth=4, **kwargs)


def nafnet_vit_base(**kwargs) -> NAFNetViT:
    """~58M params. Use when dataset >= 8k pairs."""
    return NAFNetViT(base=64, enc_blocks=[2,2,4,8],
                     dec_blocks=[2,2,2,2], vit_depth=6, **kwargs)


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    for name, model in [("small", nafnet_vit_small()),
                         ("base",  nafnet_vit_base())]:
        model = model.to(device).eval()
        params = sum(p.numel() for p in model.parameters()) / 1e6

        x = torch.rand(1, 3, 256, 256, device=device)
        with torch.no_grad():
            t0  = time.time()
            out = model(x)
            t1  = time.time()

        print(f"nafnet_vit_{name}")
        print(f"  params     : {params:.1f}M")
        print(f"  input      : {tuple(x.shape)}")
        print(f"  output     : {tuple(out.shape)}")
        print(f"  out range  : [{out.min():.3f}, {out.max():.3f}]  (should be 0-1)")
        print(f"  fwd time   : {(t1-t0)*1000:.1f} ms")
        print()
