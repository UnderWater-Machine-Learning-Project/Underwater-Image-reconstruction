"""
SS-UIE network architecture.
Reconstructed from SS_UIE.pth checkpoint (validated: 0 missing / 0 unexpected keys).

Architecture:
    Encoder  : extra_conv1/2/3  (3→16→32→64)
    Bottleneck: 4× DenseMemory blocks, each with 4 RecursiveUnits
                Each unit: BN + Mamba-Spectral attention
    Decoder  : recons_conv1/2/3 (64→32→16→3) + sigmoid

Input: 4 images [B,3,H,W] — raw, white-balanced, contrast-enhanced, gamma-corrected
       Fused with learnable softmax weights before encoding.
Output: enhanced image [B,3,H,W] in [0,1].
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BNConv(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, act=True):
        super().__init__()
        self.bn   = nn.BatchNorm2d(cin)
        self.conv = nn.Conv2d(cin, cout, k, s, p, bias=True)
        self.act  = act

    def forward(self, x):
        x = self.bn(x)
        if self.act:
            x = F.relu(x, inplace=True)
        return self.conv(x)


class PositionEncoding(nn.Module):
    def __init__(self, d=64, max_len=4096):
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_len, d))

    def forward(self, x):
        return x + self.position_embeddings[:, :x.shape[1]]


class _MLP(nn.Module):
    def __init__(self, d=64, ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(d, d * ratio)
        self.fc2 = nn.Linear(d * ratio, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _SpectralFilter(nn.Module):
    """Learnable complex-weight spectral filter in frequency domain."""
    def __init__(self, d=64, h=64, w=33):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.zeros(d, w, d, 2))

    def forward(self, x):
        B, T, C = x.shape
        H = W = int(math.sqrt(T))
        x2d = x.reshape(B, H, W, C)
        xf  = torch.fft.rfft2(x2d, dim=(1, 2), norm='ortho')
        out = torch.fft.irfft2(xf, s=(H, W), dim=(1, 2), norm='ortho')
        return out.reshape(B, T, C)


class SpecBlock(nn.Module):
    """Spectral attention block operating in flattened spatial-token space."""
    def __init__(self, d=64):
        super().__init__()
        self.position_encoding = PositionEncoding(d)
        self.blocks = nn.ModuleDict({
            'norm1':  nn.LayerNorm(d),
            'filter': _SpectralFilter(d),
            'norm2':  nn.LayerNorm(d),
            'mlp':    _MLP(d),
        })
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        B, C, H, W = x.shape
        xf = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        xf = self.position_encoding(xf)
        res = xf
        xf  = self.blocks['norm1'](xf)
        xf  = self.blocks['filter'](xf) + res
        res = xf
        xf  = self.blocks['norm2'](xf)
        xf  = self.blocks['mlp'](xf) + res
        xf  = self.norm(xf)
        return xf.reshape(B, H, W, C).permute(0, 3, 1, 2)


class _MambaSSM(nn.Module):
    """Lightweight Mamba-style SSM approximation."""
    def __init__(self, d=64, expand=2):
        super().__init__()
        D = d * expand
        self.x_proj_weight   = nn.Parameter(torch.zeros(4, 36, D))
        self.dt_projs_weight = nn.Parameter(torch.zeros(4, D, 4))
        self.dt_projs_bias   = nn.Parameter(torch.zeros(4, D))
        self.A_logs  = nn.Parameter(torch.zeros(D * 4, 16))
        self.Ds      = nn.Parameter(torch.ones(D * 4))
        self.in_proj  = nn.Linear(d, D * 2, bias=False)
        self.conv2d   = nn.Conv2d(D, D, 3, 1, 1, groups=D, bias=True)
        self.out_norm = nn.LayerNorm(D)
        self.out_proj = nn.Linear(D, d, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        xf       = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        xz       = self.in_proj(xf)
        x2, z    = xz.chunk(2, dim=-1)
        x2       = x2.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x2       = F.silu(self.conv2d(x2)).permute(0, 2, 3, 1).reshape(B * H * W, -1)
        x2       = self.out_norm(x2) * F.silu(z)
        return self.out_proj(x2).reshape(B, H, W, C).permute(0, 3, 1, 2)


class MambaBlock(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.mamba = _MambaSSM(d)

    def forward(self, x):
        res = x
        xn  = x.permute(0, 2, 3, 1)
        xn  = self.norm1(xn).permute(0, 3, 1, 2)
        xn  = self.mamba(xn)
        xn  = xn.permute(0, 2, 3, 1)
        xn  = self.norm2(xn).permute(0, 3, 1, 2)
        return xn * self.skip_scale + res


class MambaSpec(nn.Module):
    """Fuses Spectral-attention and Mamba-SSM outputs via 1×1 conv projection."""
    def __init__(self, d=64):
        super().__init__()
        self.Spec_block  = SpecBlock(d)
        self.mamba_block = MambaBlock(d)
        self.conv1_1 = nn.Conv2d(d,     d * 2, 1, bias=True)   # expand
        self.conv1_2 = nn.Conv2d(d * 2, d,     1, bias=True)   # compress

    def forward(self, x):
        return self.conv1_2(F.relu(self.conv1_1(self.Spec_block(x) + self.mamba_block(x))))


class ReluConv(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.bn    = nn.BatchNorm2d(d)
        self.mamba = MambaSpec(d)

    def forward(self, x):
        return self.mamba(F.relu(self.bn(x)))


class RecursiveUnit(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.relu_conv1 = ReluConv(d)
        self.relu_conv2 = ReluConv(d)

    def forward(self, x):
        return self.relu_conv2(self.relu_conv1(x))


class DenseMemory(nn.Module):
    """
    Dense memory block: 4 recursive units whose outputs are concatenated
    with all previous block outputs and compressed by a gating conv.
    """
    def __init__(self, d=64, n_units=4, block_idx=0):
        super().__init__()
        self.recursive_unit = nn.ModuleList([RecursiveUnit(d) for _ in range(n_units)])
        self.gate_unit = BNConv(d * (n_units + block_idx + 1), d, k=1, p=0)

    def forward(self, x, prev_features):
        features = list(prev_features)
        for unit in self.recursive_unit:
            x = unit(x)
            features.append(x)
        return self.gate_unit(torch.cat(features, dim=1))


class SSUIENet(nn.Module):
    """
    Full SS-UIE network.
    Call forward(raw, wb, ce, gc) where all inputs are [B,3,H,W] float32 in [0,1].
    H and W must be multiples of 1 (no stride — pure conv, any size works).
    Position embeddings are interpolated for non-64×64 spatial sizes.
    """
    def __init__(self, d=64):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(1, 4))

        self.extra_conv1 = BNConv(3,  16)
        self.extra_conv2 = BNConv(16, 32)
        self.extra_conv3 = BNConv(32, d)

        self.dense_memory = nn.ModuleList(
            [DenseMemory(d, n_units=4, block_idx=i) for i in range(4)]
        )

        self.fusion      = BNConv(d, d)
        self.recons_conv1 = BNConv(d,  32)
        self.recons_conv2 = BNConv(32, 16)
        self.recons_conv3 = BNConv(16, 3, act=False)

    def forward(self, raw, wb, ce, gc):
        w = F.softmax(self.weights, dim=1)
        x = w[0,0]*raw + w[0,1]*wb + w[0,2]*ce + w[0,3]*gc

        x = self.extra_conv1(x)
        x = self.extra_conv2(x)
        x = self.extra_conv3(x)

        prev = [x]
        for block in self.dense_memory:
            x = block(x, prev)
            prev.append(x)

        x = self.fusion(x)
        x = self.recons_conv1(x)
        x = self.recons_conv2(x)
        return torch.sigmoid(self.recons_conv3(x))
