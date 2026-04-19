# Underwater Image Reconstruction — Technical Report

**Project**: U-Net + Swin Transformer for Underwater Image Enhancement  
**Last Updated**: April 2026  
**Status**: Training-ready (dataset cleaned, pipeline optimized)

---

## 1. Problem Statement

Underwater images suffer from wavelength-dependent light attenuation, color cast (blue/green dominant), haze from suspended particles, and low contrast. This project trains a deep learning model to restore degraded underwater images to their clear-water appearance using paired supervised learning.

## 2. Architecture

### 2.1 Model: U-Net + Swin Transformer Bottleneck

```
Input (3×H×W) → Encoder → Bottleneck + Swin → Decoder → Output (3×H×W)
```

| Component | Channels | Details |
|-----------|----------|---------|
| **Encoder 1** | 3 → 64 | ConvBlock + Dropout2d(0.1) |
| **Encoder 2** | 64 → 128 | ConvBlock + Dropout2d(0.1) |
| **Encoder 3** | 128 → 256 | ConvBlock + Dropout2d(0.1) |
| **Encoder 4** | 256 → 512 | ConvBlock + Dropout2d(0.1) |
| **Bottleneck** | 512 → 1024 | ConvBlock |
| **Swin Bottleneck** | 1024 → 1024 | 2× Swin blocks (W-MSA + SW-MSA) |
| **Decoder 4–1** | 1024 → 64 | ConvTranspose2d + skip connections |
| **Output** | 64 → 3 | 1×1 Conv + Sigmoid |

**Total parameters**: ~58M

### 2.2 Swin Transformer Bottleneck

The Swin Transformer replaces a standard CNN bottleneck with window-based self-attention:

- **Block 1 (W-MSA)**: Standard window attention — captures local haze patterns within 4×4 windows
- **Block 2 (SW-MSA)**: Shifted window attention — cross-window connections for global scene understanding
- **Window size**: 4 (at 384×384 input → bottleneck is 24×24 → 36 windows of 16 tokens each)
- **Relative position bias**: Learnable per-head bias for spatial awareness

#### Alpha Gate

```python
output = cnn_features + α × swin_features   # α initialized at 0.1
```

- **Warm start at 0.1**: Ensures gradient flow through the Swin branch from epoch 1
- Previous initialization at 0.0 caused the transformer to be permanently suppressed (gradients vanish through zero gate)
- α is a learnable `nn.Parameter` — the model learns the optimal CNN/Swin balance

### 2.3 Encoder Regularization

Each encoder block is followed by `Dropout2d(p=0.1)`:
- Drops entire feature channels (not individual pixels)
- Appropriate for conv feature maps where adjacent pixels are correlated
- Automatically disabled during `model.eval()`

## 3. Loss Function

```
L_total = 0.5 × L1 + 0.3 × (1 - SSIM) + 0.2 × L_perceptual
```

| Component | Weight | Purpose |
|-----------|--------|---------|
| **L1 (pixel)** | 0.5 | Pixel-level accuracy, color fidelity |
| **1 - SSIM** | 0.3 | Structural similarity, luminance/contrast matching |
| **VGG Perceptual** | 0.2 | Feature-space sharpness via frozen VGG16 `features[:16]` |

**Previous loss** (0.8×L1 + 0.2×SSIM) was L1-heavy, producing blurry outputs. The perceptual term forces the model to preserve high-frequency texture detail.

### VGG Perceptual Loss

```python
VGG16(pretrained).features[:16]  # frozen, no gradients
L_perceptual = L1(VGG(pred), VGG(target))  # feature-space distance
```

Layers 0–15 of VGG16 capture edges, textures, and mid-level patterns — exactly what underwater degradation destroys.

## 4. Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Optimizer** | AdamW (lr=1e-4, wd=1e-4) | Decoupled weight decay for better generalization |
| **Scheduler** | ReduceLROnPlateau(mode='max', factor=0.5, patience=3) | Adapts LR based on actual val PSNR (not fixed cosine) |
| **Min LR** | 1e-6 | Floor to prevent training stall |
| **Gradient clipping** | clip_grad_norm = 1.0 | Prevents exploding gradients from Swin attention |
| **Batch size** | 2 | Reduced for 384×384 crops to fit GPU memory |
| **Image size** | 384×384 | Larger crops for better global context |
| **Max epochs** | 100 | Controlled by early stopping |
| **Early stopping** | Patience = 15 | Allows LR to drop multiple times before stopping |

### Scheduler Strategy

```
ReduceLROnPlateau watches val_psnr:
  PSNR plateaus for 3 epochs → LR × 0.5
  Can drop multiple times: 1e-4 → 5e-5 → 2.5e-5 → ...
  Early stopping at patience=15 allows 4-5 LR drops
```

Previous `CosineAnnealingLR(T_max=100)` was disconnected — early stopping at epoch 20 meant LR barely moved from its initial value.

## 5. Dataset

### 5.1 Source

Paired underwater image dataset:
- `dataset/hazy/` — degraded underwater images (input)
- `dataset/clear/` — clean reference images (ground truth)
- Filenames match 1:1

### 5.2 Cleaning Performed

| Step | Issue Found | Action | Impact |
|------|-------------|--------|--------|
| Pairing fix | 3 orphan clear images | Deleted | Prevents silent training errors |
| Blurry references | 6 images (Laplacian var < 15) | Deleted both sides | Removes noisy supervision signal |
| Semantic misalignment | 3 pairs (different scenes) | Deleted both sides | Prevents learning garbage mappings |
| Near-black input | 1 image (mean=23.8) | Deleted both sides | Removes degenerate sample |
| Overexposed | 0 found | — | Clean |
| Corrupted | 0 found | — | Clean |

**Final dataset: 3,687 verified pairs at 256×256**

### 5.3 Split

| Set | Ratio | Count |
|-----|-------|-------|
| Train | 80% | ~2,950 |
| Val | 10% | ~369 |
| Test | 10% | ~369 |

Shuffled with `np.random.seed(42)` for reproducibility. Test split saved to `weights/test_split.npy`.

## 6. Data Augmentation Pipeline

```
load → random_scale(both) → crop(both) → flip/rotate(both)
     → color_jitter(hazy only) → gaussian_noise(hazy only) → normalize
```

| Augmentation | Applied to | Parameters |
|-------------|------------|------------|
| **Random scale** | Both | 0.8–1.2× before crop |
| **Random crop** | Both | 70% of samples |
| **Center crop** | Both | 30% of samples (reduces edge bias) |
| **Horizontal flip** | Both | 50% probability |
| **Vertical flip** | Both | 70% probability |
| **90° rotation** | Both | Random k ∈ {0,1,2,3} |
| **Color jitter** | Hazy only | Brightness ±0.1, Contrast 0.9–1.1 |
| **Gaussian noise** | Hazy only | std ∈ [0.01, 0.03] |

**Design**: Geometric transforms apply identically to both images (preserving pairing). Photometric transforms apply only to the hazy input — the model should map diverse degradations to the same clean target.

### Synthetic Degradation (optional SYNTHETIC mode)

When only clear images are available, wavelength-dependent degradation is synthesized:

```python
R × uniform(0.4, 0.7)    # red — most absorbed underwater
G × uniform(0.7, 0.9)    # green — partially absorbed
B × uniform(1.0, 1.2)    # blue — boosted (dominates underwater)
+ uniform haze (0.05–0.25)
```

## 7. Evaluation

### Metrics

| Metric | Good Threshold | What It Measures |
|--------|---------------|-----------------|
| **PSNR** | > 30 dB | Pixel-level reconstruction accuracy |
| **SSIM** | > 0.85 | Structural similarity (luminance, contrast, structure) |

### Test Protocol

1. Load `weights/unet_final.pth` (best val PSNR checkpoint)
2. Run inference on held-out test split
3. Save 3-panel comparisons: hazy | enhanced | clear
4. Report per-image and average metrics

## 8. Object Detection Integration

A pre-trained YOLO model (`fish_model.pt`, 84 MB) runs downstream on enhanced images for underwater object (fish) detection. The enhancement model serves as a preprocessing stage to improve detection accuracy in degraded underwater conditions.

## 9. Project Structure

```
Underwater-Image-reconstruction/
├── dataset/
│   ├── hazy/              (3,687 images)
│   └── clear/             (3,687 images)
├── weights/
│   ├── unet_final.pth     (best checkpoint, ~223 MB)
│   ├── test_split.npy     (held-out test paths)
│   ├── training_log.csv   (per-epoch metrics)
│   └── training_curve.png (loss + PSNR visualization)
├── images/                (test input images for demo)
├── train.py               (training script)
├── test.py                (evaluation with PSNR/SSIM)
├── unet.py                (U-Net + Swin Transformer model)
├── enhance.py             (single-image inference pipeline)
├── main.py                (main entry point + YOLO integration)
├── fish_model.pt          (YOLO fish detector)
├── requirements.txt
└── README.md
```

## 10. Key Design Decisions & Rationale

| Decision | Why |
|----------|-----|
| Swin over ViT | Window attention is O(n) vs global attention O(n²) — feasible at bottleneck resolution |
| Alpha gate warm start (0.1) | Zero init kills Swin gradients; 0.1 gives immediate contribution |
| Dropout2d over Dropout | Drops whole channels, not pixels — correct for spatially-correlated conv maps |
| ReduceLROnPlateau over CosineAnnealing | Responds to actual training dynamics, not a fixed schedule |
| Perceptual loss (VGG) | L1 alone produces blurry outputs; VGG features enforce texture preservation |
| 384px crops | Larger receptive field captures global color cast and haze patterns |
| Batch size 2 | Memory-safe for 384px with 1024-channel bottleneck |
| Mixed crop strategy (70/30) | Pure random cropping creates bias toward image edges |

## 11. What to Monitor During Training

| Signal | Healthy | Problematic |
|--------|---------|-------------|
| `swin_alpha` | Positive and growing | Negative or stuck near 0 |
| `val_psnr` | Jumps after LR drops | Flat despite LR drops |
| `train_loss` | Slower descent with perceptual loss (expected) | Diverging or NaN |
| `lr` | Step-wise drops at plateaus | Never drops (patience too high) |

## 12. References

- Saleem, A., Paheding, S., Rawashdeh, N., Awad, A., & Kaur, N. (2023). "A Non-Reference Evaluation of Underwater Image Enhancement Methods Using a New Underwater Image Dataset." *IEEE Access*.
- Liu, Z., et al. (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." *ICCV 2021*.
- Johnson, J., et al. (2016). "Perceptual Losses for Real-Time Style Transfer and Super-Resolution." *ECCV 2016*.
