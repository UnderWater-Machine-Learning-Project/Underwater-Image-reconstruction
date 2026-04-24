"""
Training script for NAFNet + ViT Bottleneck underwater image enhancement.

Dataset structure expected:
    dataset/
    ├── hazy/     ← degraded underwater images (your input)
    └── clear/    ← clean reference images    (ground truth)

    Filenames must match: dataset/hazy/001.jpg <-> dataset/clear/001.jpg

Split: 80% train / 10% val / 10% test
Early stopping: halts training if val PSNR does not improve for PATIENCE epochs.
Outputs:
    weights/nafnet_final.pth     ← best checkpoint (loaded by main.py + test.py)
    weights/test_split.npy       ← held-out test paths (used by test.py)
    weights/training_log.csv     ← per-epoch metrics
    weights/training_curve.png   ← loss + PSNR elbow curve

Usage:
    python train.py
"""

import os
import csv
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from nafnet_vit import nafnet_vit_small, nafnet_vit_base
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────

DATASET_DIR    = "dataset"
WEIGHTS_DIR    = "weights"
HAZY_DIR       = os.path.join(DATASET_DIR, "hazy")
CLEAR_DIR      = os.path.join(DATASET_DIR, "clear")
PREPROCESS_DIR = os.path.join(DATASET_DIR, "preprocessed")

EPOCHS       = 200     # cosine annealing needs full schedule to converge
BATCH_SIZE   = 8       # nafnet_vit_small (~17M) — batch=8 safe on RTX 5070 8GB
LR           = 1e-3    # NAFNet trains better at higher LR
IMG_SIZE     = 256     # NAFNet paper trains at 256
SAVE_EVERY   = 10      # save checkpoint every N epochs
PATIENCE     = 50    # effectively disabled — let cosine LR run the full schedule
WARMUP_EPOCHS = 5      # linear LR warmup: prevents loss spike at LR=1e-3

TRAIN_SPLIT  = 0.80
VAL_SPLIT    = 0.10
TEST_SPLIT   = 0.10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Performance flags ─────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True   # auto-tune conv kernels for fixed 256×256 input


# ── Dataset ────────────────────────────────────────────────────────────────────

class UnderwaterDataset(Dataset):
    SYNTHETIC = False   # set True if you only have clear images, no hazy pairs

    def __init__(self, hazy_paths, clear_paths, img_size=256, augment=True, full_image=False,
                 preproc_paths=None, preproc_prob=0.5):
        """
        Mixed-input dataset: each sample randomly draws from hazy OR preprocessed.

        preproc_paths : parallel list of preprocessed paths (1:1 with clear_paths).
                        If None, only hazy inputs are used (original behaviour).
        preproc_prob  : probability of choosing preprocessed over hazy each step.
                        0.0 = hazy only,  1.0 = preprocessed only.

        Why mixed? enhance.py always preprocesses at inference, so production inputs
        are mostly preprocessed. But edge-case preprocessing failures produce inputs
        closer to raw hazy. Mixed training makes one model robust to both without
        needing two separate models.

        Val always uses preprocessed (stable, interpretable PSNR).
        """
        self.hazy_pairs    = list(zip(hazy_paths, clear_paths))
        self.preproc_pairs = list(zip(preproc_paths, clear_paths)) if preproc_paths else None
        self.preproc_prob  = preproc_prob if preproc_paths else 0.0
        self.img_size      = img_size
        self.augment       = augment
        self.full_image    = full_image

    def __len__(self):
        return len(self.hazy_pairs)

    def _degrade(self, img):
        """Simulate underwater physics: wavelength-dependent attenuation + haze."""
        img = img.astype(np.float32) / 255.0
        img[:, :, 0] *= np.random.uniform(0.4, 0.7)   # R — most absorbed
        img[:, :, 1] *= np.random.uniform(0.7, 0.9)   # G — partially absorbed
        img[:, :, 2]  = np.clip(img[:, :, 2] * np.random.uniform(1.0, 1.2) + 0.05, 0, 1)  # B — boosted
        haze = np.random.uniform(0.05, 0.25)
        img  = img * (1 - haze) + haze
        return np.clip(img * 255, 0, 255).astype(np.uint8)

    def _load(self, path, is_hazy):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.SYNTHETIC and is_hazy:
            img = self._degrade(img)
        return img



    def _random_crop(self, hazy, clear, size):
        # cropping only (fast)
        h, w = hazy.shape[:2]
        top = np.random.randint(0, h - size + 1) if h > size else 0
        left = np.random.randint(0, w - size + 1) if w > size else 0
        return (hazy [top:top+size, left:left+size],
                clear[top:top+size, left:left+size])

    def _center_crop(self, hazy, clear, size):
        h, w = hazy.shape[:2]
        top = (h - size) // 2 if h > size else 0
        left = (w - size) // 2 if w > size else 0
        return (hazy [top:top+size, left:left+size],
                clear[top:top+size, left:left+size])

    def _color_jitter(self, img):
        """Random brightness/contrast shift — applied to hazy input only."""
        img = img.astype(np.float32)
        # Brightness: shift by ±0.1 * 255
        img += np.random.uniform(-0.1, 0.1) * 255.0
        # Contrast: scale by 0.9–1.1
        mean = img.mean()
        img  = (img - mean) * np.random.uniform(0.9, 1.1) + mean
        return np.clip(img, 0, 255).astype(np.uint8)

    def _add_noise(self, img):
        """Add Gaussian noise (std 0.01–0.03) — applied to hazy input only."""
        std   = np.random.uniform(0.01, 0.03)
        noise = np.random.randn(*img.shape).astype(np.float32) * std
        return np.clip(img / 255.0 + noise, 0, 1) * 255.0

    def _augment(self, hazy, clear):
        if np.random.rand() > 0.5:
            hazy  = np.fliplr(hazy).copy();  clear = np.fliplr(clear).copy()
        if np.random.rand() > 0.3:
            hazy  = np.flipud(hazy).copy();  clear = np.flipud(clear).copy()
        k = np.random.randint(0, 4)
        if k:
            hazy  = np.rot90(hazy,  k).copy(); clear = np.rot90(clear, k).copy()
        return hazy, clear

    def __getitem__(self, idx):
        # Mixed-input selection: coin flip each sample
        # If preproc_pairs available and rand < preproc_prob -> use preprocessed
        # otherwise -> use raw hazy (with jitter + noise augmentation)
        use_preproc = (
            self.preproc_pairs is not None
            and np.random.rand() < self.preproc_prob
        )
        if use_preproc:
            hp, cp = self.preproc_pairs[idx]
        else:
            hp, cp = self.hazy_pairs[idx]

        inp   = self._load(hp, is_hazy=not use_preproc)
        clear = self._load(cp, is_hazy=False)

        # Single fast resize to uniform size (images are ~288-512px, not uniform)
        inp   = cv2.resize(inp,   (self.img_size, self.img_size))
        clear = cv2.resize(clear, (self.img_size, self.img_size))
        if self.augment:
            inp, clear = self._augment(inp, clear)
        inp_t   = torch.from_numpy(inp.astype(np.float32)   / 255.0).permute(2, 0, 1)
        clear_t = torch.from_numpy(clear.astype(np.float32) / 255.0).permute(2, 0, 1)
        return inp_t, clear_t


# ── Loss ───────────────────────────────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    """VGG16 feature-space L1 loss for sharper, perceptually better outputs."""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features[:23]))  # relu4_3 — deeper structural features
        for p in self.features.parameters():
            p.requires_grad = False
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        return self.l1(self.features(pred), self.features(target))


class CombinedLoss(nn.Module):
    def __init__(self, w_l1=0.5, w_ssim=0.3, w_percep=0.2):
        super().__init__()
        self.w_l1     = w_l1
        self.w_ssim   = w_ssim
        self.w_percep = w_percep
        self.l1       = nn.L1Loss()
        self.percep   = VGGPerceptualLoss()

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def _ssim(self, x, y, window_size=11):
        # Force fp32: fp16 underflows the 1e-8 epsilon to zero → NaN
        x, y   = x.float(), y.float()
        C1, C2 = 0.01**2, 0.03**2
        mu_x   = nn.functional.avg_pool2d(x, window_size, 1, window_size//2)
        mu_y   = nn.functional.avg_pool2d(y, window_size, 1, window_size//2)
        mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x*mu_y
        sx2 = nn.functional.avg_pool2d(x*x, window_size, 1, window_size//2) - mu_x2
        sy2 = nn.functional.avg_pool2d(y*y, window_size, 1, window_size//2) - mu_y2
        sxy = nn.functional.avg_pool2d(x*y, window_size, 1, window_size//2) - mu_xy
        num = (2*mu_xy + C1) * (2*sxy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sx2 + sy2 + C2)
        return (num / (den + 1e-8)).mean()

    def forward(self, pred, target):
        loss_l1     = self.l1(pred, target)
        loss_ssim   = 1.0 - self._ssim(pred, target)
        
        loss = self.w_l1 * loss_l1 + self.w_ssim * loss_ssim
        
        if self.w_percep > 0.0:
            loss_percep = self.percep(pred, target)
            loss += self.w_percep * loss_percep
            
        return loss


# ── PSNR ───────────────────────────────────────────────────────────────────────

def psnr(pred, target):
    mse = ((pred - target) ** 2).mean().item()
    return 10 * np.log10(1.0 / (mse + 1e-10))


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_pairs(input_dir, clear_dir):
    """Find matched input/clear pairs. input_dir can be hazy/ or preprocessed/."""
    exts  = {".jpg", ".jpeg", ".png"}
    files = sorted(f for f in os.listdir(input_dir)
                   if os.path.splitext(f)[1].lower() in exts)
    pairs = []
    for f in files:
        stem = os.path.splitext(f)[0]
        for ext in exts:
            cp = os.path.join(clear_dir, stem + ext)
            if os.path.exists(cp):
                pairs.append((os.path.join(input_dir, f), cp))
                break
    return pairs


def plot_curves(epochs_ran, train_losses, val_psnrs, stopped_at, save_path):
    """
    Saves the training curve — train loss (left axis) + val PSNR (right axis).
    The elbow in val PSNR shows where the model converged.
    A vertical dashed line marks where early stopping triggered (if it did).
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_loss = "#e05c5c"
    color_psnr = "#3a9fbf"

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Train Loss (L1 + SSIM)", color=color_loss, fontsize=11)
    ax1.plot(epochs_ran, train_losses, color=color_loss, linewidth=2, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Val PSNR (dB)", color=color_psnr, fontsize=11)
    ax2.plot(epochs_ran, val_psnrs, color=color_psnr, linewidth=2,
             linestyle="--", label="Val PSNR")
    ax2.tick_params(axis="y", labelcolor=color_psnr)

    # Mark best PSNR epoch
    best_epoch = epochs_ran[int(np.argmax(val_psnrs))]
    best_psnr  = max(val_psnrs)
    ax2.axvline(best_epoch, color=color_psnr, linestyle=":", alpha=0.6)
    ax2.annotate(f"  Best: {best_psnr:.2f} dB @ epoch {best_epoch}",
                 xy=(best_epoch, best_psnr),
                 fontsize=9, color=color_psnr)

    # Mark early stop point if triggered before max epochs
    if stopped_at and stopped_at < epochs_ran[-1]:
        ax1.axvline(stopped_at, color="gray", linestyle="--", alpha=0.5)
        ax1.text(stopped_at + 0.3, max(train_losses)*0.95,\
                 f"Early stop\
@ epoch {stopped_at}",
                 fontsize=8, color="gray")

    fig.suptitle("Training Curve - NAFNet + ViT Bottleneck", fontsize=13, fontweight="bold")
    fig.tight_layout()
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  curve saved -> {save_path}")


# ── Train ──────────────────────────────────────────────────────────────────────

# ── Mixed training probability ────────────────────────────────────────────────
# 0.65 = 65% preprocessed, 35% raw hazy each step.
# This matches the inference distribution (enhance.py always preprocesses)
# while keeping the model robust to raw inputs and preprocessing edge cases.
PREPROC_PROB = 1.0   # preprocessed-only: 512px files, fast disk read, matches inference


def train():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # Always use hazy as primary pairs for indexing (defines dataset size)
    hazy_pairs = get_pairs(HAZY_DIR, CLEAR_DIR)
    if not hazy_pairs:
        print(f"[error] No paired images found in {HAZY_DIR} / {CLEAR_DIR}")
        return

    # Load preprocessed pairs if available (same filenames, different folder)
    has_preproc = os.path.isdir(PREPROCESS_DIR) and len(os.listdir(PREPROCESS_DIR)) > 0
    preproc_pairs = get_pairs(PREPROCESS_DIR, CLEAR_DIR) if has_preproc else None

    if has_preproc and preproc_pairs:
        input_src = f"MIXED  ({int(PREPROC_PROB*100)}% preprocessed + {int((1-PREPROC_PROB)*100)}% raw hazy)"
        # Align: only keep indices present in both
        preproc_names = {os.path.basename(p[0]): p for p in preproc_pairs}
        aligned_hazy    = []
        aligned_preproc = []
        for hp, cp in hazy_pairs:
            name = os.path.basename(hp)
            if name in preproc_names:
                aligned_hazy.append((hp, cp))
                aligned_preproc.append(preproc_names[name])
        hazy_pairs    = aligned_hazy
        preproc_pairs = aligned_preproc
        print(f"Aligned pairs   : {len(hazy_pairs)} (hazy + preprocessed)")
    else:
        input_src     = "raw hazy only (no preprocessed folder found)"
        preproc_pairs = None

    print(f"Input mode  : {input_src}")
    print(f"Total pairs : {len(hazy_pairs)}")

    np.random.seed(42)
    idx = np.random.permutation(len(hazy_pairs)).tolist()
    hazy_pairs    = [hazy_pairs[i]    for i in idx]
    if preproc_pairs:
        preproc_pairs = [preproc_pairs[i] for i in idx]

    n_test  = max(1, int(len(hazy_pairs) * TEST_SPLIT))
    n_val   = max(1, int(len(hazy_pairs) * VAL_SPLIT))

    # Split hazy pairs (index-aligned)
    test_hazy   = hazy_pairs[:n_test]
    val_hazy    = hazy_pairs[n_test:n_test + n_val]
    train_hazy  = hazy_pairs[n_test + n_val:]

    # Split preproc pairs with same indices
    test_preproc  = preproc_pairs[:n_test]            if preproc_pairs else None
    val_preproc   = preproc_pairs[n_test:n_test+n_val] if preproc_pairs else None
    train_preproc = preproc_pairs[n_test + n_val:]    if preproc_pairs else None

    np.save(os.path.join(WEIGHTS_DIR, "test_split.npy"),
            np.array([p[0] for p in test_hazy]))

    print(f"Split  ->  train: {len(train_hazy)}  |  val: {len(val_hazy)}  |  test: {len(test_hazy)}")
    print(f"Early stopping patience: {PATIENCE} epochs")
    print(f"Test paths saved -> {WEIGHTS_DIR}/test_split.npy")

    hp_tr, cp_tr = zip(*train_hazy)
    hp_va, cp_va = zip(*val_hazy)
    pp_tr = [p[0] for p in train_preproc] if train_preproc else None
    pp_va = [p[0] for p in val_preproc]   if val_preproc   else None

    train_dl = DataLoader(
        UnderwaterDataset(hp_tr, cp_tr, IMG_SIZE, augment=True,
                          preproc_paths=pp_tr, preproc_prob=PREPROC_PROB),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
        persistent_workers=True)
    val_dl = DataLoader(
        # Val: use preprocessed only (=inference distribution) for stable PSNR
        UnderwaterDataset(hp_va, cp_va, IMG_SIZE, augment=False, full_image=True,
                          preproc_paths=pp_va, preproc_prob=1.0 if pp_va else 0.0),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
        persistent_workers=True)
    print(f"DataLoader workers: train={train_dl.num_workers}  val={val_dl.num_workers}")

    model     = nafnet_vit_small().to(DEVICE)
    total     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = CombinedLoss(w_l1=0.5, w_ssim=0.5, w_percep=0.0).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Warmup: LR ramps linearly from 0 to LR over WARMUP_EPOCHS,
    # then CosineAnnealingLR takes over for the remaining epochs.
    # Prevents loss spike in early epochs when LR=1e-3 is large.
    warmup_sched  = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 1e-6 / LR,   # start near 0
        end_factor   = 1.0,
        total_iters  = WARMUP_EPOCHS)
    cosine_sched  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler     = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers   = [warmup_sched, cosine_sched],
        milestones   = [WARMUP_EPOCHS])

    print(f"Parameters : {total:,}")
    print(f"Device     : {DEVICE}")
    print(f"Max epochs : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")

    # ── CSV log setup ──────────────────────────────────────────────────────────
    csv_path = os.path.join(WEIGHTS_DIR, "training_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_psnr", "lr"])

    # ── Tracking ───────────────────────────────────────────────────────────────
    best_psnr       = 0.0
    epochs_no_impv  = 0
    stopped_at      = None
    hist_epochs     = []
    hist_loss       = []
    hist_psnr       = []

    # ── Mixed precision (bf16) ─────────────────────────────────────────────────
    # bf16 has same exponent range as fp32 — no overflow, no GradScaler needed.
    # RTX 5070 (Blackwell) has native bf16 tensor core support.
    amp_dtype = torch.bfloat16
    print(f"Mixed precision : enabled (bf16 — no overflow, no scaler needed)")

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch [{epoch:>3}/{EPOCHS}]", leave=False)
        for hazy_b, clear_b in pbar:
            hazy_b = hazy_b.to(DEVICE, non_blocking=True)
            clear_b = clear_b.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                loss = criterion(model(hazy_b), clear_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= len(train_dl)

        # Val
        model.eval()
        val_psnr = 0.0
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
            for hazy_b, clear_b in val_dl:
                hazy_b = hazy_b.to(DEVICE, non_blocking=True)
                clear_b = clear_b.to(DEVICE, non_blocking=True)
                val_psnr += psnr(model(hazy_b), clear_b)
        val_psnr /= len(val_dl)
        # Track history for curve (must be before running avg calculation)
        hist_epochs.append(epoch)
        hist_loss.append(train_loss)
        hist_psnr.append(val_psnr)
        # Use 3-epoch running average for early-stop decisions
        avg_psnr = np.mean(hist_psnr[-3:])
        scheduler.step()

        lr_now     = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch:>3}/{EPOCHS}]  loss={train_loss:.4f}  "
              f"val_psnr={val_psnr:.2f} dB  avg3={avg_psnr:.2f}  "
              f"lr={lr_now:.2e}")

        # Log to CSV
        csv_writer.writerow([epoch, round(train_loss, 6),
                              round(val_psnr, 4), lr_now])
        csv_file.flush()

        # Numbered checkpoint
        if epoch % SAVE_EVERY == 0:
            ckpt = os.path.join(WEIGHTS_DIR, f"nafnet_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  checkpoint -> {ckpt}")

        # Best model + early stopping counter
        if avg_psnr > best_psnr:
            best_psnr      = avg_psnr
            epochs_no_impv = 0
            torch.save(model.state_dict(),
                       os.path.join(WEIGHTS_DIR, "nafnet_final.pth"))
        else:
            epochs_no_impv += 1
            print(f"  no improvement ({epochs_no_impv}/{PATIENCE})  [avg_psnr={avg_psnr:.2f}]")
            if epochs_no_impv >= PATIENCE:
                stopped_at = epoch
                print(f"\
Early stopping triggered at epoch {epoch}.")
                print(f"Best val PSNR was {best_psnr:.2f} dB -- saved as nafnet_final.pth")
                break

    csv_file.close()

    # Plot and save curve
    curve_path = os.path.join(WEIGHTS_DIR, "training_curve.png")
    plot_curves(hist_epochs, hist_loss, hist_psnr, stopped_at, curve_path)

    print(f"Training complete.")
    print(f"Best val PSNR  : {best_psnr:.2f} dB")
    print(f"Weights        : {WEIGHTS_DIR}/nafnet_final.pth")
    print(f"Training log   : {WEIGHTS_DIR}/training_log.csv")
    print(f"Training curve : {WEIGHTS_DIR}/training_curve.png")
    print(f"Run next       : python test.py")


if __name__ == "__main__":
    train()