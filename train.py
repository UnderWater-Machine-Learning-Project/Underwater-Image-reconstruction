"""
Training script for U-Net + Swin Transformer underwater image enhancement.

Dataset structure expected:
    dataset/
    ├── hazy/     ← degraded underwater images (your input)
    └── clear/    ← clean reference images    (ground truth)

    Filenames must match: dataset/hazy/001.jpg <-> dataset/clear/001.jpg

Split: 80% train / 10% val / 10% test
Early stopping: halts training if val PSNR does not improve for PATIENCE epochs.
Outputs:
    weights/unet_final.pth       ← best checkpoint (loaded by main.py + test.py)
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
from unet import UNet


# ── Config ─────────────────────────────────────────────────────────────────────

DATASET_DIR  = "dataset"
WEIGHTS_DIR  = "weights"
HAZY_DIR     = os.path.join(DATASET_DIR, "hazy")
CLEAR_DIR    = os.path.join(DATASET_DIR, "clear")

EPOCHS       = 100     # max epochs — early stopping will cut this short
BATCH_SIZE   = 4       # reduce to 2 if GPU OOM, 1 for CPU
LR           = 1e-4
IMG_SIZE     = 256     # training crop size — must be multiple of 16
SAVE_EVERY   = 5       # save numbered checkpoint every N epochs
PATIENCE     = 8       # early stopping: halt after N epochs with no val PSNR improvement

TRAIN_SPLIT  = 0.80
VAL_SPLIT    = 0.10
TEST_SPLIT   = 0.10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ────────────────────────────────────────────────────────────────────

class UnderwaterDataset(Dataset):
    SYNTHETIC = False   # set True if you only have clear images, no hazy pairs

    def __init__(self, hazy_paths, clear_paths, img_size=256, augment=True):
        self.pairs    = list(zip(hazy_paths, clear_paths))
        self.img_size = img_size
        self.augment  = augment

    def __len__(self):
        return len(self.pairs)

    def _degrade(self, img):
        img = img.astype(np.float32) / 255.0
        img[:, :, 0] *= 0.6
        img[:, :, 1] *= 0.85
        img[:, :, 2]  = np.clip(img[:, :, 2] * 1.1 + 0.05, 0, 1)
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
        h, w = hazy.shape[:2]
        if h < size or w < size:
            hazy  = cv2.resize(hazy,  (max(w, size), max(h, size)))
            clear = cv2.resize(clear, (max(w, size), max(h, size)))
            h, w  = hazy.shape[:2]
        top  = np.random.randint(0, h - size + 1)
        left = np.random.randint(0, w - size + 1)
        return (hazy [top:top+size, left:left+size],
                clear[top:top+size, left:left+size])

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
        hp, cp  = self.pairs[idx]
        hazy    = self._load(hp, is_hazy=True)
        clear   = self._load(cp, is_hazy=False)
        hazy, clear = self._random_crop(hazy, clear, self.img_size)
        if self.augment:
            hazy, clear = self._augment(hazy, clear)
        hazy_t  = torch.from_numpy(hazy.astype(np.float32)  / 255.0).permute(2, 0, 1)
        clear_t = torch.from_numpy(clear.astype(np.float32) / 255.0).permute(2, 0, 1)
        return hazy_t, clear_t


# ── Loss ───────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha
        self.l1    = nn.L1Loss()

    def _ssim(self, x, y, window_size=11):
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
        return self.alpha * self.l1(pred, target) + (1-self.alpha) * (1.0 - self._ssim(pred, target))


# ── PSNR ───────────────────────────────────────────────────────────────────────

def psnr(pred, target):
    mse = ((pred - target) ** 2).mean().item()
    return 10 * np.log10(1.0 / (mse + 1e-10))


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_pairs(hazy_dir, clear_dir):
    exts  = {".jpg", ".jpeg", ".png"}
    files = sorted(f for f in os.listdir(hazy_dir)
                   if os.path.splitext(f)[1].lower() in exts)
    pairs = []
    for f in files:
        stem = os.path.splitext(f)[0]
        for ext in exts:
            cp = os.path.join(clear_dir, stem + ext)
            if os.path.exists(cp):
                pairs.append((os.path.join(hazy_dir, f), cp))
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

    fig.suptitle("Training Curve — U-Net + Swin Transformer", fontsize=13, fontweight="bold")
    fig.tight_layout()
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  curve saved → {save_path}")


# ── Train ──────────────────────────────────────────────────────────────────────

def train():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    pairs = get_pairs(HAZY_DIR, CLEAR_DIR)
    if not pairs:
        print(f"[error] No paired images found in {HAZY_DIR} / {CLEAR_DIR}")
        return

    print(f"Total pairs : {len(pairs)}")
    np.random.seed(42)
    np.random.shuffle(pairs)

    n_test  = max(1, int(len(pairs) * TEST_SPLIT))
    n_val   = max(1, int(len(pairs) * VAL_SPLIT))
    test_pairs  = pairs[:n_test]
    val_pairs   = pairs[n_test:n_test + n_val]
    train_pairs = pairs[n_test + n_val:]

    np.save(os.path.join(WEIGHTS_DIR, "test_split.npy"),
            np.array([p[0] for p in test_pairs]))

    print(f"Split  →  train: {len(train_pairs)}  |  val: {len(val_pairs)}  |  test: {len(test_pairs)}")
    print(f"Early stopping patience: {PATIENCE} epochs")
    print(f"Test paths saved → {WEIGHTS_DIR}/test_split.npy")

    hp_tr, cp_tr = zip(*train_pairs)
    hp_va, cp_va = zip(*val_pairs)

    train_dl = DataLoader(
        UnderwaterDataset(hp_tr, cp_tr, IMG_SIZE, augment=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(
        UnderwaterDataset(hp_va, cp_va, IMG_SIZE, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model     = UNet(base=64).to(DEVICE)
    total     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = CombinedLoss(alpha=0.8)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    print(f"Parameters : {total:,}")
    print(f"Device     : {DEVICE}")
    print(f"Max epochs : {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")

    # ── CSV log setup ──────────────────────────────────────────────────────────
    csv_path = os.path.join(WEIGHTS_DIR, "training_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_psnr", "swin_alpha", "lr"])

    # ── Tracking ───────────────────────────────────────────────────────────────
    best_psnr       = 0.0
    epochs_no_impv  = 0
    stopped_at      = None
    hist_epochs     = []
    hist_loss       = []
    hist_psnr       = []

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for hazy_b, clear_b in train_dl:
            hazy_b, clear_b = hazy_b.to(DEVICE), clear_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(hazy_b), clear_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # Val
        model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for hazy_b, clear_b in val_dl:
                hazy_b, clear_b = hazy_b.to(DEVICE), clear_b.to(DEVICE)
                val_psnr += psnr(model(hazy_b), clear_b)
        val_psnr /= len(val_dl)
        scheduler.step()

        swin_alpha = model.swin.alpha.item()
        lr_now     = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch:>3}/{EPOCHS}]  loss={train_loss:.4f}  "
              f"val_psnr={val_psnr:.2f} dB  swin_alpha={swin_alpha:.4f}  "
              f"lr={lr_now:.2e}")

        # Log to CSV
        csv_writer.writerow([epoch, round(train_loss, 6),
                              round(val_psnr, 4), round(swin_alpha, 6), lr_now])
        csv_file.flush()

        # Track history for curve
        hist_epochs.append(epoch)
        hist_loss.append(train_loss)
        hist_psnr.append(val_psnr)

        # Numbered checkpoint
        if epoch % SAVE_EVERY == 0:
            ckpt = os.path.join(WEIGHTS_DIR, f"unet_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  checkpoint → {ckpt}")

        # Best model + early stopping counter
        if val_psnr > best_psnr:
            best_psnr      = val_psnr
            epochs_no_impv = 0
            torch.save(model.state_dict(),
                       os.path.join(WEIGHTS_DIR, "unet_final.pth"))
        else:
            epochs_no_impv += 1
            print(f"  no improvement ({epochs_no_impv}/{PATIENCE})")
            if epochs_no_impv >= PATIENCE:
                stopped_at = epoch
                print(f"\
Early stopping triggered at epoch {epoch}.")
                print(f"Best val PSNR was {best_psnr:.2f} dB — saved as unet_final.pth")
                break

    csv_file.close()

    # Plot and save curve
    curve_path = os.path.join(WEIGHTS_DIR, "training_curve.png")
    plot_curves(hist_epochs, hist_loss, hist_psnr, stopped_at, curve_path)

    print(f"Training complete.")
    print(f"Best val PSNR  : {best_psnr:.2f} dB")
    print(f"Weights        : {WEIGHTS_DIR}/unet_final.pth")
    print(f"Training log   : {WEIGHTS_DIR}/training_log.csv")
    print(f"Training curve : {WEIGHTS_DIR}/training_curve.png")
    print(f"Run next       : python test.py")


if __name__ == "__main__":
    train()
