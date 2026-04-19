"""
Test script for U-Net + Swin Transformer underwater image enhancement.

Loads the held-out test split saved by train.py (weights/test_split.npy)
and evaluates the trained model on images it has never seen.

Metrics computed per image and averaged:
    PSNR  — Peak Signal-to-Noise Ratio (higher = better, >30dB is good)
    SSIM  — Structural Similarity Index (higher = better, >0.85 is good)

Output:
    test_outputs/<name>_result.jpg  — hazy | enhanced | clear (3-panel)
    Terminal                        — per-image + overall metrics table

Usage:
    python test.py
"""

import os
import cv2
import numpy as np
import torch
from unet import UNet


# ── Config ─────────────────────────────────────────────────────────────────────

WEIGHTS_PATH  = "weights/unet_final.pth"
TEST_SPLIT    = "weights/test_split.npy"
CLEAR_DIR     = "dataset/clear"
OUTPUT_DIR    = "test_outputs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_psnr(pred, target):
    mse = ((pred.astype(np.float64) - target.astype(np.float64)) ** 2).mean()
    return 10 * np.log10(255.0**2 / (mse + 1e-10))


def compute_ssim(pred, target):
    C1, C2 = (0.01*255)**2, (0.03*255)**2
    pred_f, tgt_f = pred.astype(np.float64), target.astype(np.float64)
    mu1    = cv2.GaussianBlur(pred_f, (11,11), 1.5)
    mu2    = cv2.GaussianBlur(tgt_f,  (11,11), 1.5)
    mu1_sq, mu2_sq, mu1mu2 = mu1**2, mu2**2, mu1*mu2
    s11 = cv2.GaussianBlur(pred_f*pred_f, (11,11), 1.5) - mu1_sq
    s22 = cv2.GaussianBlur(tgt_f*tgt_f,  (11,11), 1.5) - mu2_sq
    s12 = cv2.GaussianBlur(pred_f*tgt_f,  (11,11), 1.5) - mu1mu2
    num = (2*mu1mu2 + C1) * (2*s12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (s11 + s22 + C2)
    return float((num / (den + 1e-8)).mean())


# ── Inference ──────────────────────────────────────────────────────────────────

def enhance_image(model, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w    = img_rgb.shape[:2]
    ph = (16 - h % 16) % 16
    pw = (16 - w % 16) % 16
    if ph or pw:
        img_rgb = np.pad(img_rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(t)
    out_np = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)[:h, :w]
    return cv2.cvtColor((np.clip(out_np, 0, 1) * 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR)


# ── Main ───────────────────────────────────────────────────────────────────────

def test():
    # Load model
    if not os.path.exists(WEIGHTS_PATH):
        print(f"[error] Weights not found: {WEIGHTS_PATH}")
        print("        Run train.py first.")
        return

    model = UNet(base=64).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE), strict=False)
    model.eval()
    print(f"Model loaded from {WEIGHTS_PATH}")
    print(f"Device: {DEVICE}\n")

    # Load test split
    if not os.path.exists(TEST_SPLIT):
        print(f"[error] Test split not found: {TEST_SPLIT}")
        print("        Run train.py first — it saves the test split.")
        return

    hazy_paths = np.load(TEST_SPLIT, allow_pickle=True).tolist()
    print(f"Test images: {len(hazy_paths)}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Per-image evaluation ───────────────────────────────────────────────────
    bar = "─" * 62
    print(f"{bar}")
    print(f"  {'Image':<28}  {'PSNR (dB)':>9}  {'SSIM':>7}")
    print(f"{bar}")

    all_psnr, all_ssim = [], []
    exts = {".jpg", ".jpeg", ".png"}

    for hp in hazy_paths:
        fname = os.path.basename(hp)
        stem  = os.path.splitext(fname)[0]

        # Find matching clear image
        clear_path = None
        for ext in exts:
            cp = os.path.join(CLEAR_DIR, stem + ext)
            if os.path.exists(cp):
                clear_path = cp
                break

        hazy_img = cv2.imread(hp)
        if hazy_img is None:
            print(f"  {fname:<28}  [cannot read hazy image]")
            continue

        # Run U-Net
        enhanced = enhance_image(model, hazy_img)

        # Compute metrics if clear reference available
        p_val = ssim_val = None
        if clear_path:
            clear_img = cv2.imread(clear_path)
            if clear_img is not None:
                # Resize clear to match enhanced if needed
                if clear_img.shape[:2] != enhanced.shape[:2]:
                    clear_img = cv2.resize(clear_img, (enhanced.shape[1], enhanced.shape[0]))
                p_val    = compute_psnr(enhanced, clear_img)
                ssim_val = compute_ssim(enhanced, clear_img)
                all_psnr.append(p_val)
                all_ssim.append(ssim_val)

        if p_val is not None:
            print(f"  {fname:<28}  {p_val:>9.2f}  {ssim_val:>7.4f}")
        else:
            print(f"  {fname:<28}  {'no clear ref':>9}  {'—':>7}")

        # Save 3-panel: hazy | enhanced | clear
        panels = [hazy_img, enhanced]
        if clear_path and clear_img is not None:
            panels.append(clear_img)
        sep = np.full((hazy_img.shape[0], 4, 3), 20, dtype=np.uint8)
        result = panels[0]
        for p in panels[1:]:
            result = np.hstack([result, sep, p])
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{stem}_result.jpg"), result,
                    [cv2.IMWRITE_JPEG_QUALITY, 93])

    # ── Overall results ────────────────────────────────────────────────────────
    print(f"{bar}")
    if all_psnr:
        print(f"  {'AVERAGE':<28}  {np.mean(all_psnr):>9.2f}  {np.mean(all_ssim):>7.4f}")
        print(f"  {'BEST':<28}  {max(all_psnr):>9.2f}  {max(all_ssim):>7.4f}")
        print(f"  {'WORST':<28}  {min(all_psnr):>9.2f}  {min(all_ssim):>7.4f}")
        print(f"{bar}")
        print("\nPSNR > 30 dB  = good quality")
        print("SSIM > 0.85   = good structural similarity")
    print(f"\nVisuals saved → {OUTPUT_DIR}/")


if __name__ == "__main__":
    test()
