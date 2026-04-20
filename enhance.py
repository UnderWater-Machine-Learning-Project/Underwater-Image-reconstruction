"""
Underwater image enhancement pipeline.

Enhancement stages (applied in order):
    1. Classical preprocessing      -- WB + UDCP + CLAHE + Sharpen (preprocess.py)
    2. U-Net + Swin bottleneck      -- trained neural enhancement (MSR fallback)
    3. Bilateral denoise            -- final smooth preserving edges
    4. Saturation restore           -- compensates depth absorption

The classical preprocessing stage (Stage 1) is the SAME module used to
preprocess the training dataset. This ensures train-inference consistency:
the U-Net sees the same kind of input at both training and inference time.

Neural priority: U-Net -> WaterNet -> MSR classical fallback.
"""

import cv2
import numpy as np
import torch

from preprocess import preprocess as classical_preprocess


# -- Neural model loaders -------------------------------------------------------

def load_unet(weights_path: str):
    """
    Loads trained U-Net + Swin checkpoint.
    Returns (model, device) or None on failure.
    """
    try:
        from unet import UNet
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = UNet().to(device)
        model.load_state_dict(
            torch.load(weights_path, map_location=device), strict=False)
        model.eval()
        print(f"U-Net loaded ({device})  <-  {weights_path}")
        return model, device
    except Exception as e:
        print(f"[warn] U-Net load failed: {e}")
        return None


def infer_unet(bundle, img_bgr: np.ndarray) -> np.ndarray:
    """U-Net + Swin forward pass. Fed the preprocessed base, not raw image."""
    model, device = bundle
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w    = img_rgb.shape[:2]
    ph = (16 - h % 16) % 16
    pw = (16 - w % 16) % 16
    if ph or pw:
        img_rgb = np.pad(img_rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
    out_np = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)[:h, :w]
    return cv2.cvtColor(
        (np.clip(out_np, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def load_waternet():
    """
    Loads WaterNet via torch.hub -- fallback when U-Net weights unavailable.
    Smoke-tests preprocess output (must return 3 tensors).
    Returns (preprocess, postprocess, model) or None.
    """
    try:
        pre, post, model = torch.hub.load(
            "tnwei/waternet", "waternet", pretrained=True, verbose=False)
        model.eval()
        inputs = list(pre(np.zeros((64, 64, 3), dtype=np.uint8)))
        if len(inputs) != 3:
            print(f"[warn] WaterNet preprocess returned {len(inputs)} tensors -- skipping.")
            return None
        print("WaterNet loaded  (fallback)")
        return pre, post, model
    except Exception as e:
        print(f"[warn] WaterNet unavailable: {e}")
        return None


def infer_waternet(bundle, img_bgr: np.ndarray) -> np.ndarray:
    pre, post, model = bundle
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    wb_t, ce_t, gc_t = list(pre(img_rgb))
    with torch.no_grad():
        out = model(wb_t, ce_t, gc_t)
    return cv2.cvtColor(post(out), cv2.COLOR_RGB2BGR)


# -- Internal helpers -----------------------------------------------------------

def _msr_blend(img_bgr: np.ndarray,
               base: np.ndarray,
               sigmas: list = [15, 80, 250],
               blend: float = 0.25) -> np.ndarray:
    """
    Multi-Scale Retinex blended lightly onto an already-corrected base.
    Used only when no neural model is available.
    Blending at 25% adds illumination correction benefit without blowout.
    """
    img_f = img_bgr.astype(np.float32) + 1.0
    msr   = np.zeros_like(img_f)
    for s in sigmas:
        L    = cv2.GaussianBlur(img_f, (0, 0), s)
        msr += np.log(img_f) - np.log(L + 1.0)
    msr /= len(sigmas)
    msr_out = np.zeros_like(img_bgr, dtype=np.float32)
    for c in range(3):
        ch = msr[:, :, c]
        msr_out[:, :, c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6) * 255.0
    return np.clip(blend * msr_out + (1 - blend) * base.astype(np.float32),
                   0, 255).astype(np.uint8)


# -- Master enhance function ----------------------------------------------------

def enhance(img_bgr: np.ndarray,
            unet_bundle=None,
            waternet_bundle=None) -> np.ndarray:
    """
    Main enhancement entry point.

    Stage 1: Classical preprocessing (WB + UDCP + CLAHE + Sharpen)
             Uses the SAME preprocess.py module as training dataset prep.
             This ensures train-inference consistency.

    Stage 2: Neural enhancement (U-Net -> WaterNet -> MSR fallback)
             U-Net receives the preprocessed base (not raw) -- it was
             trained on preprocessed inputs, so it expects them.

    Stage 3: Post-processing (bilateral denoise + saturation restore)
    """
    orig = img_bgr.copy()

    # -- Stage 1: Classical preprocessing (same as training input) -----------
    base = classical_preprocess(img_bgr)

    # -- Stage 2: Neural enhancement ----------------------------------------
    # U-Net receives preprocessed base -- matches training distribution
    neural = None
    if unet_bundle is not None:
        try:
            neural = infer_unet(unet_bundle, base)
        except Exception as e:
            print(f"  [warn] U-Net inference failed: {e}")
    if neural is None and waternet_bundle is not None:
        try:
            neural = infer_waternet(waternet_bundle, img_bgr)
        except Exception as e:
            print(f"  [warn] WaterNet inference failed: {e}")

    if neural is not None:
        fused = np.clip(0.60 * neural.astype(np.float32) +
                        0.40 * base.astype(np.float32), 0, 255).astype(np.uint8)
    else:
        fused = _msr_blend(img_bgr, base, blend=0.25)

    # -- Stage 3: Post-processing -------------------------------------------
    # Bilateral denoise -- preserves edges while smoothing
    fused = cv2.bilateralFilter(fused, d=5, sigmaColor=25, sigmaSpace=25)

    # Saturation restore -- compensates depth absorption
    bright = float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean())
    b_m    = float(img_bgr[:, :, 0].mean())
    g_m    = float(img_bgr[:, :, 1].mean())
    r_m    = float(img_bgr[:, :, 2].mean())
    cast   = max(min((b_m / max(r_m, 1.0)) / 2.0, 1.0),
                 min((g_m / max(r_m, 1.0)) / 2.0, 1.0))

    hsv = cv2.cvtColor(fused, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.15 + 0.15 * cast), 0, 255)
    fused = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return fused if fused.mean() >= orig.mean() * 0.75 else orig
