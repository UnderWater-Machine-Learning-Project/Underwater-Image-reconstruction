"""
Underwater Image Enhancement Pipeline
======================================
Stage 1 — Classical preprocessing  : preprocess.py  (WB → UDCP → CLAHE → Sharpen)
Stage 2 — Neural enhancement       : NAFNet + ViT Bottleneck
Stage 3 — Post-processing          : bilateral denoise + saturation restore

Train-inference consistency:
    preprocess.py is the SAME module used by preprocess_dataset.py to prepare
    the training dataset. The NAFNet therefore sees the same input distribution
    at inference time as it did during training.

Usage:
    from enhance import enhance, load_model
    bundle = load_model("weights/nafnet_final.pth")
    enhanced_bgr = enhance(raw_bgr, bundle)
"""

import cv2
import numpy as np
import torch

from preprocess import preprocess as classical_preprocess


# ── Model loader ───────────────────────────────────────────────────────────────

def load_model(weights_path: str):
    """
    Load trained NAFNet + ViT checkpoint.

    Returns:
        (model, device) tuple on success, None on failure.
    """
    try:
        from nafnet_vit import nafnet_vit_small
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = nafnet_vit_small().to(device)
        state  = torch.load(weights_path, map_location=device)
        model.load_state_dict(state, strict=True)
        model.eval()
        print(f"NAFNet loaded  ({device})  ←  {weights_path}")
        return model, device
    except Exception as e:
        print(f"[warn] NAFNet load failed: {e}")
        return None


# ── Neural inference ───────────────────────────────────────────────────────────

def _infer(bundle, img_bgr: np.ndarray) -> np.ndarray:
    """
    NAFNet forward pass.
    Pads H and W to multiples of 16, runs inference, crops padding back off.
    Input/output: uint8 BGR.
    """
    model, device = bundle
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w    = img_rgb.shape[:2]

    # Pad to multiple of 16
    ph = (16 - h % 16) % 16
    pw = (16 - w % 16) % 16
    if ph or pw:
        img_rgb = np.pad(img_rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")

    t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)

    out_np = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)[:h, :w]
    return cv2.cvtColor((np.clip(out_np, 0, 1) * 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR)


# ── Post-processing ────────────────────────────────────────────────────────────

def _post_process(img_bgr: np.ndarray, orig_bgr: np.ndarray) -> np.ndarray:
    """
    Bilateral denoise + saturation restore.

    Bilateral filter preserves edges while smoothing compression artefacts.
    Saturation boost is proportional to detected blue/green cast — deeper
    images get more saturation recovery, shallow ones get less.
    """
    # Bilateral denoise
    out = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=25, sigmaSpace=25)

    # Saturation restore — scale based on original cast ratio
    b_m = float(orig_bgr[:, :, 0].mean())
    g_m = float(orig_bgr[:, :, 1].mean())
    r_m = float(orig_bgr[:, :, 2].mean())
    cast = max(
        min((b_m / max(r_m, 1.0)) / 2.0, 1.0),
        min((g_m / max(r_m, 1.0)) / 2.0, 1.0),
    )
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.10 + 0.15 * cast), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ── Master entry point ─────────────────────────────────────────────────────────

def enhance(img_bgr: np.ndarray, bundle=None) -> np.ndarray:
    """
    Full enhancement pipeline.

    Stage 1: Classical preprocessing — same as training input preparation.
             (Gray World WB → UDCP → CLAHE → Unsharp Mask)
    Stage 2: NAFNet + ViT neural enhancement.
             Falls back to classical-only if no model is loaded.
    Stage 3: Bilateral denoise + saturation restore.

    Safety guard: if the enhanced output is significantly darker than the
    original (mean < 75% of original), returns the classical-only result.

    Args:
        img_bgr : uint8 BGR image, any resolution.
        bundle  : (model, device) from load_model(), or None for classical only.

    Returns:
        Enhanced uint8 BGR image, same resolution as input.
    """
    orig = img_bgr.copy()

    # Stage 1: Classical preprocessing (same module as training dataset prep)
    preprocessed = classical_preprocess(img_bgr)

    # Stage 2: Neural enhancement
    if bundle is not None:
        try:
            neural = _infer(bundle, preprocessed)
        except Exception as e:
            print(f"  [warn] NAFNet inference failed: {e}")
            neural = None
    else:
        neural = None

    if neural is not None:
        # Blend: 70% neural, 30% preprocessed classical baseline
        fused = np.clip(
            0.70 * neural.astype(np.float32) +
            0.30 * preprocessed.astype(np.float32),
            0, 255
        ).astype(np.uint8)
    else:
        fused = preprocessed

    # Stage 3: Post-processing
    fused = _post_process(fused, orig)

    # Safety: don't return a result darker than 75% of the original
    if fused.mean() < orig.mean() * 0.75:
        return preprocessed

    return fused
