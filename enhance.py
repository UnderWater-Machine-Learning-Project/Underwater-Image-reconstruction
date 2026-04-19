"""
Underwater image enhancement pipeline.

Enhancement stages (applied in order):
    1. LAB chrominance correction   — corrects blue AND green cast via a/b axes
    2. CLAHE on L + adaptive gamma  — local contrast + brightness lift
    3. U-Net + ViT bottleneck       — trained neural enhancement (MSR fallback)
    4. UDCP dehazing                — underwater dark channel prior (R+G only)
    5. Unsharp mask sharpening      — controlled edge sharpening
    6. Bilateral denoise            — final smooth preserving edges
    7. Saturation restore           — compensates depth absorption

FFT-based injection was removed. It amplifies noise in turbid/green-cast
images to produce confetti artifacts. Unsharp mask gives equivalent edge
sharpening with no frequency-domain instability.

Neural priority: U-Net → WaterNet → MSR classical fallback.
"""

import cv2
import numpy as np
import torch


# ── Neural model loaders ───────────────────────────────────────────────────────

def load_unet(weights_path: str):
    """
    Loads trained U-Net + ViT checkpoint.
    Returns (model, device) or None on failure.
    """
    try:
        from unet import UNet
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = UNet().to(device)
        model.load_state_dict(
            torch.load(weights_path, map_location=device), strict=False)
        model.eval()
        print(f"U-Net loaded ({device})  ←  {weights_path}")
        return model, device
    except Exception as e:
        print(f"[warn] U-Net load failed: {e}")
        return None


def infer_unet(bundle, img_bgr: np.ndarray) -> np.ndarray:
    """U-Net + ViT forward pass. Fed the CLAHE-corrected base, not raw image."""
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
    Loads WaterNet via torch.hub — fallback when U-Net weights unavailable.
    Smoke-tests preprocess output (must return 3 tensors).
    Returns (preprocess, postprocess, model) or None.
    """
    try:
        pre, post, model = torch.hub.load(
            "tnwei/waternet", "waternet", pretrained=True, verbose=False)
        model.eval()
        inputs = list(pre(np.zeros((64, 64, 3), dtype=np.uint8)))
        if len(inputs) != 3:
            print(f"[warn] WaterNet preprocess returned {len(inputs)} tensors — skipping.")
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


# ── Internal helpers ───────────────────────────────────────────────────────────

def _udcp_dehaze(img_bgr: np.ndarray,
                 omega: float = 0.85,
                 patch_size: int = 15) -> np.ndarray:
    """
    Underwater Dark Channel Prior (Drews et al. 2013).

    Standard DCP uses min across all 3 RGB channels to find the dark channel.
    This fails underwater because blue is always high — DCP reads the entire
    scene as haze and massively over-dehazed it.

    UDCP fix: compute the dark channel using only R and G channels (BGR
    indices 2 and 1). Blue is excluded because it dominates underwater and
    is not part of the scattering model in this domain.
    """
    img_f  = img_bgr.astype(np.float64) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))

    # Dark channel: min of R and G only
    rg_min = np.minimum(img_f[:, :, 2], img_f[:, :, 1])
    dark   = cv2.erode(rg_min.astype(np.float32), kernel)

    # Atmospheric light: average of top 0.1% brightest dark-channel pixels
    n_px    = max(1, int(dark.size * 0.001))
    indices = np.unravel_index(np.argsort(dark.ravel())[-n_px:], dark.shape)
    A       = np.clip(img_f[indices[0], indices[1], :].mean(axis=0), 0.2, 1.0)

    # Transmission map
    rg_norm = np.minimum(img_f[:, :, 2] / max(A[2], 1e-6),
                         img_f[:, :, 1] / max(A[1], 1e-6))
    t = np.clip(
        1.0 - omega * cv2.erode(rg_norm.astype(np.float32), kernel),
        0.1, 1.0)

    # Recover scene radiance: J = (I - A) / t + A
    out = np.zeros_like(img_f)
    for c in range(3):
        out[:, :, c] = (img_f[:, :, c] - A[c]) / t + A[c]

    return np.clip(out * 255, 0, 255).astype(np.uint8)


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


# ── Master enhance function ────────────────────────────────────────────────────

def enhance(img_bgr: np.ndarray,
            unet_bundle=None,
            waternet_bundle=None) -> np.ndarray:
    """
    Main enhancement entry point.

    Detects cast type from channel ratios:
      blue cast  — common in open water,    corrected via b-axis shift in LAB
      green cast — common in turbid/coastal, corrected via a-axis shift

    Neural priority: U-Net → WaterNet → MSR classical fallback.
    U-Net is fed the CLAHE-corrected base (not raw) — it only needs to learn
    the residual haze/scatter on top of already color-corrected input.

    Stages 4-7 (UDCP, sharpen, denoise, saturation) apply identically
    regardless of which neural path was used.
    """
    orig   = img_bgr.copy()
    bright = float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean())
    b_m    = float(img_bgr[:, :, 0].mean())
    g_m    = float(img_bgr[:, :, 1].mean())
    r_m    = float(img_bgr[:, :, 2].mean())

    cast_b = min((b_m / max(r_m, 1.0)) / 2.0, 1.0)
    cast_g = min((g_m / max(r_m, 1.0)) / 2.0, 1.0)
    cast   = max(cast_b, cast_g)

    # ── Stage 1: LAB chrominance correction ──────────────────────────────
    # a-axis: negative=green, positive=red  → green cast → push a up
    # b-axis: positive=yellow, negative=blue → blue cast → push b up
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_ch, a_ch, b_ch = cv2.split(lab)
    s   = 0.55 + 0.35 * cast
    cap = 30 + int(cast * 15)
    a_ch = np.clip(a_ch + np.clip((128 - a_ch.mean()) * s, -cap, cap), 0, 255)
    b_ch = np.clip(b_ch + np.clip((128 - b_ch.mean()) * s, -cap, cap), 0, 255)

    # ── Stage 2: CLAHE on L + adaptive gamma ─────────────────────────────
    l_u8 = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(
               np.clip(l_ch, 0, 255).astype(np.uint8))
    base = cv2.cvtColor(
        cv2.merge([l_u8, a_ch.astype(np.uint8), b_ch.astype(np.uint8)]),
        cv2.COLOR_LAB2BGR)

    gamma = float(np.clip(0.50 + (bright / 255.0) * 0.40, 0.50, 0.90))
    lut   = np.array([int(((i / 255.0) ** gamma) * 255) for i in range(256)],
                     dtype=np.uint8)
    base  = cv2.LUT(base, lut)

    # ── Stage 3: U-Net + ViT → WaterNet → MSR fallback ───────────────────
    # U-Net receives the CLAHE base (not raw) — simpler task, better results
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

    # ── Stage 4: UDCP dehazing ────────────────────────────────────────────
    # Skip for very dark scenes or extreme cast (transmission map unreliable)
    if bright >= 80 and cast < 0.80:
        dh    = _udcp_dehaze(fused)
        fused = np.clip(0.60 * dh.astype(np.float32) +
                        0.40 * fused.astype(np.float32), 0, 255).astype(np.uint8)

    # ── Stage 5: Unsharp mask sharpening ─────────────────────────────────
    blur  = cv2.GaussianBlur(fused.astype(np.float32), (0, 0), 1.5)
    fused = np.clip(fused.astype(np.float32) + 0.35 * (fused.astype(np.float32) - blur),
                    0, 255).astype(np.uint8)

    # ── Stage 6: Bilateral denoise ────────────────────────────────────────
    fused = cv2.bilateralFilter(fused, d=5, sigmaColor=25, sigmaSpace=25)

    # ── Stage 7: Saturation restore ───────────────────────────────────────
    hsv = cv2.cvtColor(fused, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.15 + 0.15 * cast), 0, 255)
    fused = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return fused if fused.mean() >= orig.mean() * 0.75 else orig
