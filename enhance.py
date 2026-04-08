"""
Underwater image enhancement pipeline.

Enhancement stages (applied in order):
    1. LAB chrominance correction   — corrects blue AND green cast via a/b axes
    2. CLAHE on L + adaptive gamma  — local contrast + brightness lift
    3. MSR blend                    — 25% illumination decomposition blended in
    4. Guided filter dehazing       — edge-preserving scatter removal (conditional)
    5. Unsharp mask sharpening      — controlled edge sharpening
    6. Bilateral denoise            — final smooth preserving edges
    7. Saturation restore           — compensates depth absorption

FFT-based injection was removed. It amplifies noise in turbid/green-cast
images to produce confetti artifacts. Unsharp mask gives equivalent edge
sharpening with no frequency-domain instability.

Neural models (SS-UIE, WaterNet) slot in before Stage 3 when available.
"""

import cv2
import numpy as np
import torch


# ── Neural model loaders ───────────────────────────────────────────────────────

def load_ssuie(weights_path: str):
    """
    Loads SS-UIE (CVPR 2023) checkpoint.
    Tries multiple checkpoint formats: training dict, eval dict, raw state dict.
    Returns (model, device) or None on failure.
    """
    try:
        from ssuie_arch import SSUIENet
        ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
        sd   = (ckpt.get('generator') or ckpt.get('model') or
                ckpt.get('state_dict') or ckpt) if isinstance(ckpt, dict) else ckpt
        model = SSUIENet()
        model.load_state_dict(sd, strict=False)
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"SS-UIE loaded ({device})")
        return model.to(device), device
    except ImportError:
        pass
    except Exception as e:
        print(f"[warn] SS-UIE load failed: {e}")
    try:
        ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, torch.nn.Module):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"SS-UIE loaded (full module, {device})")
            return ckpt.eval().to(device), device
    except Exception as e:
        print(f"[warn] SS-UIE generic load failed: {e}")
    return None


def infer_ssuie(model_bundle, img_bgr: np.ndarray) -> np.ndarray:
    """SS-UIE forward pass. Pads to multiples of 16, removes padding after."""
    model, device = model_bundle
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w    = img_rgb.shape[:2]
    ph = (16 - h % 16) % 16; pw = (16 - w % 16) % 16
    if ph or pw:
        img_rgb = np.pad(img_rgb, ((0,ph),(0,pw),(0,0)), mode='reflect')
    t = torch.from_numpy(img_rgb.transpose(2,0,1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
    out_np = out.squeeze(0).cpu().numpy().transpose(1,2,0)[:h,:w]
    return cv2.cvtColor((np.clip(out_np,0,1)*255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def load_waternet():
    """
    Loads WaterNet via torch.hub. Smoke-tests preprocess output (must return 3
    tensors — some cached builds return 2, causing a TypeError downstream).
    Returns (preprocess, postprocess, model) or None.
    """
    try:
        pre, post, model = torch.hub.load(
            'tnwei/waternet', 'waternet', pretrained=True, verbose=False)
        model.eval()
        inputs = list(pre(np.zeros((64,64,3), dtype=np.uint8)))
        if len(inputs) != 3:
            print(f"[warn] WaterNet preprocess returned {len(inputs)} tensors — skipping.")
            return None
        print("WaterNet loaded")
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

def _guided_filter_ch(guide: np.ndarray, src: np.ndarray,
                      r: int = 8, eps: float = 0.02) -> np.ndarray:
    """Single-channel guided filter (He et al. 2013)."""
    I  = guide.astype(np.float64) / 255.0
    p  = src.astype(np.float64)   / 255.0
    mI = cv2.boxFilter(I,   cv2.CV_64F, (r,r))
    mp = cv2.boxFilter(p,   cv2.CV_64F, (r,r))
    cov = cv2.boxFilter(I*p, cv2.CV_64F, (r,r)) - mI*mp
    var = cv2.boxFilter(I*I, cv2.CV_64F, (r,r)) - mI*mI
    a   = cov / (var + eps);  b = mp - a*mI
    q   = cv2.boxFilter(a,-1,(r,r))*I + cv2.boxFilter(b,-1,(r,r))
    return np.clip(q*255, 0, 255).astype(np.uint8)


def _msr_blend(img_bgr: np.ndarray,
               base: np.ndarray,
               sigmas: list = [15, 80, 250],
               blend: float = 0.25) -> np.ndarray:
    """
    Multi-Scale Retinex blended lightly onto an already-corrected base.

    Full MSR (100% weight) overcorrects when individual channels are already
    near-white — per-channel normalisation pushes them all to 255 and loses
    colour ratios. Blending at 25% adds illumination correction benefit
    without blowout.
    """
    img_f = img_bgr.astype(np.float32) + 1.0
    msr   = np.zeros_like(img_f)
    for s in sigmas:
        L   = cv2.GaussianBlur(img_f, (0,0), s)
        msr += np.log(img_f) - np.log(L + 1.0)
    msr /= len(sigmas)
    msr_out = np.zeros_like(img_bgr, dtype=np.float32)
    for c in range(3):
        ch = msr[:,:,c]
        msr_out[:,:,c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6) * 255.0
    return np.clip(blend*msr_out + (1-blend)*base.astype(np.float32),
                   0, 255).astype(np.uint8)


# ── Master enhance function ────────────────────────────────────────────────────

def enhance(img_bgr: np.ndarray,
            ssuie_bundle=None,
            waternet_bundle=None) -> np.ndarray:
    """
    Main enhancement entry point.

    Detects cast type from channel ratios:
      blue cast  — common in open water, corrected via b-axis shift in LAB
      green cast — common in turbid/coastal water, corrected via a-axis shift

    Neural models (SS-UIE → WaterNet priority) replace Stage 3 when loaded.

    Stages 4-7 (dehazing, sharpen, denoise, saturation) apply identically
    regardless of whether neural or classical path was used.
    """
    orig   = img_bgr.copy()
    bright = float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean())
    b_m    = float(img_bgr[:,:,0].mean())
    g_m    = float(img_bgr[:,:,1].mean())
    r_m    = float(img_bgr[:,:,2].mean())

    cast_b = min((b_m / max(r_m, 1.0)) / 2.0, 1.0)
    cast_g = min((g_m / max(r_m, 1.0)) / 2.0, 1.0)
    cast   = max(cast_b, cast_g)

    # ── Stage 1: LAB chrominance correction ─────────────────────────────
    # a-axis: negative=green, positive=red  → green cast → push a up
    # b-axis: positive=yellow, negative=blue → blue cast  → push b up
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_ch, a_ch, b_ch = cv2.split(lab)
    s   = 0.55 + 0.35 * cast
    cap = 30 + int(cast * 15)
    a_ch = np.clip(a_ch + np.clip((128 - a_ch.mean()) * s, -cap,  cap), 0, 255)
    b_ch = np.clip(b_ch + np.clip((128 - b_ch.mean()) * s, -cap,  cap), 0, 255)

    # ── Stage 2: CLAHE on L + adaptive gamma ────────────────────────────
    l_u8 = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8)).apply(
               np.clip(l_ch, 0, 255).astype(np.uint8))
    base = cv2.cvtColor(
        cv2.merge([l_u8, a_ch.astype(np.uint8), b_ch.astype(np.uint8)]),
        cv2.COLOR_LAB2BGR)

    gamma = float(np.clip(0.50 + (bright/255.0)*0.40, 0.50, 0.90))
    lut   = np.array([int(((i/255.0)**gamma)*255) for i in range(256)], dtype=np.uint8)
    base  = cv2.LUT(base, lut)

    # ── Stage 3: Neural or MSR blend ────────────────────────────────────
    neural = None
    if ssuie_bundle is not None:
        try:
            neural = infer_ssuie(ssuie_bundle, img_bgr)
        except Exception as e:
            print(f"  [warn] SS-UIE inference failed: {e}")
    if neural is None and waternet_bundle is not None:
        try:
            neural = infer_waternet(waternet_bundle, img_bgr)
        except Exception as e:
            print(f"  [warn] WaterNet inference failed: {e}")

    if neural is not None:
        # Blend neural output with LAB-corrected base to get best of both
        fused = np.clip(0.60*neural.astype(np.float32) +
                        0.40*base.astype(np.float32), 0, 255).astype(np.uint8)
    else:
        # MSR blended lightly — 25% MSR + 75% corrected base
        fused = _msr_blend(img_bgr, base, blend=0.25)

    # ── Stage 4: Guided dehazing ─────────────────────────────────────────
    # Skip for very dark scenes or extreme cast (guide would be biased)
    if bright >= 80 and cast < 0.80:
        guide = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        dh    = np.stack([_guided_filter_ch(guide, fused[:,:,c]) for c in range(3)], 2)
        fused = np.clip(0.50*dh + 0.50*fused.astype(np.float32), 0, 255).astype(np.uint8)

    # ── Stage 5: Unsharp mask sharpening ────────────────────────────────
    # Controlled amount=0.35; no FFT — frequency domain amplifies noise
    # in turbid/green-cast images producing confetti artifacts
    blur  = cv2.GaussianBlur(fused.astype(np.float32), (0,0), 1.5)
    fused = np.clip(fused.astype(np.float32) + 0.35*(fused.astype(np.float32)-blur),
                    0, 255).astype(np.uint8)

    # ── Stage 6: Bilateral denoise ───────────────────────────────────────
    fused = cv2.bilateralFilter(fused, d=5, sigmaColor=25, sigmaSpace=25)

    # ── Stage 7: Saturation restore ──────────────────────────────────────
    hsv = cv2.cvtColor(fused, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * (1.15 + 0.15*cast), 0, 255)
    fused = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return fused if fused.mean() >= orig.mean() * 0.75 else orig
