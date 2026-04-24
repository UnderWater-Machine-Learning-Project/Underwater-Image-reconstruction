"""
Classical underwater image preprocessing pipeline.

Stages (applied in order):
    1. UDCP Dehazing             -- remove underwater scattering (R+G dark channel)
    2. Gray World White Balance  -- correct residual color cast POST-dehaze
    3. CLAHE on L channel        -- mild local contrast enhancement in LAB space
    4. Unsharp Mask Sharpening   -- light edge restoration

Order rationale:
    UDCP first  -- operates on the raw color ratios as captured; running WB before
                   UDCP distorts the R/G ratio that UDCP's dark channel relies on,
                   and WB after a big blue cast produces the pink/magenta blowout
                   seen in early experiments.
    WB second   -- corrects residual cast on the DEHAZED image; the magnitude of
                   correction is much smaller here, so Gray World stays safe.
    CLAHE third -- mild contrast lift on a clean dehazed+balanced image.
    Sharpen last -- conservative sharpening; avoids amplifying haze/noise edges.

Parameter philosophy (vs earlier aggressive settings):
    omega 0.55  (was 0.85) -- less transmission suppression; avoids washed-out look
    patch_size 7 (was 15)  -- finer dark-channel detail, less halo around objects
    t_min 0.35  (was 0.1)  -- higher floor prevents extreme brightening in dense haze
    t blur σ=3             -- smooth transmission map kills block/tiling artifacts
    clip_limit 1.2 (was 2.5) -- gentler CLAHE; was over-boosting local contrast
    gamma removed          -- adaptive gamma was lifting darks too aggressively
    sharpen σ=0.8, s=0.20  (was 1.5, 0.35) -- tighter kernel, weaker boost

This module is pure OpenCV -- no PyTorch dependency.
Used by:
    preprocess_dataset.py  -- offline dataset preparation
    enhance.py             -- inference-time preprocessing (train-inference consistency)
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Stage 1: Underwater Dark Channel Prior
# ---------------------------------------------------------------------------

def udcp_dehaze(img_bgr: np.ndarray,
                omega: float = 0.55,
                patch_size: int = 7,
                t_min: float = 0.35) -> np.ndarray:
    """
    Underwater Dark Channel Prior (Drews et al. 2013).

    Key differences from standard DCP:
    - Dark channel uses only R and G channels (excludes B) because blue
      dominates underwater and is NOT scattering haze in this domain.
    - omega=0.55: conservative dehazing strength; 0.85 was washing out textures.
    - patch_size=7: finer spatial resolution than 15; fewer halo artifacts.
    - t_min=0.35: higher transmission floor; prevents extreme scene brightening
      in very dense haze regions (which caused blown-out flippers/fins).
    - Transmission map is Gaussian-blurred (σ=3) before clamping to kill
      the tiling/block artifacts that appear at patch boundaries.

    MUST run before white_balance -- see module docstring.
    """
    img_f  = img_bgr.astype(np.float32) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))

    # Dark channel: min of R and G only (BGR indices 2 and 1)
    rg_min = np.minimum(img_f[:, :, 2], img_f[:, :, 1])
    dark   = cv2.erode(rg_min, kernel)

    # Atmospheric light: average of top 0.1% brightest dark-channel pixels
    n_px    = max(1, int(dark.size * 0.001))
    indices = np.unravel_index(np.argsort(dark.ravel())[-n_px:], dark.shape)
    A       = np.clip(img_f[indices[0], indices[1], :].mean(axis=0), 0.3, 1.0)

    # Transmission map (R+G only)
    rg_norm = np.minimum(
        img_f[:, :, 2] / max(A[2], 1e-6),
        img_f[:, :, 1] / max(A[1], 1e-6),
    )
    t = 1.0 - omega * cv2.erode(rg_norm, kernel)

    # Smooth transmission to remove block artifacts at patch boundaries
    t = cv2.GaussianBlur(t, (0, 0), 3)
    t = np.clip(t, t_min, 1.0)

    # Recover scene radiance: J = (I - A) / t + A
    out = np.empty_like(img_f)
    for c in range(3):
        out[:, :, c] = (img_f[:, :, c] - A[c]) / t + A[c]

    return np.clip(out * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage 2: Gray World White Balance
# ---------------------------------------------------------------------------

def white_balance(img_bgr: np.ndarray) -> np.ndarray:
    """
    Gray World white balance -- applied AFTER UDCP.

    Running WB on the raw hazy image causes problems:
    - The extreme blue cast in underwater raw images means avg_b >> avg_r/g.
    - Gray World then boosts R and G by a large factor.
    - After UDCP (which also adjusts channels), the net R/G amplification
      produces the pink/magenta blowout observed in early results.

    After UDCP the channel means are much more balanced, so Gray World
    applies a small corrective nudge rather than a large destructive boost.
    The scale factors are clamped to [0.7, 1.4] as an extra safeguard.
    """
    img_f   = img_bgr.astype(np.float32)
    avg_b   = float(img_f[:, :, 0].mean())
    avg_g   = float(img_f[:, :, 1].mean())
    avg_r   = float(img_f[:, :, 2].mean())
    avg_all = (avg_b + avg_g + avg_r) / 3.0

    scale_b = np.clip(avg_all / max(avg_b, 1e-6), 0.7, 1.4)
    scale_g = np.clip(avg_all / max(avg_g, 1e-6), 0.7, 1.4)
    scale_r = np.clip(avg_all / max(avg_r, 1e-6), 0.7, 1.4)

    img_f[:, :, 0] *= scale_b
    img_f[:, :, 1] *= scale_g
    img_f[:, :, 2] *= scale_r

    return np.clip(img_f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage 3: CLAHE on L channel
# ---------------------------------------------------------------------------

def clahe_enhance(img_bgr: np.ndarray,
                  clip_limit: float = 1.2,
                  grid: tuple = (8, 8)) -> np.ndarray:
    """
    Mild CLAHE on the L channel in LAB space.

    clip_limit=1.2 (was 2.5): lower clip limit means less aggressive
    redistribution; avoids the over-contrasty / over-textured appearance
    seen in early manatee outputs where skin looked like painted stone.

    Adaptive gamma correction REMOVED:
    - The gamma LUT was lifting dark areas too aggressively, compounding
      with UDCP's brightening and creating the washed-out flat look.
    - Post-UDCP images already have reasonable brightness; no gamma needed.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    l_ch  = clahe.apply(l_ch)

    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Stage 4: Unsharp Mask Sharpening
# ---------------------------------------------------------------------------

def unsharp_mask(img_bgr: np.ndarray,
                 sigma: float = 0.8,
                 strength: float = 0.20) -> np.ndarray:
    """
    Light Gaussian unsharp mask.

    output = image + strength * (image - blurred)

    sigma=0.8 (was 1.5):  tighter kernel captures only fine edges,
                           not broad mid-frequency structure.
    strength=0.20 (was 0.35): gentler boost; avoids the artificial
                           "over-sharpened painting" look on organic textures
                           (manatee skin, coral, fish scales).

    Still runs last -- edges must be well-defined (post-CLAHE) before
    sharpening; doing it earlier amplifies haze and noise gradients.
    """
    img_f = img_bgr.astype(np.float32)
    blur  = cv2.GaussianBlur(img_f, (0, 0), sigma)
    sharp = img_f + strength * (img_f - blur)
    return np.clip(sharp, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """
    Full classical preprocessing pipeline.

    UDCP -> WB -> CLAHE -> Sharpen

    Pipeline order changed from WB->UDCP to UDCP->WB:
    - UDCP on raw image uses original captured color ratios (correct).
    - WB on dehazed image corrects only the residual cast (safe, small).
    - Old order (WB first) caused pink/magenta blowout because Gray World
      applied a large R/G boost to the extreme blue cast, which was then
      compounded by UDCP's channel recovery.

    Single entry point for both offline dataset prep and inference.
    Deterministic, fast, pure OpenCV, no GPU needed.

    Args:
        img_bgr: Input BGR image (uint8, any resolution).

    Returns:
        Preprocessed BGR image (uint8, same resolution).
    """
    out = udcp_dehaze(img_bgr,  omega=0.55, patch_size=7, t_min=0.35)
    out = white_balance(out)
    out = clahe_enhance(out,    clip_limit=1.2)
    out = unsharp_mask(out,     sigma=0.8, strength=0.20)
    return out