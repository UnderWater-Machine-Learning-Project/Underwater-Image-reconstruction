"""
Classical underwater image preprocessing pipeline.

Stages (applied in order):
    1. Gray World White Balance  -- correct global color cast
    2. UDCP Dehazing             -- remove underwater scattering (R+G dark channel)
    3. CLAHE on L channel        -- local contrast enhancement in LAB space
    4. Unsharp Mask Sharpening   -- restore edge detail

Order rationale:
    WB first   -- DCP needs color-corrected input to estimate transmission accurately
    UDCP second -- removes haze/scattering, works best after color correction
    CLAHE third -- contrast on dehazed image, not on haze (avoids amplifying scatter)
    Sharpen last -- edges restored after contrast is finalized

This module is pure OpenCV -- no PyTorch dependency.
Used by:
    preprocess_dataset.py  -- offline dataset preparation
    enhance.py             -- inference-time preprocessing (train-inference consistency)
"""

import cv2
import numpy as np


# -- Stage 1: Gray World White Balance -----------------------------------------

def white_balance(img_bgr: np.ndarray) -> np.ndarray:
    """
    Gray World white balance.

    Assumption: the average color of a scene should be neutral gray.
    Scales each channel so its mean equals the overall image mean.
    Unconditional, deterministic, no cast-detection heuristic needed.

    This MUST run before UDCP -- otherwise the dark channel prior
    estimates transmission on a color-shifted image and produces
    artifacts (especially in blue-dominant underwater scenes).
    """
    img_f = img_bgr.astype(np.float32)
    avg_b = img_f[:, :, 0].mean()
    avg_g = img_f[:, :, 1].mean()
    avg_r = img_f[:, :, 2].mean()
    avg_all = (avg_b + avg_g + avg_r) / 3.0

    # Scale each channel to match the global mean
    img_f[:, :, 0] *= avg_all / max(avg_b, 1e-6)
    img_f[:, :, 1] *= avg_all / max(avg_g, 1e-6)
    img_f[:, :, 2] *= avg_all / max(avg_r, 1e-6)

    return np.clip(img_f, 0, 255).astype(np.uint8)


# -- Stage 2: Underwater Dark Channel Prior ------------------------------------

def udcp_dehaze(img_bgr: np.ndarray,
                omega: float = 0.85,
                patch_size: int = 15) -> np.ndarray:
    """
    Underwater Dark Channel Prior (Drews et al. 2013).

    Standard DCP uses min across all 3 RGB channels to find the dark channel.
    This fails underwater because blue is always high -- DCP reads the entire
    scene as haze and massively over-dehazes it.

    UDCP fix: compute the dark channel using only R and G channels (BGR
    indices 2 and 1). Blue is excluded because it dominates underwater and
    is not part of the scattering model in this domain.

    Extracted from enhance.py _udcp_dehaze() -- identical algorithm.
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


# -- Stage 3: CLAHE on L channel -----------------------------------------------

def clahe_enhance(img_bgr: np.ndarray,
                  clip_limit: float = 2.5,
                  grid: tuple = (8, 8)) -> np.ndarray:
    """
    CLAHE applied to the L channel in LAB color space.

    Enhances local contrast without distorting color. Includes adaptive
    gamma correction to lift dark regions.

    Runs AFTER UDCP -- enhancing contrast on a dehazed image produces
    clean results. Running CLAHE on a hazy image amplifies the haze.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # CLAHE on lightness
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    l_ch = clahe.apply(l_ch)

    # Adaptive gamma: lift dark scenes, leave bright ones alone
    brightness = float(l_ch.mean()) / 255.0
    gamma = float(np.clip(0.50 + brightness * 0.40, 0.50, 0.90))
    lut   = np.array([int(((i / 255.0) ** gamma) * 255) for i in range(256)],
                     dtype=np.uint8)
    l_ch = cv2.LUT(l_ch, lut)

    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


# -- Stage 4: Unsharp Mask Sharpening ------------------------------------------

def unsharp_mask(img_bgr: np.ndarray,
                 sigma: float = 1.5,
                 strength: float = 0.35) -> np.ndarray:
    """
    Gaussian unsharp mask for controlled edge sharpening.

    output = image + strength * (image - blurred)

    Runs last in the pipeline -- sharpening before dehazing would
    amplify haze edges. After CLAHE, edges are well-defined and
    sharpening produces clean results.

    sigma=1.5, strength=0.35 is conservative -- avoids ringing artifacts
    while restoring detail lost to underwater scattering.
    """
    img_f = img_bgr.astype(np.float32)
    blur  = cv2.GaussianBlur(img_f, (0, 0), sigma)
    sharp = img_f + strength * (img_f - blur)
    return np.clip(sharp, 0, 255).astype(np.uint8)


# -- Master pipeline -----------------------------------------------------------

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """
    Full classical preprocessing pipeline.

    WB -> UDCP -> CLAHE -> Sharpen

    Single entry point for both offline dataset prep and inference.
    Deterministic (no randomness), fast (pure OpenCV), no GPU needed.

    Args:
        img_bgr: Input BGR image (uint8, any resolution).

    Returns:
        Preprocessed BGR image (uint8, same resolution).
    """
    out = white_balance(img_bgr)
    out = udcp_dehaze(out)
    out = clahe_enhance(out)
    out = unsharp_mask(out)
    return out
