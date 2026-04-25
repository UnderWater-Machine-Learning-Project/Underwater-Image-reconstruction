"""
Underwater image enhancement pipeline.

Stages:
    1. Classical preprocessing from preprocess.py
    2. NAFNet + ViT neural enhancement, if weights are available
    3. Quality-aware fusion (colorfulness + color-shift guards)
    4. Light denoise and mild saturation guard
"""

import cv2
import numpy as np
import torch

from preprocess import preprocess as classical_preprocess


def load_model(weights_path: str):
    """Load a trained NAFNet + ViT checkpoint."""
    try:
        from nafnet_vit import nafnet_vit_small

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nafnet_vit_small().to(device)
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state, strict=True)
        model.eval()
        print(f"NAFNet loaded  ({device})  <-  {weights_path}")
        return model, device
    except Exception as e:
        print(f"[warn] NAFNet load failed: {e}")
        return None


def _infer(bundle, img_bgr: np.ndarray) -> np.ndarray:
    """
    Run NAFNet on a uint8 BGR image.

    Pads H and W to multiples of 16, runs inference, then crops the padding.
    """
    model, device = bundle
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = img_rgb.shape[:2]

    ph = (16 - h % 16) % 16
    pw = (16 - w % 16) % 16
    if ph or pw:
        img_rgb = np.pad(img_rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")

    t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)

    out_np = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)[:h, :w]
    out_u8 = (np.clip(out_np, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)


def _colorfulness(img_bgr: np.ndarray) -> float:
    """
    Hasler & Süsstrunk colorfulness metric.

    Measures how diverse and saturated the colors are. Higher = more colorful.
    Typical ranges:
        Monotone blue underwater  :  5–15
        Moderate reef scene       : 20–35
        Healthy colorful reef     : 35–60+

    Reference: Hasler, D. & Süsstrunk, S. (2003). "Measuring Colourfulness
    in Natural Images." SPIE Human Vision and Electronic Imaging.
    """
    B = img_bgr[:, :, 0].astype(np.float32)
    G = img_bgr[:, :, 1].astype(np.float32)
    R = img_bgr[:, :, 2].astype(np.float32)

    rg = R - G
    yb = 0.5 * (R + G) - B

    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std_root + 0.3 * mean_root)


def _post_process(img_bgr: np.ndarray, orig_bgr: np.ndarray) -> np.ndarray:
    """
    Light denoise plus conservative saturation correction.

    The neural model should do the restoration. This stage only smooths tiny
    compression noise and avoids the heavy HSV boost that made some outputs
    look artificially warm or over-processed.
    """
    out = cv2.bilateralFilter(img_bgr, d=3, sigmaColor=15, sigmaSpace=15)

    b_m = float(orig_bgr[:, :, 0].mean())
    g_m = float(orig_bgr[:, :, 1].mean())
    r_m = float(orig_bgr[:, :, 2].mean())
    cast = max(
        min((b_m / max(r_m, 1.0)) / 2.0, 1.0),
        min((g_m / max(r_m, 1.0)) / 2.0, 1.0),
    )

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.02 + 0.06 * cast), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _fusion_weight(neural_bgr: np.ndarray, pre_bgr: np.ndarray) -> float:
    """
    Choose how much neural output to trust.

    Uses three guards:
        1. Red-shift guard — detects when neural output adds warm/magenta cast
        2. Brightness guard — detects when neural output darkens the scene
        3. Colorfulness guard — detects when neural output washes out color
           diversity (blue-washing on reef scenes, monotone deep-water outputs)

    Returns a weight in [0.0, 0.65] for the neural branch.
    """
    n = neural_bgr.astype(np.float32).mean(axis=(0, 1)) + 1.0
    p = pre_bgr.astype(np.float32).mean(axis=(0, 1)) + 1.0

    # Guard 1: red/magenta shift
    neural_red_ratio = n[2] / max(n[1], 1.0)
    pre_red_ratio = p[2] / max(p[1], 1.0)
    red_jump = neural_red_ratio > pre_red_ratio + 0.30 and n[2] > p[2] * 1.12
    blue_drop = n[0] < p[0] * 0.88

    # Guard 2: brightness drop
    dark_drop = neural_bgr.mean() < pre_bgr.mean() * 0.88

    # Guard 3: colorfulness loss
    cf_neural = _colorfulness(neural_bgr)
    cf_pre = _colorfulness(pre_bgr)
    color_loss = cf_neural < cf_pre * 0.70          # neural lost >30% color diversity
    pre_already_good = cf_pre > 30.0                # input already has healthy color

    if red_jump and blue_drop:
        return 0.20
    if color_loss:
        # Neural is destroying color — minimal trust
        return 0.15
    if dark_drop:
        return 0.45
    if pre_already_good and cf_neural < cf_pre * 0.90:
        # Input is colorful and neural is slightly dulling it — reduce trust
        return 0.35
    if pre_already_good:
        # Input is already good — conservative blend
        return 0.45
    return 0.65


def enhance(img_bgr: np.ndarray, bundle=None) -> np.ndarray:
    """
    Enhance one BGR image.

    Returns the same resolution as the input.
    """
    orig = img_bgr.copy()
    preprocessed = classical_preprocess(img_bgr)

    neural = None
    if bundle is not None:
        try:
            neural = _infer(bundle, preprocessed)
        except Exception as e:
            print(f"  [warn] NAFNet inference failed: {e}")

    if neural is not None:
        w = _fusion_weight(neural, preprocessed)
        fused = np.clip(
            w * neural.astype(np.float32)
            + (1.0 - w) * preprocessed.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)
    else:
        fused = preprocessed

    fused = _post_process(fused, orig)

    # Fallback: if enhancement degraded brightness, use classical result
    if fused.mean() < orig.mean() * 0.75:
        return preprocessed

    # Fallback: if enhancement washed out color vs. classical preprocessing
    cf_fused = _colorfulness(fused)
    cf_pre = _colorfulness(preprocessed)
    if cf_fused < cf_pre * 0.60:
        return preprocessed

    return fused
