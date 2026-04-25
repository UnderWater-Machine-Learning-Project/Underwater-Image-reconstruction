"""
Underwater Enhancement + Detection Pipeline
============================================
Stage 1 — Classical preprocessing  (preprocess.py)
Stage 2 — NAFNet + ViT enhancement (enhance.py)
Stage 3 — Two-pass tiled YOLO      (detect.py)

Usage:
    python main.py                       # enhance then detect (default)
    python main.py --detect-on raw       # detect on raw original only
    python main.py --detect-on both      # side-by-side comparison

Output:
    outputs/<name>_compare.jpg   — hazy | enhanced | detections panel
    Terminal                     — structured detection table per image
"""

import os
import cv2
import argparse
import numpy as np

from enhance import enhance, load_model
from detect  import detect, draw, print_results


# ── Config ─────────────────────────────────────────────────────────────────────

IMAGE_FOLDER  = "images"
OUTPUT_FOLDER = "outputs"
MODEL_PATH    = "fish_model.pt"
WEIGHTS_PATH  = "weights/nafnet_final.pth"


# ── Panel builder ──────────────────────────────────────────────────────────────

def _panel(orig: np.ndarray,
           enh:  np.ndarray,
           dets_orig,
           dets_enh) -> np.ndarray:
    """
    Three-panel comparison: Raw | Enhanced | Enhanced+Detections.
    All panels resized to the same height if needed.
    """
    sep  = np.full((orig.shape[0], 6, 3), 30, dtype=np.uint8)
    raw_drawn = draw(orig, dets_orig)
    enh_drawn = draw(enh,  dets_enh)
    return np.hstack([raw_drawn, sep, enh_drawn])


# ── Per-image processing ───────────────────────────────────────────────────────

def process(path: str,
            yolo,
            bundle,
            detect_on: str = "enhanced"):

    img = cv2.imread(path)
    if img is None:
        print(f"[error] Cannot read: {path}")
        return

    filename = os.path.basename(path)
    stem     = os.path.splitext(filename)[0]

    # Always enhance (needed for the output panel even in raw-detect mode)
    enh = enhance(img, bundle=bundle)

    if detect_on == "both":
        dets_raw = detect(yolo, img)
        dets_enh = detect(yolo, enh)
        print_results(f"{filename}  [RAW]",      dets_raw)
        print_results(f"{filename}  [ENHANCED]", dets_enh)
        panel = _panel(img, enh, dets_raw, dets_enh)

    elif detect_on == "raw":
        dets = detect(yolo, img)
        print_results(filename, dets)
        panel = _panel(img, enh, dets, dets)

    else:   # "enhanced" — default
        dets = detect(yolo, enh)
        print_results(filename, dets)
        panel = _panel(img, enh, [], dets)

    out_path = os.path.join(OUTPUT_FOLDER, f"{stem}_compare.jpg")
    cv2.imwrite(out_path, panel, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(f"  saved  →  {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(
        description="Underwater Enhancement + Fish Detection Pipeline")
    parser.add_argument(
        "--detect-on",
        choices=["raw", "enhanced", "both"],
        default="enhanced",
        help="Run YOLO on raw, enhanced, or both (default: enhanced)")
    parser.add_argument(
        "--weights",
        default=WEIGHTS_PATH,
        help=f"NAFNet weights path (default: {WEIGHTS_PATH})")
    parser.add_argument(
        "--images",
        default=IMAGE_FOLDER,
        help=f"Input image folder (default: {IMAGE_FOLDER})")
    args = parser.parse_args()

    print("=" * 60)
    print("  Underwater Enhancement + Detection Pipeline")
    print("=" * 60)

    # Load models
    bundle = load_model(args.weights) if os.path.exists(args.weights) else None
    if bundle is None:
        print("[warn] No NAFNet weights found — running classical preprocessing only.")
        print(f"       Expected: {args.weights}")
        print(f"       Train first: python train.py\n")

    yolo   = YOLO(MODEL_PATH)
    sample = yolo(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)[0]

    mode = "NAFNet + ViT + classical" if bundle else "classical only (WB+UDCP+CLAHE)"
    print(f"enhancement : {mode}")
    print(f"detector    : {MODEL_PATH}")
    print(f"species     : {list(sample.names.values())}")
    print(f"detect-on   : {args.detect_on}")
    print()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    images = sorted(
        f for f in os.listdir(args.images)
        if f.lower().endswith((".jpg", ".jpeg", ".png")))

    if not images:
        print(f"[error] No images found in '{args.images}/'")
        return

    for fname in images:
        process(os.path.join(args.images, fname),
                yolo, bundle, detect_on=args.detect_on)

    print(f"\nDone — {len(images)} image(s) → '{OUTPUT_FOLDER}/'")


if __name__ == "__main__":
    main()
