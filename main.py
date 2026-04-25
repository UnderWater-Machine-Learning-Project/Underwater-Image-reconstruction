"""
Underwater Enhancement + Detection Pipeline
============================================

Usage:
    python main.py
    python main.py --detect-on raw
    python main.py --detect-on both
    python main.py --output outputs_baseline

Output panel:
    Raw | Enhanced | Enhanced + detections
"""

import argparse
import os

import cv2
import numpy as np

from detect import detect, draw, print_results
from enhance import enhance, load_model


IMAGE_FOLDER = "images"
OUTPUT_FOLDER = "outputs"
MODEL_PATH = "fish_model.pt"
WEIGHTS_PATH = "weights/nafnet_final.pth"


def _panel(orig: np.ndarray, enh: np.ndarray, dets_orig, dets_enh) -> np.ndarray:
    """Three-panel comparison: Raw | Enhanced | Enhanced+Detections."""
    sep = np.full((orig.shape[0], 6, 3), 30, dtype=np.uint8)
    raw_panel = draw(orig, dets_orig) if dets_orig else orig.copy()
    enh_clean = enh.copy()
    enh_drawn = draw(enh, dets_enh) if dets_enh else enh.copy()
    return np.hstack([raw_panel, sep, enh_clean, sep, enh_drawn])


def process(path: str, yolo, bundle, detect_on: str = "enhanced",
            output_folder: str = OUTPUT_FOLDER):
    img = cv2.imread(path)
    if img is None:
        print(f"[error] Cannot read: {path}")
        return

    filename = os.path.basename(path)
    stem = os.path.splitext(filename)[0]

    enh = enhance(img, bundle=bundle)

    if detect_on == "both":
        dets_raw = detect(yolo, img)
        dets_enh = detect(yolo, enh)
        print_results(f"{filename}  [RAW]", dets_raw)
        print_results(f"{filename}  [ENHANCED]", dets_enh)
        panel = _panel(img, enh, dets_raw, dets_enh)
    elif detect_on == "raw":
        dets_raw = detect(yolo, img)
        print_results(filename, dets_raw)
        panel = _panel(img, enh, dets_raw, [])
    else:
        dets_enh = detect(yolo, enh)
        print_results(filename, dets_enh)
        panel = _panel(img, enh, [], dets_enh)

    out_path = os.path.join(output_folder, f"{stem}_compare.jpg")
    cv2.imwrite(out_path, panel, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(f"  saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Underwater Enhancement + Fish Detection Pipeline")
    parser.add_argument(
        "--detect-on",
        choices=["raw", "enhanced", "both"],
        default="enhanced",
        help="Run YOLO on raw, enhanced, or both")
    parser.add_argument(
        "--weights",
        default=WEIGHTS_PATH,
        help=f"NAFNet weights path (default: {WEIGHTS_PATH})")
    parser.add_argument(
        "--images",
        default=IMAGE_FOLDER,
        help=f"Input image folder (default: {IMAGE_FOLDER})")
    parser.add_argument(
        "--output",
        default=OUTPUT_FOLDER,
        help=f"Output folder (default: {OUTPUT_FOLDER})")
    args = parser.parse_args()

    yolo_config_dir = os.path.join(os.getcwd(), "Ultralytics")
    os.makedirs(yolo_config_dir, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", yolo_config_dir)
    from ultralytics import YOLO

    print("=" * 60)
    print("  Underwater Enhancement + Detection Pipeline")
    print("=" * 60)

    bundle = load_model(args.weights) if os.path.exists(args.weights) else None
    if bundle is None:
        print("[warn] No NAFNet weights found; running classical preprocessing only.")
        print(f"       Expected: {args.weights}")
        print("       Train first: python train.py\n")

    yolo = YOLO(MODEL_PATH)
    sample = yolo(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)[0]

    mode = "NAFNet + ViT + classical" if bundle else "classical only"
    print(f"enhancement : {mode}")
    print(f"detector    : {MODEL_PATH}")
    print(f"species     : {list(sample.names.values())}")
    print(f"detect-on   : {args.detect_on}")
    print(f"output      : {args.output}")
    print()

    os.makedirs(args.output, exist_ok=True)

    images = sorted(
        f for f in os.listdir(args.images)
        if f.lower().endswith((".jpg", ".jpeg", ".png")))

    if not images:
        print(f"[error] No images found in '{args.images}/'")
        return

    for fname in images:
        process(
            os.path.join(args.images, fname),
            yolo,
            bundle,
            detect_on=args.detect_on,
            output_folder=args.output,
        )

    print(f"\nDone - {len(images)} image(s) -> '{args.output}/'")


if __name__ == "__main__":
    main()
