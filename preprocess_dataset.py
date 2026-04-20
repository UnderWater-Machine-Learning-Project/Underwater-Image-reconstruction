"""
Offline dataset preprocessor.

Applies the classical preprocessing pipeline (preprocess.py) to every
image in dataset/hazy/ and saves results to dataset/preprocessed/.

This creates the input data for training -- the U-Net will learn to map
preprocessed images -> clear images, instead of raw hazy -> clear.

Run this ONCE before retraining:
    python preprocess_dataset.py

Output:
    dataset/preprocessed/   -- 1:1 filename match with dataset/clear/

The original dataset/hazy/ is NOT modified -- it's kept for baseline
comparison and backward compatibility with enhance.py.
"""

import os
import cv2
import argparse
from tqdm import tqdm
from preprocess import preprocess


DATASET_DIR    = "dataset"
HAZY_DIR       = os.path.join(DATASET_DIR, "hazy")
PREPROCESS_DIR = os.path.join(DATASET_DIR, "preprocessed")
EXTS           = {".jpg", ".jpeg", ".png"}


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess dataset/hazy/ images for training.")
    parser.add_argument("--force", action="store_true",
                        help="Re-process all images, even if output exists.")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG save quality (default: 95).")
    args = parser.parse_args()

    if not os.path.isdir(HAZY_DIR):
        print(f"[error] Directory not found: {HAZY_DIR}")
        print("        Expected dataset/hazy/ with degraded underwater images.")
        return

    os.makedirs(PREPROCESS_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(HAZY_DIR)
                   if os.path.splitext(f)[1].lower() in EXTS)

    if not files:
        print(f"[error] No images found in {HAZY_DIR}")
        return

    print(f"Source      : {HAZY_DIR}")
    print(f"Destination : {PREPROCESS_DIR}")
    print(f"Images      : {len(files)}")
    print(f"Force       : {args.force}")
    print()

    skipped  = 0
    processed = 0
    failed   = 0

    for fname in tqdm(files, desc="Preprocessing", unit="img"):
        src = os.path.join(HAZY_DIR, fname)
        dst = os.path.join(PREPROCESS_DIR, fname)

        # Skip if already exists (unless --force)
        if os.path.exists(dst) and not args.force:
            skipped += 1
            continue

        img = cv2.imread(src)
        if img is None:
            tqdm.write(f"  [warn] Cannot read: {src}")
            failed += 1
            continue

        result = preprocess(img)

        # Save with same extension
        ext = os.path.splitext(fname)[1].lower()
        if ext in {".jpg", ".jpeg"}:
            cv2.imwrite(dst, result, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        else:
            cv2.imwrite(dst, result)

        processed += 1

    print()
    print(f"Done.")
    print(f"  Processed : {processed}")
    print(f"  Skipped   : {skipped} (already exist)")
    print(f"  Failed    : {failed}")
    print(f"  Output    : {PREPROCESS_DIR}/")
    print()
    print(f"Next step: python train.py")
    print(f"  train.py will auto-detect dataset/preprocessed/ and use it.")


if __name__ == "__main__":
    main()
