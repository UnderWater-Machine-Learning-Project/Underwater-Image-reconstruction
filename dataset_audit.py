"""
Dataset quality audit — run once before training.
Checks: resolution consistency, dark/overexposed/blurry samples, pair alignment.
"""
import os, cv2, numpy as np
from collections import Counter

HAZY_DIR  = "dataset/hazy"
CLEAR_DIR = "dataset/clear"

exts = {".jpg", ".jpeg", ".png"}
hazy_files = sorted(f for f in os.listdir(HAZY_DIR) if os.path.splitext(f)[1].lower() in exts)

resolutions = Counter()
bad_dark, bad_bright, bad_blur, bad_size_mismatch = [], [], [], []

for i, fname in enumerate(hazy_files):
    stem = os.path.splitext(fname)[0]
    hp   = os.path.join(HAZY_DIR, fname)

    # find clear match
    cp = None
    for ext in exts:
        candidate = os.path.join(CLEAR_DIR, stem + ext)
        if os.path.exists(candidate):
            cp = candidate
            break
    if cp is None:
        continue

    hazy_img  = cv2.imread(hp)
    clear_img = cv2.imread(cp)
    if hazy_img is None or clear_img is None:
        continue

    h1, w1 = hazy_img.shape[:2]
    h2, w2 = clear_img.shape[:2]
    resolutions[(h1, w1)] += 1

    # Resolution mismatch between pair
    if (h1, w1) != (h2, w2):
        bad_size_mismatch.append((fname, (h1,w1), (h2,w2)))

    # Too dark (mean pixel < 20)
    if hazy_img.mean() < 20:
        bad_dark.append((fname, round(hazy_img.mean(), 1)))

    # Overexposed (mean pixel > 240)
    if clear_img.mean() > 240:
        bad_bright.append((fname, round(clear_img.mean(), 1)))

    # Blurry clear reference (Laplacian variance < 15)
    gray = cv2.cvtColor(clear_img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 15:
        bad_blur.append((fname, round(lap_var, 2)))

    if (i + 1) % 500 == 0:
        print(f"  scanned {i+1}/{len(hazy_files)}...")

# ── Report ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  DATASET AUDIT REPORT")
print(f"{'='*60}")
print(f"\n  Total pairs: {len(hazy_files)}")

print(f"\n  Resolution distribution (top 5):")
for (h, w), cnt in resolutions.most_common(5):
    print(f"    {w}x{h}: {cnt} images")

print(f"\n  ⚠ Size mismatches (hazy vs clear): {len(bad_size_mismatch)}")
for f, s1, s2 in bad_size_mismatch[:5]:
    print(f"    {f}: hazy={s1} clear={s2}")

print(f"\n  ⚠ Extremely dark (mean < 20): {len(bad_dark)}")
for f, m in bad_dark[:5]:
    print(f"    {f}: mean={m}")

print(f"\n  ⚠ Overexposed clear ref (mean > 240): {len(bad_bright)}")
for f, m in bad_bright[:5]:
    print(f"    {f}: mean={m}")

print(f"\n  ⚠ Blurry clear ref (laplacian var < 15): {len(bad_blur)}")
for f, m in bad_blur[:5]:
    print(f"    {f}: var={m}")

total_bad = set(f for f,*_ in bad_dark + bad_bright + bad_blur)
print(f"\n  🗑 Unique bad samples to consider removing: {len(total_bad)}")
print(f"{'='*60}")

# Save list of bad files for optional cleanup
if total_bad:
    bad_path = "dataset/bad_samples.txt"
    with open(bad_path, "w") as fp:
        for f in sorted(total_bad):
            fp.write(f + "\n")
    print(f"  Bad sample list saved → {bad_path}")
