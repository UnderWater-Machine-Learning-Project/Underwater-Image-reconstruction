"""
Semantic alignment audit -- detects hazy/clear pairs that are NOT the same scene.

Method:
  1. Convert both to grayscale, resize to 128x128 (fast, resolution-invariant)
  2. Compute histogram correlation  -- measures global tone similarity
  3. Compute ORB keypoint match ratio -- measures structural/content alignment
  4. Flag pairs where BOTH scores are low (misaligned scene)

Also re-checks for extremely dark and overexposed images.

Output:
  dataset/misaligned_pairs.txt  -- filenames flagged for manual review
  Terminal                      -- summary stats
"""
import os, cv2, numpy as np

HAZY_DIR  = "dataset/hazy"
CLEAR_DIR = "dataset/clear"
CHECK_SIZE = 128  # resize for fast comparison

exts = {".jpg", ".jpeg", ".png"}
hazy_files = sorted(f for f in os.listdir(HAZY_DIR) if os.path.splitext(f)[1].lower() in exts)

# Thresholds
HIST_CORR_THRESHOLD = 0.25   # below = very different tone distribution
DARK_THRESHOLD      = 25     # mean pixel < this = near-black
BRIGHT_THRESHOLD    = 235    # mean pixel > this = blown out

flagged_misalign = []
flagged_dark     = []
flagged_bright   = []
hist_corrs       = []

orb = cv2.ORB_create(nfeatures=200)
bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

for i, fname in enumerate(hazy_files):
    stem = os.path.splitext(fname)[0]
    hp   = os.path.join(HAZY_DIR, fname)

    # Find clear match
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

    # -- Dark / bright check --
    if hazy_img.mean() < DARK_THRESHOLD:
        flagged_dark.append((fname, round(hazy_img.mean(), 1)))
    if clear_img.mean() > BRIGHT_THRESHOLD:
        flagged_bright.append((fname, round(clear_img.mean(), 1)))

    # -- Semantic alignment check --
    # Resize to common size
    h_sm = cv2.resize(hazy_img,  (CHECK_SIZE, CHECK_SIZE))
    c_sm = cv2.resize(clear_img, (CHECK_SIZE, CHECK_SIZE))

    h_gray = cv2.cvtColor(h_sm, cv2.COLOR_BGR2GRAY)
    c_gray = cv2.cvtColor(c_sm, cv2.COLOR_BGR2GRAY)

    # 1. Histogram correlation (global tone)
    h_hist = cv2.calcHist([h_gray], [0], None, [64], [0, 256]).flatten()
    c_hist = cv2.calcHist([c_gray], [0], None, [64], [0, 256]).flatten()
    h_hist = h_hist / (h_hist.sum() + 1e-8)
    c_hist = c_hist / (c_hist.sum() + 1e-8)
    corr = cv2.compareHist(
        h_hist.astype(np.float32), c_hist.astype(np.float32),
        cv2.HISTCMP_CORREL
    )
    hist_corrs.append(corr)

    # 2. ORB keypoint match (structural alignment)
    kp1, des1 = orb.detectAndCompute(h_gray, None)
    kp2, des2 = orb.detectAndCompute(c_gray, None)

    match_ratio = 0.0
    if des1 is not None and des2 is not None and len(des1) > 5 and len(des2) > 5:
        matches = bf.match(des1, des2)
        good = [m for m in matches if m.distance < 50]
        match_ratio = len(good) / max(len(kp1), len(kp2), 1)

    # Flag if BOTH indicators are weak
    if corr < HIST_CORR_THRESHOLD and match_ratio < 0.05:
        flagged_misalign.append((fname, round(corr, 3), round(match_ratio, 3)))

    if (i + 1) % 500 == 0:
        print(f"  scanned {i+1}/{len(hazy_files)}...")

# -- Report --
print(f"\n{'='*65}")
print(f"  SEMANTIC ALIGNMENT AUDIT")
print(f"{'='*65}")
print(f"\n  Total pairs scanned: {len(hazy_files)}")
print(f"\n  Histogram correlation stats:")
arr = np.array(hist_corrs)
print(f"    mean={arr.mean():.3f}  std={arr.std():.3f}  min={arr.min():.3f}  max={arr.max():.3f}")

print(f"\n  Misaligned pairs (low hist corr + low ORB match): {len(flagged_misalign)}")
for f, hc, mr in flagged_misalign[:10]:
    print(f"    {f}: hist_corr={hc}, orb_match={mr}")
if len(flagged_misalign) > 10:
    print(f"    ... and {len(flagged_misalign) - 10} more")

print(f"\n  Extremely dark hazy (mean < {DARK_THRESHOLD}): {len(flagged_dark)}")
for f, m in flagged_dark[:5]:
    print(f"    {f}: mean={m}")

print(f"\n  Overexposed clear (mean > {BRIGHT_THRESHOLD}): {len(flagged_bright)}")
for f, m in flagged_bright[:5]:
    print(f"    {f}: mean={m}")

total_bad = set(f for f, *_ in flagged_misalign + flagged_dark + flagged_bright)
print(f"\n  Total unique flagged samples: {len(total_bad)}")
print(f"{'='*65}")

if flagged_misalign:
    out_path = "dataset/misaligned_pairs.txt"
    with open(out_path, "w") as fp:
        for f, hc, mr in flagged_misalign:
            fp.write(f"{f}  hist_corr={hc}  orb_match={mr}\n")
    print(f"  Misaligned pair list saved -> {out_path}")

if total_bad:
    out_path2 = "dataset/all_flagged.txt"
    with open(out_path2, "w") as fp:
        for f in sorted(total_bad):
            fp.write(f + "\n")
    print(f"  All flagged list saved -> {out_path2}")
    print(f"\n  >> Review these files manually, then run:")
    print(f"     python -c \"import os; [os.remove(os.path.join(d,f)) for f in open('dataset/all_flagged.txt').read().split() for d in ['dataset/hazy','dataset/clear'] if os.path.exists(os.path.join(d,f))]\"")
else:
    print(f"\n  Dataset is CLEAN. No misaligned or bad samples detected.")
