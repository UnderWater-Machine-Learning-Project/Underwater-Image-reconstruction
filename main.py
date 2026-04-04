"""
Underwater Image Enhancement and Species Detection
===================================================
A two-stage pipeline for underwater footage analysis.

Stage 1 — Enhancement  (right panel, display only)
    LAB chrominance correction → CLAHE → bilateral filter → gamma → saturation.
    Never fed to the detector; YOLO confidence drops when trained-distribution
    is shifted by post-processing.

Stage 2 — Detection  (always on raw original)
    Two-pass tiled YOLO with smart cross-class NMS.

    Pass 1: full image — catches large and medium objects.
    Pass 2: 640 px tiles at 20 % overlap — catches small fish that get
            compressed below the detection threshold during full-frame resize.

    NMS runs in two stages:
      a) Per-class NMS  — removes tile duplicates of the same species.
      b) Cross-class NMS — collapses generic 'fish' boxes onto overlapping
         species-specific boxes. When the model fires both 'fish' (generic)
         and 'chaetodontidae' (specific) on the same object, the specific
         label wins. This eliminates the double-box artifact seen in earlier
         runs without discarding any real detections.

    Box validity filters:
      • Area ≥ MIN_BOX_AREA        — removes sub-pixel noise
      • Area ≤ 20 % of frame       — removes whole-scene false positives
      • Aspect ratio ≤ 5 : 1       — removes non-fish-shaped artifacts

Output:
    outputs/<name>_compare.jpg  — side-by-side, clean boxes + species label
    Terminal                    — structured table per image with object metrics

Author: [Your Name]
"""

import cv2
import numpy as np
import os


# ── Configuration ──────────────────────────────────────────────────────────────

IMAGE_FOLDER  = "images"
OUTPUT_FOLDER = "outputs"
MODEL_PATH    = "fish_model.pt"

CONF_THRESH    = 0.15    # lower catches small background fish; box filters handle noise
IOU_THRESH     = 0.45    # NMS IoU threshold (per-class and cross-class)
IMGSZ          = 640     # must match model training resolution
TILE_OVERLAP   = 0.20    # 20 % tile overlap — 9 passes vs 33 at 70 %
MIN_BOX_AREA   = 400     # px²  — reject sub-pixel noise detections
MAX_FRAME_FRAC = 0.20    # reject boxes occupying > 20 % of frame area
MAX_ASPECT     = 5.0     # reject boxes with width/height ratio > 5:1

# Classes the model knows are specific species (everything else is generic)
GENERIC_LABELS = {"fish"}

BOX_COLOR   = (0, 220, 90)
LABEL_COLOR = (0, 0, 0)


# ── Enhancement ────────────────────────────────────────────────────────────────

def enhance(img_bgr: np.ndarray) -> np.ndarray:
    """
    Perceptual underwater enhancement. See module docstring for design rationale.

    Operates in LAB colour space to avoid the channel-blowout artifacts
    produced by independent RGB scaling (grey-world white balance). Shifts
    the a and b chrominance axes toward neutral proportional to measured
    blue-cast severity, leaving luminance untouched until the gamma step.
    """
    orig = img_bgr.copy()

    b_mean = float(img_bgr[:, :, 0].mean())
    r_mean = float(img_bgr[:, :, 2].mean())
    cast   = min((b_mean / max(r_mean, 1.0)) / 2.0, 1.0)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_ch, a_ch, b_ch = cv2.split(lab)

    strength = 0.50 + 0.40 * cast
    a_ch = np.clip(a_ch + np.clip((128.0 - a_ch.mean()) * strength, -30, 30), 0, 255)
    b_ch = np.clip(b_ch + np.clip((128.0 - b_ch.mean()) * strength, -30, 30), 0, 255)

    l_u8 = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(
        np.clip(l_ch, 0, 255).astype(np.uint8)
    )
    img_u8 = cv2.cvtColor(
        cv2.merge([l_u8, a_ch.astype(np.uint8), b_ch.astype(np.uint8)]),
        cv2.COLOR_LAB2BGR,
    )

    img_u8 = cv2.bilateralFilter(img_u8, d=5, sigmaColor=35, sigmaSpace=35)

    brightness = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean()
    gamma = float(np.clip(0.50 + (brightness / 255.0) * 0.40, 0.50, 0.90))
    lut   = np.array([int(((i / 255.0) ** gamma) * 255) for i in range(256)], dtype=np.uint8)
    img_u8 = cv2.LUT(img_u8, lut)

    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.15 + 0.20 * cast), 0, 255)
    img_u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img_u8 if img_u8.mean() >= orig.mean() * 0.65 else orig


# ── Detection ──────────────────────────────────────────────────────────────────

def _is_valid_box(x1, y1, x2, y2, frame_w, frame_h):
    """
    Reject boxes that cannot plausibly be fish.

    Three filters applied in order:
      1. Minimum area — sub-pixel noise from low-confidence tiles.
      2. Maximum frame fraction — whole-scene false positives (dark bottom
         mass misidentified as a single large fish).
      3. Aspect ratio — extremely wide or tall boxes are structural artefacts,
         not fish shapes.
    """
    bw, bh = x2 - x1, y2 - y1
    area   = bw * bh
    if area < MIN_BOX_AREA:
        return False
    if area > frame_w * frame_h * MAX_FRAME_FRAC:
        return False
    aspect = max(bw, bh) / max(min(bw, bh), 1)
    if aspect > MAX_ASPECT:
        return False
    return True


def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-6)


def _nms(detections):
    """
    Two-stage NMS.

    Stage 1 — per-class NMS
        Standard approach: for each species independently, sort by confidence
        and suppress lower-scoring boxes with IoU > IOU_THRESH. Removes tile
        boundary duplicates of the same species.

    Stage 2 — cross-class NMS
        Resolves conflicts between generic 'fish' and specific species labels
        on the same physical object. After stage 1, any two surviving boxes
        (regardless of class) with IoU > IOU_THRESH are compared:
          • generic vs specific  → keep specific, suppress generic
          • specific vs specific → keep higher confidence
        This prevents the double-box artifact where a butterflyfish is labelled
        both 'fish' and 'chaetodontidae' simultaneously.
    """
    if not detections:
        return []

    labels = [d[0] for d in detections]
    scores = np.array([d[1] for d in detections], dtype=np.float32)
    boxes  = np.array([[*d[2]] for d in detections], dtype=np.float32)

    # Stage 1: per-class NMS
    stage1 = []
    for cls in set(labels):
        idx   = [i for i, l in enumerate(labels) if l == cls]
        b, s  = boxes[idx], scores[idx]
        order = s.argsort()[::-1]
        while len(order):
            i = order[0]
            stage1.append(detections[idx[i]])
            if len(order) == 1:
                break
            xx1   = np.maximum(b[i, 0], b[order[1:], 0])
            yy1   = np.maximum(b[i, 1], b[order[1:], 1])
            xx2   = np.minimum(b[i, 2], b[order[1:], 2])
            yy2   = np.minimum(b[i, 3], b[order[1:], 3])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            ai    = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
            aj    = (b[order[1:], 2] - b[order[1:], 0]) * (b[order[1:], 3] - b[order[1:], 1])
            iou   = inter / (ai + aj - inter + 1e-6)
            order = order[1:][iou < IOU_THRESH]

    # Stage 2: cross-class NMS
    stage1.sort(key=lambda x: x[1], reverse=True)
    suppressed = set()
    final      = []

    for i, di in enumerate(stage1):
        if i in suppressed:
            continue
        for j, dj in enumerate(stage1):
            if j <= i or j in suppressed:
                continue
            if _iou(di[2], dj[2]) > IOU_THRESH:
                li, lj = di[0], dj[0]
                if li in GENERIC_LABELS and lj not in GENERIC_LABELS:
                    suppressed.add(i); break
                elif lj in GENERIC_LABELS and li not in GENERIC_LABELS:
                    suppressed.add(j)
                else:
                    suppressed.add(j)   # lower conf (list is sorted desc)
        if i not in suppressed:
            final.append(di)

    return final


def _forward(model, img: np.ndarray, frame_w: int, frame_h: int):
    """Single YOLO forward pass with box validity filtering."""
    results = model(img, imgsz=IMGSZ, conf=CONF_THRESH, verbose=False)[0]
    out = []
    for box in results.boxes:
        lbl = results.names[int(box.cls)]
        c   = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if _is_valid_box(x1, y1, x2, y2, frame_w, frame_h):
            out.append((lbl, c, (x1, y1, x2, y2)))
    return out


def _tiles(img: np.ndarray):
    """
    Yield (crop, origin_x, origin_y) covering the image with IMGSZ tiles.
    20 % overlap ensures objects within 128 px of a tile boundary appear
    fully in an adjacent tile; NMS resolves the duplicate.
    """
    h, w  = img.shape[:2]
    step  = max(1, int(IMGSZ * (1 - TILE_OVERLAP)))
    seen: set = set()

    ys = list(range(0, h, step))
    xs = list(range(0, w, step))
    if not ys or ys[-1] + IMGSZ < h: ys.append(max(0, h - IMGSZ))
    if not xs or xs[-1] + IMGSZ < w: xs.append(max(0, w - IMGSZ))

    for y in ys:
        for x in xs:
            ox = min(x, max(0, w - IMGSZ))
            oy = min(y, max(0, h - IMGSZ))
            if (ox, oy) in seen: continue
            seen.add((ox, oy))
            x2, y2 = min(ox + IMGSZ, w), min(oy + IMGSZ, h)
            crop = img[oy:y2, ox:x2].copy()
            if crop.shape[0] < IMGSZ or crop.shape[1] < IMGSZ:
                pad = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
                pad[:crop.shape[0], :crop.shape[1]] = crop
                crop = pad
            yield crop, ox, oy


def detect(model, img: np.ndarray):
    """
    Two-pass tiled detection with smart NMS.

    Pass 1 runs on the full frame — effective for objects occupying > 3 %
    of frame area. Pass 2 tiles the image so small fish that get compressed
    to < 10 px during full-frame downsampling are detected at their native
    size within the tile. Both passes share the same box validity filters
    and feed into the unified two-stage NMS.
    """
    h, w     = img.shape[:2]
    all_dets = []

    # Pass 1: full image
    all_dets.extend(_forward(model, img, w, h))

    # Pass 2: tiles
    if max(h, w) > IMGSZ:
        for tile, ox, oy in _tiles(img):
            for lbl, c, (tx1, ty1, tx2, ty2) in _forward(model, tile, IMGSZ, IMGSZ):
                x1 = min(ox + tx1, w); y1 = min(oy + ty1, h)
                x2 = min(ox + tx2, w); y2 = min(oy + ty2, h)
                if _is_valid_box(x1, y1, x2, y2, w, h):
                    all_dets.append((lbl, c, (x1, y1, x2, y2)))

    return _nms(all_dets)


# ── Metrics ────────────────────────────────────────────────────────────────────

def _metrics(x1, y1, x2, y2):
    bw, bh   = x2 - x1, y2 - y1
    area     = bw * bh
    diameter = (bw + bh) / 2.0
    circ     = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0.0
    return area, round(diameter, 1), round(circ, 3)


def print_results(filename: str, detections: list) -> None:
    """
    Structured per-object terminal output.
    Metrics are printed here and not drawn on the image to keep
    the visual output clean and submission-ready.
    """
    bar = "─" * 68
    print(f"\n{bar}")
    print(f"  {filename}   {len(detections)} object(s) detected")
    print(bar)
    if not detections:
        print("  No objects detected above threshold.")
        print(bar)
        return
    print(f"  {'#':<4}  {'Species':<24}  {'Conf':>5}  {'Area':>7}  {'Diam':>6}  {'Circ':>5}")
    print(f"  {'─'*4}  {'─'*24}  {'─'*5}  {'─'*7}  {'─'*6}  {'─'*5}")
    for i, (lbl, conf, (x1, y1, x2, y2)) in enumerate(detections, 1):
        area, diam, circ = _metrics(x1, y1, x2, y2)
        print(f"  {i:<4}  {lbl:<24}  {conf:>5.2f}  {area:>7}  {diam:>6.1f}  {circ:>5.3f}")
    print(bar)


# ── Visualisation ──────────────────────────────────────────────────────────────

def _draw(img: np.ndarray, detections: list) -> np.ndarray:
    """
    Minimal bounding box overlay.
    Species label only — no confidence scores or metrics on the image.
    """
    out = img.copy()
    for lbl, _, (x1, y1, x2, y2) in detections:
        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        top = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, top), (x1 + tw + 6, y1), BOX_COLOR, -1)
        cv2.putText(out, lbl, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, LABEL_COLOR, 1, cv2.LINE_AA)
    return out


def _comparison(original: np.ndarray,
                enhanced: np.ndarray,
                detections: list) -> np.ndarray:
    """Raw original (left) | Enhanced (right). Same boxes on both panels."""
    sep = np.full((original.shape[0], 5, 3), 15, dtype=np.uint8)
    return np.hstack([_draw(original, detections),
                      sep,
                      _draw(enhanced, detections)])


# ── Pipeline ───────────────────────────────────────────────────────────────────

def process(path: str, model) -> None:
    img = cv2.imread(path)
    if img is None:
        print(f"[error] cannot read {path}")
        return

    filename = os.path.basename(path)
    enhanced = enhance(img)
    dets     = detect(model, img)

    print_results(filename, dets)

    stem     = os.path.splitext(filename)[0]
    out_path = os.path.join(OUTPUT_FOLDER, f"{stem}_compare.jpg")
    cv2.imwrite(out_path, _comparison(img, enhanced, dets),
                [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(f"  saved  {out_path}")


def main():
    from ultralytics import YOLO

    model  = YOLO(MODEL_PATH)
    sample = model(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)[0]

    print(f"\nmodel   : {MODEL_PATH}")
    print(f"species : {list(sample.names.values())}")
    print(
        f"conf={CONF_THRESH}  iou={IOU_THRESH}  imgsz={IMGSZ}  "
        f"tile_overlap={int(TILE_OVERLAP*100)}%  "
        f"max_frame={int(MAX_FRAME_FRAC*100)}%\n"
    )

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    images = sorted(
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not images:
        print(f"no images found in '{IMAGE_FOLDER}/'")
        return

    for fname in images:
        process(os.path.join(IMAGE_FOLDER, fname), model)

    print(f"\nfinished — {len(images)} image(s) processed → '{OUTPUT_FOLDER}/'")


if __name__ == "__main__":
    main()
