"""
Underwater Image Enhancement and Species Detection
===================================================
Three-stage pipeline:

    Stage 1 -- Classical Preprocessing (preprocess.py)
        Gray World White Balance -> UDCP Dehazing -> CLAHE -> Sharpen
        Same module used for training dataset preparation.

    Stage 2 -- Neural Enhancement (enhance.py)
        Preprocessed image -> U-Net + Swin Transformer -> Enhanced output
        Post-processing: bilateral denoise + saturation restore

    Stage 3 -- Detection (this file)
        Enhanced image -> YOLO -> Fish/species detections
        Two-pass tiled YOLO with smart cross-class NMS.

        Pass 1: full image -- catches large and medium objects.
        Pass 2: 640 px tiles at 20% overlap -- catches small fish that get
                compressed below the detection threshold during full-frame resize.

        NMS runs in two stages:
          a) Per-class NMS  -- removes tile duplicates of the same species.
          b) Cross-class NMS -- collapses generic 'fish' boxes onto overlapping
             species-specific boxes. When the model fires both 'fish' (generic)
             and 'chaetodontidae' (specific) on the same object, the specific
             label wins.

        Box validity filters:
          - Area >= MIN_BOX_AREA        -- removes sub-pixel noise
          - Area <= 20% of frame        -- removes whole-scene false positives
          - Aspect ratio <= 5:1         -- removes non-fish-shaped artifacts

Usage:
    python main.py                      # detect on enhanced (default)
    python main.py --detect-on raw      # detect on raw original
    python main.py --detect-on both     # compare raw vs enhanced detections

Output:
    outputs/<name>_compare.jpg  -- side-by-side, clean boxes + species label
    Terminal                    -- structured table per image with object metrics
"""

import cv2
import numpy as np
import os
import argparse

from enhance import enhance, load_unet, load_waternet

IMAGE_FOLDER   = "images"
OUTPUT_FOLDER  = "outputs"
MODEL_PATH     = "fish_model.pt"
UNET_WEIGHTS   = "weights/unet_final.pth"

CONF_THRESH    = 0.15
IOU_THRESH     = 0.45
IMGSZ          = 640
TILE_OVERLAP   = 0.20
MIN_BOX_AREA   = 400
MAX_FRAME_FRAC = 0.20
MAX_ASPECT     = 5.0

GENERIC_LABELS = {"fish"}
BOX_COLOR      = (0, 220, 90)
LABEL_COLOR    = (0, 0, 0)


# ── Detection ──────────────────────────────────────────────────────────────────

def _valid(x1, y1, x2, y2, fw, fh):
    bw, bh = x2 - x1, y2 - y1
    area   = bw * bh
    return (area >= MIN_BOX_AREA and
            area <= fw * fh * MAX_FRAME_FRAC and
            max(bw, bh) / max(min(bw, bh), 1) <= MAX_ASPECT)


def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = (a[2]-a[0]) * (a[3]-a[1]); ab = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (aa + ab - inter + 1e-6)


def _nms(dets):
    """
    Per-class NMS removes tile duplicates.
    Cross-class NMS collapses generic 'fish' onto overlapping species labels.
    """
    if not dets:
        return []
    labels = [d[0] for d in dets]
    scores = np.array([d[1] for d in dets], dtype=np.float32)
    boxes  = np.array([[*d[2]] for d in dets], dtype=np.float32)
    s1 = []
    for cls in set(labels):
        idx = [i for i, l in enumerate(labels) if l == cls]
        b, s = boxes[idx], scores[idx]; order = s.argsort()[::-1]
        while len(order):
            i = order[0]; s1.append(dets[idx[i]])
            if len(order) == 1: break
            xx1 = np.maximum(b[i,0], b[order[1:],0]); yy1 = np.maximum(b[i,1], b[order[1:],1])
            xx2 = np.minimum(b[i,2], b[order[1:],2]); yy2 = np.minimum(b[i,3], b[order[1:],3])
            inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
            ai = (b[i,2]-b[i,0]) * (b[i,3]-b[i,1])
            aj = (b[order[1:],2]-b[order[1:],0]) * (b[order[1:],3]-b[order[1:],1])
            order = order[1:][inter / (ai + aj - inter + 1e-6) < IOU_THRESH]
    s1.sort(key=lambda x: x[1], reverse=True)
    sup = set(); final = []
    for i, di in enumerate(s1):
        if i in sup: continue
        for j, dj in enumerate(s1):
            if j <= i or j in sup: continue
            if _iou(di[2], dj[2]) > IOU_THRESH:
                li, lj = di[0], dj[0]
                if li in GENERIC_LABELS and lj not in GENERIC_LABELS: sup.add(i); break
                elif lj in GENERIC_LABELS and li not in GENERIC_LABELS: sup.add(j)
                else: sup.add(j)
        if i not in sup: final.append(di)
    return final


def _forward(model, img, fw, fh):
    res = model(img, imgsz=IMGSZ, conf=CONF_THRESH, verbose=False)[0]
    return [(res.names[int(b.cls)], float(b.conf), tuple(map(int, b.xyxy[0])))
            for b in res.boxes if _valid(*map(int, b.xyxy[0]), fw, fh)]


def _tiles(img):
    h, w = img.shape[:2]; step = max(1, int(IMGSZ * (1 - TILE_OVERLAP)))
    seen = set()
    ys = list(range(0, h, step)); xs = list(range(0, w, step))
    if not ys or ys[-1] + IMGSZ < h: ys.append(max(0, h - IMGSZ))
    if not xs or xs[-1] + IMGSZ < w: xs.append(max(0, w - IMGSZ))
    for y in ys:
        for x in xs:
            ox = min(x, max(0, w-IMGSZ)); oy = min(y, max(0, h-IMGSZ))
            if (ox, oy) in seen: continue
            seen.add((ox, oy))
            x2, y2 = min(ox+IMGSZ, w), min(oy+IMGSZ, h)
            crop = img[oy:y2, ox:x2].copy()
            if crop.shape[0] < IMGSZ or crop.shape[1] < IMGSZ:
                pad = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
                pad[:crop.shape[0], :crop.shape[1]] = crop; crop = pad
            yield crop, ox, oy


def detect(model, img):
    """Two-pass tiled YOLO with smart cross-class NMS."""
    h, w = img.shape[:2]; all_d = []
    all_d.extend(_forward(model, img, w, h))
    if max(h, w) > IMGSZ:
        for tile, ox, oy in _tiles(img):
            for lbl, c, (tx1, ty1, tx2, ty2) in _forward(model, tile, IMGSZ, IMGSZ):
                x1 = min(ox+tx1, w); y1 = min(oy+ty1, h)
                x2 = min(ox+tx2, w); y2 = min(oy+ty2, h)
                if x2 > x1 and y2 > y1 and _valid(x1, y1, x2, y2, w, h):
                    all_d.append((lbl, c, (x1, y1, x2, y2)))
    return _nms(all_d)


# ── Metrics ────────────────────────────────────────────────────────────────────

def _metrics(x1, y1, x2, y2):
    bw, bh = x2-x1, y2-y1
    return bw*bh, round((bw+bh)/2.0, 1), round(min(bw,bh)/max(bw,bh,1), 3)


def print_results(filename, dets):
    bar = "─" * 68
    print(f"{bar}")
    print(f"  {filename}   {len(dets)} detection(s)")
    print(bar)
    if not dets:
        print("  No objects detected above threshold.")
        print(bar); return
    print(f"  {'#':<4}  {'Species':<24}  {'Conf':>5}  {'Area':>7}  {'Diam':>6}  {'Circ':>5}")
    print(f"  {'─'*4}  {'─'*24}  {'─'*5}  {'─'*7}  {'─'*6}  {'─'*5}")
    for i, (lbl, conf, (x1, y1, x2, y2)) in enumerate(dets, 1):
        area, diam, circ = _metrics(x1, y1, x2, y2)
        print(f"  {i:<4}  {lbl:<24}  {conf:>5.2f}  {area:>7}  {diam:>6.1f}  {circ:>5.3f}")
    print(bar)


# ── Visualisation ──────────────────────────────────────────────────────────────

def _draw(img, dets):
    out = img.copy()
    for lbl, _, (x1, y1, x2, y2) in dets:
        cv2.rectangle(out, (x1,y1), (x2,y2), BOX_COLOR, 2)
        (tw, th), _ = cv2.getTex# -- Pipeline -------------------------------------------------------------------

def process(path, yolo_model, unet_bundle, waternet_bundle, detect_on="enhanced"):
    img = cv2.imread(path)
    if img is None: print(f"[error] cannot read {path}"); return
    filename = os.path.basename(path)

    enh = enhance(img, unet_bundle=unet_bundle, waternet_bundle=waternet_bundle)

    if detect_on == "both":
        dets_raw = detect(yolo_model, img)
        dets_enh = detect(yolo_model, enh)
        print_results(f"{filename} [RAW]", dets_raw)
        print_results(f"{filename} [ENHANCED]", dets_enh)
        dets = dets_enh   # use enhanced for saved output
    elif detect_on == "raw":
        dets = detect(yolo_model, img)
        print_results(filename, dets)
    else:   # "enhanced" (default)
        dets = detect(yolo_model, enh)
        print_results(filename, dets)

    stem     = os.path.splitext(filename)[0]
    out_path = os.path.join(OUTPUT_FOLDER, f"{stem}_compare.jpg")
    cv2.imwrite(out_path, _comparison(img, enh, dets), [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(f"  saved  {out_path}")


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(
        description="Underwater Enhancement + Detection Pipeline")
    parser.add_argument("--detect-on", choices=["raw", "enhanced", "both"],
                        default="enhanced",
                        help="Run YOLO on raw, enhanced, or both (default: enhanced).")
    args = parser.parse_args()

    print("" + "="*60)
    print("  Underwater Enhancement + Detection Pipeline")
    print("="*60 + "")

    unet_bundle     = load_unet(UNET_WEIGHTS) if os.path.exists(UNET_WEIGHTS) else None
    waternet_bundle = load_waternet()

    mode = ("U-Net + WaterNet + classical"  if unet_bundle and waternet_bundle else
            "U-Net + classical"             if unet_bundle else
            "WaterNet + classical"          if waternet_bundle else
            "classical (MSR + UDCP)")

    yolo_model = YOLO(MODEL_PATH)
    sample     = yolo_model(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)[0]

    print(f"detector   : {MODEL_PATH}")
    print(f"species    : {list(sample.names.values())}")
    print(f"enhance    : {mode}")
    print(f"detect on  : {args.detect_on}")
    print(f"conf={CONF_THRESH}  iou={IOU_THRESH}  tile_overlap={int(TILE_OVERLAP*100)}%")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    images = sorted(f for f in os.listdir(IMAGE_FOLDER)
                    if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not images:
        print(f"no images in '{IMAGE_FOLDER}/'"); return

    for fname in images:
        process(os.path.join(IMAGE_FOLDER, fname),
                yolo_model, unet_bundle, waternet_bundle,
                detect_on=args.detect_on)

    print(f"finished -- {len(images)} image(s) -> '{OUTPUT_FOLDER}/'")


if __name__ == "__main__":
    main()ode}")
    print(f"conf={CONF_THRESH}  iou={IOU_THRESH}  tile_overlap={int(TILE_OVERLAP*100)}%")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    images = sorted(f for f in os.listdir(IMAGE_FOLDER)
                    if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not images:
        print(f"no images in '{IMAGE_FOLDER}/'"); return

    for fname in images:
        process(os.path.join(IMAGE_FOLDER, fname),
                yolo_model, unet_bundle, waternet_bundle)

    print(f"finished — {len(images)} image(s) → '{OUTPUT_FOLDER}/'")


if __name__ == "__main__":
    main()