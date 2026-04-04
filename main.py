import cv2
import numpy as np
import os
from ultralytics import YOLO

IMAGE_FOLDER = "images"
OUTPUT_FOLDER = "outputs"
MODEL_PATH = "fish_model.pt"

CONF_THRESH = 0.28
IOU_THRESH = 0.45
TILE_SIZE = 640
TILE_OVERLAP = 0.70
USE_ENHANCED_FOR_DETECTION = True
MIN_BOX_AREA = 250

TARGET_CLASSES = {"fish": (0, 255, 100)}
DEFAULT_COLOR = (0, 255, 200)

def enhance(img_bgr):
    orig = img_bgr.copy()
    img_f = img_bgr.astype(np.float32)
    means = [img_f[:, :, c].mean() for c in range(3)]
    overall = sum(means) / 3.0
    for c, m in enumerate(means):
        scale = np.clip(overall / m, 0.5, 2.0) if m > 1e-3 else 1.0
        img_f[:, :, c] = np.clip(img_f[:, :, c] * scale, 0, 255)
    img_u8 = img_f.astype(np.uint8)
    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_u8 = cv2.cvtColor(cv2.merge([l, a, b_ch]), cv2.COLOR_LAB2BGR)
    brightness = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean()
    gamma = float(np.clip(0.55 + (brightness / 255.0) * 0.35, 0.55, 0.90))
    lut = np.array([int(((i / 255.0) ** gamma) * 255) for i in range(256)], dtype=np.uint8)
    img_u8 = cv2.LUT(img_u8, lut)
    return img_u8 if img_u8.mean() >= orig.mean() * 0.7 else orig

def tile_image(img):
    h, w = img.shape[:2]
    step = max(1, int(TILE_SIZE * (1 - TILE_OVERLAP)))
    tiles = []
    seen = set()
    ys = list(range(0, h, step))
    xs = list(range(0, w, step))
    if not ys or ys[-1] + TILE_SIZE < h: ys.append(max(0, h - TILE_SIZE))
    if not xs or xs[-1] + TILE_SIZE < w: xs.append(max(0, w - TILE_SIZE))
    for y in ys:
        for x in xs:
            key = (min(x, w - TILE_SIZE), min(y, h - TILE_SIZE))
            if key in seen: continue
            seen.add(key)
            x1, y1 = key
            x2, y2 = min(x1 + TILE_SIZE, w), min(y1 + TILE_SIZE, h)
            crop = img[y1:y2, x1:x2].copy()
            if crop.shape[0] < TILE_SIZE or crop.shape[1] < TILE_SIZE:
                padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                padded[:crop.shape[0], :crop.shape[1]] = crop
                crop = padded
            tiles.append((crop, x1, y1))
    return tiles

def nms(detections):
    if not detections: return []
    labels = [d[0] for d in detections]
    scores = np.array([d[1] for d in detections], dtype=np.float32)
    boxes = np.array([[*d[2]] for d in detections], dtype=np.float32)
    kept = []
    for cls in set(labels):
        idx = [i for i, l in enumerate(labels) if l == cls]
        b = boxes[idx]
        s = scores[idx]
        order = s.argsort()[::-1]
        while len(order):
            i = order[0]
            kept.append(detections[idx[i]])
            if len(order) == 1: break
            xx1 = np.maximum(b[i, 0], b[order[1:], 0])
            yy1 = np.maximum(b[i, 1], b[order[1:], 1])
            xx2 = np.minimum(b[i, 2], b[order[1:], 2])
            yy2 = np.minimum(b[i, 3], b[order[1:], 3])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area_i = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
            area_j = ((b[order[1:], 2] - b[order[1:], 0]) * (b[order[1:], 3] - b[order[1:], 1]))
            iou = inter / (area_i + area_j - inter + 1e-6)
            order = order[1:][iou < IOU_THRESH]
    return kept

def detect(model, img):
    h, w = img.shape[:2]
    all_dets = []
    res = model(img, verbose=False, conf=CONF_THRESH, imgsz=1280)[0]
    for box in res.boxes:
        c = float(box.conf)
        lbl = res.names[int(box.cls)]
        if lbl.lower() not in ["fish"]: continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        if area < MIN_BOX_AREA: continue
        all_dets.append((lbl, c, (x1, y1, x2, y2)))
    if max(h, w) > TILE_SIZE:
        for tile, ox, oy in tile_image(img):
            res = model(tile, verbose=False, conf=CONF_THRESH, imgsz=1280)[0]
            for box in res.boxes:
                c = float(box.conf)
                lbl = res.names[int(box.cls)]
                if lbl.lower() not in ["fish"]: continue
                tx1, ty1, tx2, ty2 = map(int, box.xyxy[0])
                x1 = min(ox + tx1, w)
                y1 = min(oy + ty1, h)
                x2 = min(ox + tx2, w)
                y2 = min(oy + ty2, h)
                if x2 > x1 and y2 > y1:
                    area = (x2 - x1) * (y2 - y1)
                    if area < MIN_BOX_AREA: continue
                    all_dets.append((lbl, c, (x1, y1, x2, y2)))
    return nms(all_dets)

def draw_boxes(img, detections, alpha=0.25):
    out = img.copy()
    overlay = img.copy()
    for label, conf, (x1, y1, x2, y2) in detections:
        color = TARGET_CLASSES.get(label, DEFAULT_COLOR)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, text, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out

def put_text(img, lines, x=10, y=26, dy=22, color=(0, 255, 80)):
    for i, line in enumerate(lines):
        yy = y + i * dy
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)

def make_comparison(original, enhanced, detections):
    h = original.shape[0]
    n = len(detections)
    avg = sum(c for _, c, _ in detections) / n if n else 0.0
    by_class = {}
    for lbl, c, _ in detections:
        by_class[lbl] = by_class.get(lbl, 0) + 1
    left = draw_boxes(original, detections)
    right = draw_boxes(enhanced, detections)
    class_str = "  ".join(f"{k}:{v}" for k, v in sorted(by_class.items()))
    put_text(left, ["Original", f"{n} detections | conf {avg:.2f}", class_str or "—"], color=(0, 255, 0))
    put_text(right, ["Enhanced", f"{n} detections | conf {avg:.2f}", class_str or "—"], color=(0, 255, 150))
    put_text(left, ["Enhancement Applied"], y=h - 22, color=(0, 220, 255))
    sep = np.full((h, 5, 3), 20, dtype=np.uint8)
    return np.hstack([left, sep, right])

def process_image(path, model):
    img = cv2.imread(path)
    if img is None: return
    enhanced = enhance(img)
    detection_input = enhanced if USE_ENHANCED_FOR_DETECTION else img
    detections = detect(model, detection_input)
    comparison = make_comparison(img, enhanced, detections)
    name = os.path.splitext(os.path.basename(path))[0] + "_compare.jpg"
    out_path = os.path.join(OUTPUT_FOLDER, name)
    cv2.imwrite(out_path, comparison, [cv2.IMWRITE_JPEG_QUALITY, 93])

def main():
    model = YOLO(MODEL_PATH)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    imgs = sorted(f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    for fname in imgs:
        process_image(os.path.join(IMAGE_FOLDER, fname), model)

if __name__ == "__main__":
    main()