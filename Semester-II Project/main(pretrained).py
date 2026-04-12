import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import config
from enhance import apply_lab_correction, apply_clahe, infer_ssuie


def get_tiles(image, tile_size=640, overlap=0.2):
    """
    Splits the image into overlapping tiles to catch small objects.
    """
    h, w, _ = image.shape
    stride = int(tile_size * (1 - overlap))
    tiles = []
    
    for y in range(0, h - stride, stride):
        for x in range(0, w - stride, stride):
            # Extract tile and ensure it's exactly tile_size x tile_size
            tile = image[y:y + tile_size, x:x + tile_size]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                tile = cv2.copyMakeBorder(tile, 0, tile_size - tile.shape[0], 
                                         0, tile_size - tile.shape[1], 
                                         cv2.BORDER_CONSTANT, value=(0,0,0))
            tiles.append({'img': tile, 'x': x, 'y': y})
            
    return tiles

def run_detection(image, model):
    """
    Executes Two-Pass Inference: Full Frame + Tiled.
    """
    # Pass 1: Full Frame Inference
    full_results = model.predict(image, conf=config.CONF_THRESHOLD, 
                                 iou=config.IOU_THRESHOLD, verbose=False)
    
    # Pass 2: Tiled Inference for small objects
    tiles = get_tiles(image, tile_size=config.TILE_SIZE, overlap=config.OVERLAP)
    
    # We aggregate all detections here
    all_boxes = full_results[0].boxes.data.cpu().numpy()
    
    for tile in tiles:
        tile_results = model.predict(tile['img'], conf=config.CONF_THRESHOLD, 
                                     iou=config.IOU_THRESHOLD, verbose=False)
        
        # Shift tile coordinates back to global image coordinates
        for box in tile_results[0].boxes.data.cpu().numpy():
            box[0] += tile['x'] # x1
            box[1] += tile['y'] # y1
            box[2] += tile['x'] # x2
            box[3] += tile['y'] # y2
            all_boxes = np.append(all_boxes, [box], axis=0)
            
    return all_boxes

def main():
    # 1. Initialize Model
    print("🚀 Initializing YOLOv8 Underwater Detector...")
    model = YOLO(str(config.FISH_MODEL_PATH))
    
    # 2. Get Input Images
    image_files = list(config.IMAGE_FOLDER.glob("*.jpg"))
    if not image_files:
        print(f"❌ No images found in {config.IMAGE_FOLDER}")
        return

    # 3. Process Pipeline
    for img_path in tqdm(image_files, desc="🌊 Processing Underwater Pipeline"):
        # Load Raw Image
        raw_img = cv2.imread(str(img_path))
        if raw_img is None: continue

        # --- STAGE 1: ENHANCEMENT ---
        # Sequential pipeline: LAB -> CLAHE -> SS-UIE
        corrected = apply_lab_correction(raw_img)
        clahe_img = apply_clahe(corrected)
        enhanced_img = clahe_img
        # --- STAGE 2: DETECTION ---
        # Perform Two-Pass Tiled Inference
        boxes = run_detection(enhanced_img, model)

        # 4. Reporting & Visualization
        # Note: We use the YOLO plot() utility on a copy for visualization
        res = model.predict(enhanced_img, conf=config.CONF_THRESHOLD)[0]
        final_render = res.plot() 
        
        # Save output
        save_name = f"enhanced_detected_{img_path.name}"
        cv2.imwrite(str(config.OUTPUT_FOLDER / "detections" / save_name), final_render)

    print(f"\n✅ All stages complete. Results saved in: {config.OUTPUT_FOLDER / 'detections'}")

if __name__ == "__main__":
    main()