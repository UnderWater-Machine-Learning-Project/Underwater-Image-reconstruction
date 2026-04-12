import cv2
import numpy as np
import os
from pathlib import Path

def calculate_uciqe(image):
    """Calculates Underwater Color Image Quality Evaluation"""
    # Convert to Lab color space to measure chroma and luminance
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1] / 255.0  # Saturation
    v = hsv[:, :, 2] / 255.0  # Brightness
    
    # Simple UCIQE approximation: combination of saturation and contrast
    con_lum = np.std(v)
    avg_sat = np.mean(s)
    center_sat = np.std(s)
    
    score = 0.4680 * center_sat + 0.2745 * con_lum + 0.2575 * avg_sat
    return round(score, 4)

def run_lab():
    input_dir = Path("../images")
    output_dir = Path("../outputs/detections")
    
    print(f"{'Image Name':<25} | {'Original UCIQE':<15} | {'Enhanced UCIQE':<15} | {'Improvement'}")
    print("-" * 75)

    for img_path in input_dir.glob("*.jpg"):
        enhanced_path = output_dir / f"enhanced_detected_{img_path.name}"
        
        if not enhanced_path.exists():
            continue
            
        img_orig = cv2.imread(str(img_path))
        img_enh = cv2.imread(str(enhanced_path))
        
        # Remove the YOLO boxes for a fair color test
        # (Ideally we'd use the 'enhanced' folder, but let's test the detections first)
        
        score_orig = calculate_uciqe(img_orig)
        score_enh = calculate_uciqe(img_enh)
        improvement = round(((score_enh - score_orig) / score_orig) * 100, 2)
        
        print(f"{img_path.name:<25} | {score_orig:<15} | {score_enh:<15} | {improvement}%")

if __name__ == "__main__":
    run_lab()