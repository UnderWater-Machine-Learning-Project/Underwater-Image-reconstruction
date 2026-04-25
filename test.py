"""
Evaluate the full underwater image enhancement pipeline on the held-out test split.

Calculates PSNR and SSIM for the final enhanced output vs the clear ground truth.
Outputs results to weights/test_results.csv.

Usage:
    python test.py
"""

import os
import csv
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from enhance import load_model, enhance

WEIGHTS_PATH = "weights/nafnet_final.pth"
TEST_SPLIT_PATH = "weights/test_split.npy"
CLEAR_DIR = "dataset/clear"
RESULTS_CSV = "weights/test_results.csv"

def calculate_psnr(img1, img2):
    """Calculate PSNR for uint8 images [0, 255]."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim_pt(img1_bgr, img2_bgr):
    """
    Calculate SSIM using PyTorch (faster, matches train.py logic).
    Inputs are BGR uint8 numpy arrays [H, W, 3].
    """
    # Convert to RGB, [B, C, H, W], float32 [0, 1]
    t1 = torch.from_numpy(img1_bgr[:, :, ::-1].copy()).float() / 255.0
    t2 = torch.from_numpy(img2_bgr[:, :, ::-1].copy()).float() / 255.0
    t1 = t1.permute(2, 0, 1).unsqueeze(0).cuda()
    t2 = t2.permute(2, 0, 1).unsqueeze(0).cuda()
    
    window_size = 11
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu_x = F.avg_pool2d(t1, window_size, 1, window_size//2)
    mu_y = F.avg_pool2d(t2, window_size, 1, window_size//2)
    
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y
    
    sx2 = F.avg_pool2d(t1 * t1, window_size, 1, window_size//2) - mu_x2
    sy2 = F.avg_pool2d(t2 * t2, window_size, 1, window_size//2) - mu_y2
    sxy = F.avg_pool2d(t1 * t2, window_size, 1, window_size//2) - mu_xy
    
    num = (2 * mu_xy + C1) * (2 * sxy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sx2 + sy2 + C2)
    
    ssim_val = (num / (den + 1e-8)).mean().item()
    return max(0.0, min(1.0, ssim_val))


def main():
    if not os.path.exists(TEST_SPLIT_PATH):
        print(f"[error] Test split not found: {TEST_SPLIT_PATH}")
        print("        You must run train.py first to generate the test split.")
        return
        
    test_paths = np.load(TEST_SPLIT_PATH, allow_pickle=True)
    print(f"Found {len(test_paths)} test images in split.")
    
    bundle = load_model(WEIGHTS_PATH)
    if bundle is None:
        print("[error] Could not load NAFNet model.")
        return
        
    psnr_scores = []
    ssim_scores = []
    
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "psnr", "ssim"])
        
        pbar = tqdm(test_paths, desc="Evaluating Test Set")
        for hazy_path in pbar:
            filename = os.path.basename(hazy_path)
            clear_path = os.path.join(CLEAR_DIR, filename)
            
            if not os.path.exists(clear_path):
                print(f"[warn] Missing clear reference for {filename}")
                continue
                
            img_hazy = cv2.imread(str(hazy_path))
            img_clear = cv2.imread(clear_path)
            
            if img_hazy is None or img_clear is None:
                continue
                
            # Run full enhancement pipeline (Classical + NAFNet + Fusion + Guards)
            img_enh = enhance(img_hazy, bundle)
            
            # Ensure same size for metric calculation (just in case)
            if img_enh.shape != img_clear.shape:
                img_enh = cv2.resize(img_enh, (img_clear.shape[1], img_clear.shape[0]))
                
            psnr_val = calculate_psnr(img_enh, img_clear)
            ssim_val = calculate_ssim_pt(img_enh, img_clear)
            
            psnr_scores.append(psnr_val)
            ssim_scores.append(ssim_val)
            
            writer.writerow([filename, round(psnr_val, 4), round(ssim_val, 4)])
            
            pbar.set_postfix(psnr=f"{np.mean(psnr_scores):.2f}", ssim=f"{np.mean(ssim_scores):.3f}")
            
    final_psnr = np.mean(psnr_scores)
    final_ssim = np.mean(ssim_scores)
    
    print("\n" + "="*50)
    print("Final Test Set Evaluation (Full Pipeline)")
    print("="*50)
    print(f"Images Evaluated : {len(psnr_scores)}")
    print(f"Average PSNR     : {final_psnr:.3f} dB")
    print(f"Average SSIM     : {final_ssim:.4f}")
    print("="*50)
    print(f"Detailed results saved to {RESULTS_CSV}")

if __name__ == "__main__":
    main()
