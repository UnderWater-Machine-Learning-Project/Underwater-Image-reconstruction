import cv2
import numpy as np
import torch
import config
from ssuie_arch import SS_UIE

def apply_lab_correction(image):
    """Stage 1: Neutralizes blue/green color casts by shifting chrominance."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Push a (green-red) and b (blue-yellow) toward neutral 128
    a = cv2.addWeighted(a, 1.2, np.full(a.shape, 128, a.dtype), -0.2, 0)
    b = cv2.addWeighted(b, 1.2, np.full(b.shape, 128, b.dtype), -0.2, 0)
    
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def apply_clahe(image):
    """Stage 2: Restores local contrast using Adaptive Histogram Equalization."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # clipLimit=2.5 prevents over-amplification of noise in turbid water
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def infer_ssuie(image):
    """Stage 3: Neural Enhancement Pass using Mamba-Spectral architecture."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SS_UIE().to(device)
    
    # Load pre-trained weights from the weights folder
    if config.SSUIE_WEIGHTS_PATH.exists():
        print(f"Loading SS-UIE weights from {config.SSUIE_WEIGHTS_PATH}...")
        model.load_state_dict(torch.load(config.SSUIE_WEIGHTS_PATH, map_location=device), strict=False)
    else:
        print("Warning: SS-UIE weights not found. Running with random initialization.")
    
    model.eval()
    
    # Pre-process: BGR to RGB, normalize [0,1], and CHW tensor conversion
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(t)
    
    # Post-process: Convert back to NumPy HWC BGR
    out_img = output.squeeze().permute(1, 2, 0).cpu().numpy()
    out_img = (out_img * 255).astype(np.uint8)
    return cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    # Load test image
    img_path = str(config.IMAGE_FOLDER / "test.jpg")
    test_img = cv2.imread(img_path)
    
    if test_img is not None:
        print(f"Processing {img_path}...")
        
        # 1. Classical Stage: Color and Contrast
        corrected = apply_lab_correction(test_img)
        classical_enhanced = apply_clahe(corrected)
        
        # 2. Neural Stage: SS-UIE Deep Learning Pass
        final_output = infer_ssuie(classical_enhanced)
        
        # Save results
        output_path = str(config.OUTPUT_FOLDER / "enhanced" / "final_reconstruction.jpg")
        cv2.imwrite(output_path, final_output)
        
        print(f"Pipeline Complete! Enhanced image saved to: {output_path}")
    else:
        print(f"Error: Could not find image at {img_path}. Please check your 'images' folder.")