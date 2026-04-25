import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from enhance import load_model, _fusion_weight
from preprocess import preprocess
import torch

output_dir = r"C:\Users\athar\.gemini\antigravity\brain\610f40cf-09bf-4c8a-807b-9c11393c6cc8"
WEIGHTS_PATH = "weights/nafnet_final.pth"

def calculate_colorfulness(img):
    (B, G, R) = cv2.split(img.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    stdRoot = np.sqrt((np.std(rg) ** 2) + (np.std(yb) ** 2))
    meanRoot = np.sqrt((np.mean(rg) ** 2) + (np.mean(yb) ** 2))
    return stdRoot + (0.3 * meanRoot)

def generate_grid(img_name):
    hazy_path = os.path.join("images", img_name)
    if not os.path.exists(hazy_path):
        return
        
    img_hazy = cv2.imread(hazy_path)
    bundle = load_model(WEIGHTS_PATH)
    
    # 1. Preprocess
    img_preproc = preprocess(img_hazy)
    
    # 2. NAFNet
    model, device = bundle
    amp_dtype = torch.bfloat16
    
    img_rgb = cv2.cvtColor(img_preproc, cv2.COLOR_BGR2RGB)
    inp_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    
    _, _, h, w = inp_t.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h > 0 or pad_w > 0:
        inp_t = torch.nn.functional.pad(inp_t, (0, pad_w, 0, pad_h), mode='reflect')
        
    inp_t = inp_t.to(device)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
        out_t = model(inp_t)
        
    if pad_h > 0 or pad_w > 0:
        out_t = out_t[:, :, :h, :w]
        
    out_rgb = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_rgb = np.clip(out_rgb * 255.0, 0, 255).astype(np.uint8)
    img_nafnet = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    
    # 3. Fusion
    alpha = _fusion_weight(img_preproc, img_nafnet)
    img_fused = cv2.addWeighted(img_nafnet, alpha, img_preproc, 1.0 - alpha, 0)
    
    # Calculate colorfulness
    c_hazy = calculate_colorfulness(img_hazy)
    c_pre = calculate_colorfulness(img_preproc)
    c_naf = calculate_colorfulness(img_nafnet)
    c_fus = calculate_colorfulness(img_fused)
    
    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    def show_img(ax, img, title, c_val):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{title}\nColorfulness: {c_val:.1f}")
        ax.axis('off')
        
    show_img(axes[0], img_hazy, "Raw Hazy", c_hazy)
    show_img(axes[1], img_preproc, "Classical Preprocessed", c_pre)
    show_img(axes[2], img_nafnet, "Raw NAFNet (Neural)", c_naf)
    show_img(axes[3], img_fused, f"Final Fusion (alpha={alpha:.2f})", c_fus)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"grid_{img_name.split('.')[0]}.png"), dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    for img in ['test_0013.jpg', 'test2.jpg', 'test3.jpg']:
        generate_grid(img)
    print("Grids generated.")
