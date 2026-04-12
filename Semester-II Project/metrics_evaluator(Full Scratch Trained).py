import torch
import cv2
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from models.dna_net import DNANet
from torchvision import transforms

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DNA_WEIGHTS = "weights/dnanet_epoch_50.pth"
TEST_MURKY_DIR = "datasets/Paired/trainA"
TEST_CLEAR_DIR = "datasets/Paired/trainB"
NUM_TEST_IMAGES = 10 # Increase this for a more rigorous research paper

def load_dnanet():
    model = DNANet().to(DEVICE)
    model.load_state_dict(torch.load(DNA_WEIGHTS, map_location=DEVICE))
    model.eval()
    return model

def enhance_with_dnanet(model, img_bgr):
    # Standard preprocessing
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    input_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
    
    out_img = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    out_img = (out_img * 255).astype(np.uint8)
    return cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

def compare_metrics():
    print(f"🔬 Starting Quantitative Evaluation on {DEVICE}...")
    dna_model = load_dnanet()
    
    results = {
        "Murky": {"psnr": [], "ssim": []},
        "DNA-Net": {"psnr": [], "ssim": []}
    }

    # Get sample images
    files = [f for f in os.listdir(TEST_MURKY_DIR) if f.endswith('.jpg')][:NUM_TEST_IMAGES]

    for filename in files:
        # 1. Load Ground Truth (Target) and Murky (Input)
        clear_gt = cv2.imread(os.path.join(TEST_CLEAR_DIR, filename))
        murky_in = cv2.imread(os.path.join(TEST_MURKY_DIR, filename))
        
        # Resize GT to match model output (256x256)
        clear_gt = cv2.resize(clear_gt, (256, 256))
        murky_resized = cv2.resize(murky_in, (256, 256))

        # 2. Enhance
        dna_enhanced = enhance_with_dnanet(dna_model, murky_in)

        # 3. Calculate Math Metrics
        # Baseline (Murky vs Clear)
        results["Murky"]["psnr"].append(psnr(clear_gt, murky_resized))
        results["Murky"]["ssim"].append(ssim(clear_gt, murky_resized, channel_axis=2))

        # DNA-Net (Enhanced vs Clear)
        results["DNA-Net"]["psnr"].append(psnr(clear_gt, dna_enhanced))
        results["DNA-Net"]["ssim"].append(ssim(clear_gt, dna_enhanced, channel_axis=2))

    # --- PRINT FINAL COMPARISON TABLE ---
    print("\n" + "="*45)
    print(f"{'Model':<15} | {'Avg PSNR (dB)':<15} | {'Avg SSIM':<10}")
    print("-" * 45)
    
    for model_name, metrics in results.items():
        avg_p = np.mean(metrics["psnr"])
        avg_s = np.mean(metrics["ssim"])
        print(f"{model_name:<15} | {avg_p:<15.2f} | {avg_s:<10.4f}")
    print("="*45)

if __name__ == "__main__":
    compare_metrics()