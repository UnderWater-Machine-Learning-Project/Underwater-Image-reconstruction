import torch
import cv2
import numpy as np
from ultralytics import YOLO
from models.dna_net import DNANet
from torchvision import transforms
import os


# --- SETTINGS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DNA_WEIGHTS = "weights/dnanet_epoch_50.pth"
YOLO_WEIGHTS = "weights/fish_model.pt"

def run_pipeline(image_path):
    # 1. Initialize DNA-Net (Stage 1)
    dna_model = DNANet().to(DEVICE)
    dna_model.load_state_dict(torch.load(DNA_WEIGHTS, map_location=DEVICE))
    dna_model.eval()

    # 2. Initialize YOLOv8 (Stage 2)
    yolo_model = YOLO(YOLO_WEIGHTS)

    # 3. Load and Enhance Image
    print(f"🌊 Processing: {image_path}")
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        print(f"❌ Error: Could not load {image_path}")
        return
        
    img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    input_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        enhanced_tensor = dna_model(input_tensor)

    # Convert back to OpenCV format
    enhanced_img = enhanced_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    enhanced_img = (enhanced_img * 255).astype(np.uint8)
    enhanced_img_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)

    # 🚀 HIGH ACCURACY FIX: Upscale to 1024x1024 using Lanczos interpolation
    # This provides the smoothest edges for the YOLO kernels to grab onto.
    enhanced_img_bgr = cv2.resize(enhanced_img_bgr, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)

    # 4. Detection with High Sensitivity
    results = yolo_model.predict(
        source=enhanced_img_bgr, 
        conf=0.30,      # Balanced threshold for scientific validation
        iou=0.45, 
        device=DEVICE,
        augment=True,   # Test Time Augmentation
        agnostic_nms=True
    )

    # 5. Save & Show
    res_plotted = results[0].plot()
    output_name = f"detection_final_{os.path.basename(image_path)}"
    cv2.imwrite(output_name, res_plotted)
    print(f"✅ Final Detection saved as: {output_name}")

import glob

if __name__ == "__main__":
    # 1. Look inside the test_suite folder and grab all .jpg files
    test_images = glob.glob("test_suite/*.jpg")
    
    if len(test_images) == 0:
        print("⚠️ No images found! Please put some .jpg files in the 'test_suite' folder.")
    else:
        print(f"🚀 Found {len(test_images)} images. Starting Batch Processing...")
        
        # 2. Loop through every image automatically
        for img_path in test_images:
            run_pipeline(img_path)
            
        print("✅ Batch Processing Complete! Check your folder for the 'detection_final_' images.")