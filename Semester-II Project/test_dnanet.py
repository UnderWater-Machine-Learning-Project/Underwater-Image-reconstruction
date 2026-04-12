import torch
import cv2
import numpy as np
from torchvision import transforms
from models.dna_net import DNANet
import sys
import os

def test_single_image(image_path, weight_path, output_path):
    # 1. Hardware setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔬 Running inference on: {device}")

    # 2. Load the Model Architecture
    model = DNANet().to(device)
    
    # Check if weights exist before trying to load them
    if not os.path.exists(weight_path):
        print(f"❌ Error: Could not find the weights file at {weight_path}")
        print("Did your training finish? Check your 'weights' folder for the correct filename.")
        return

    # Load the "Brain" (Weights)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    
    # ⚠️ CRITICAL: Set model to evaluation mode (disables training-specific layers like dropout/batchnorm updates)
    model.eval() 

    # 3. Preprocess the Input Image
    print(f"📂 Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not load the image from {image_path}")
        return

    # Convert BGR (OpenCV standard) to RGB (PyTorch standard)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply the exact same mathematical transformations used during training
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)), # Must match the training resolution!
        transforms.ToTensor()
    ])
    
    # Add a "Batch" dimension: Model expects [Batch, Channels, Height, Width]
    # unsqueeze(0) turns [3, 256, 256] into [1, 3, 256, 256]
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    # 4. The Inference Phase
    print("🧠 Enhancing image through DNA-Net...")
    with torch.no_grad(): # Disables gradient calculation to save VRAM and speed up execution
        output_tensor = model(input_tensor)

    # 5. Post-process the Output
    # Remove the batch dimension, move off the GPU to CPU, and convert to NumPy array
    out_img = output_tensor.squeeze(0).cpu().numpy()
    
    # PyTorch format is (Channels, Height, Width). OpenCV needs (Height, Width, Channels)
    out_img = np.transpose(out_img, (1, 2, 0))
    
    # Denormalize from 0.0-1.0 float back to 0-255 integer pixels
    out_img = (out_img * 255).astype(np.uint8)
    
    # Convert RGB back to BGR so OpenCV saves the colors correctly
    final_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

    # 6. Save the Result
    cv2.imwrite(output_path, final_img)
    print(f"✅ Success! Enhanced image saved to: {output_path}")

if __name__ == "__main__":
    # 1. Check if the user provided an image path in the terminal
    if len(sys.argv) < 2:
        print("❌ Error: You forgot to specify an image path!")
        print("➡️ Usage: python test_dnanet.py <path_to_your_bad_image.jpg>")
        sys.exit(1)

    # 2. Get the input image from the terminal command
    TEST_IMAGE = sys.argv[1]
    
    # 3. Set the weights file 
    # ⚠️ IMPORTANT: Look inside your 'weights' folder and update this filename 
    # to match the highest epoch number you successfully trained!
    WEIGHT_FILE = "weights/dnanet_epoch_50.pth" 
    
    # 4. Automatically name the output file based on the input name
    image_name = os.path.basename(TEST_IMAGE)
    OUTPUT_IMAGE = f"enhanced_output_{image_name}"
    
    # Run the enhancement pipeline
    test_single_image(TEST_IMAGE, WEIGHT_FILE, OUTPUT_IMAGE)