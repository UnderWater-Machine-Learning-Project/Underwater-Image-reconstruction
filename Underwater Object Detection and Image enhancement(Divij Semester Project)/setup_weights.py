import os
import torch
from ultralytics import YOLO

def finalize_environment():
    # 1. Create directory structure
    folders = ["weights", "datasets/train_murky", "datasets/train_clear", "models"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    print("🚀 Project directories verified.")

    # 2. Stage 2: YOLO11 Setup
    # Using 'yolo11n.pt' (Nano) provides the best balance of speed and 
    # VRAM overhead for your RTX desktop.
    yolo_model_name = "yolo11n.pt"
    print(f"📦 Fetching {yolo_model_name} for Stage 2 Detection...")
    
    try:
        # Initializing the model automatically handles the download
        model = YOLO(yolo_model_name)
        
        # Move weights to the dedicated folder if they were downloaded to root
        if os.path.exists(yolo_model_name):
            os.replace(yolo_model_name, f"weights/{yolo_model_name}")
            print(f"✅ Stage 2 weights secured at weights/{yolo_model_name}")
    except Exception as e:
        print(f"❌ Error during YOLO setup: {e}")

    # 3. Stage 1: Checkpoint Verification
    # This checks if you moved your manually downloaded file into the right spot.
    ckp_path = "weights/model.ckpt"
    if os.path.exists(ckp_path):
        print(f"✅ Stage 1 Checkpoint detected: {ckp_path}")
        
        # Quick integrity check: verify it is a valid torch file
        try:
            torch.load(ckp_path, map_location='cpu', weights_only=True)
            print("💎 Checkpoint integrity verified.")
        except Exception:
            print("⚠️ Checkpoint found but appears corrupted. Re-download if training fails.")
    else:
        print("🚨 CRITICAL: 'model.ckpt' NOT FOUND in /weights folder!")
        print("   Please move your downloaded file there before running training.")

    # 4. GPU Performance Check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n🖥️  Training Hardware: {gpu_name} (Ready for RTX Acceleration)")
    else:
        print("\n⚠️  CUDA not detected! The model will run slowly on CPU.")

    print("\n🎉 Setup Complete. You are ready to start training.")

if __name__ == "__main__":
    finalize_environment()