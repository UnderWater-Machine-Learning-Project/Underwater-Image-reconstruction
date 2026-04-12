import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("SUCCESS: CUDA is available!")
    print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
else:
    print("FAILED: PyTorch is defaulting to the CPU.")