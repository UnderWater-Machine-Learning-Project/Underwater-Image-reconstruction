import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

class UnderwaterDataset(Dataset):
    def __init__(self, murky_dir, clear_dir, size=256):
        self.murky_path = Path(murky_dir)
        self.clear_path = Path(clear_dir)
        
        # Gather all valid image filenames
        self.image_files = [f for f in os.listdir(murky_dir) if f.endswith(('.jpg', '.png'))]
        
        # The Transformation Pipeline
        # 1. Resize: CNNs require fixed matrix dimensions. 256x256 is standard.
        # 2. ToTensor: Converts pixel arrays (0-255) into float tensors (0.0-1.0).
        #    This normalization is mathematically necessary to prevent exploding gradients.
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        # PyTorch needs to know the total size to calculate epochs
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load the images
        murky_img = cv2.imread(str(self.murky_path / img_name))
        clear_img = cv2.imread(str(self.clear_path / img_name))
        
        # OpenCV loads in BGR. We convert to RGB for standard color processing.
        murky_img = cv2.cvtColor(murky_img, cv2.COLOR_BGR2RGB)
        clear_img = cv2.cvtColor(clear_img, cv2.COLOR_BGR2RGB)
        
        # Apply the mathematical transformations
        murky_tensor = self.transform(murky_img)
        clear_tensor = self.transform(clear_img)
        
        return murky_tensor, clear_tensor