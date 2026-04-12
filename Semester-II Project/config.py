import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Folders
IMAGE_FOLDER = BASE_DIR / "images"
OUTPUT_FOLDER = BASE_DIR / "outputs"
WEIGHTS_FOLDER = BASE_DIR / "weights"

# Model Paths
FISH_MODEL_PATH = WEIGHTS_FOLDER / "fish_model.pt"
SSUIE_WEIGHTS_PATH = WEIGHTS_FOLDER / "SS_UIE.pth"

# Detection Hyperparameters
CONF_THRESHOLD = 0.15  # Aggressive recall for murky water
IOU_THRESHOLD = 0.45
TILE_SIZE = 640
OVERLAP = 0.20

# Create folders if they don't exist
for folder in [IMAGE_FOLDER, OUTPUT_FOLDER, WEIGHTS_FOLDER, 
                OUTPUT_FOLDER/"enhanced", OUTPUT_FOLDER/"detections"]:
    folder.mkdir(parents=True, exist_ok=True)