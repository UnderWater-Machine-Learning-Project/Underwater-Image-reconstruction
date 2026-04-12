import cv2
import os
from pathlib import Path
from models.scratch_enhancer import ScratchEnhancer
import config

def main():
    print("🛠️ Initializing your Scratch UDCP Model...")
    enhancer = ScratchEnhancer()

    # 2. Setup folders
    output_path = config.OUTPUT_FOLDER / "scratch_results"
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = list(config.IMAGE_FOLDER.glob("*.jpg"))
    
    print(f"🌊 Processing {len(image_files)} images through pure mathematical logic...")

    for img_path in image_files:
        # Load
        img = cv2.imread(str(img_path))
        if img is None: continue

        enhanced = enhancer.enhance(img)

        # Save
        save_name = f"scratch_{img_path.name}"
        cv2.imwrite(str(output_path / save_name), enhanced)
        print(f"✅ Saved: {save_name}")

    print(f"\n🚀 Done! Go to {output_path} to see trained model results.")

if __name__ == "__main__":
    main()