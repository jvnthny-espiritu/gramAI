import cv2
import os
from pathlib import Path
from modules.utils import is_image_file, save_image

def clean_and_resize(src_folder, dst_folder, size=(224, 224)):
    Path(dst_folder).mkdir(parents=True, exist_ok=True)
    for img_path in Path(src_folder).rglob("*"):
        if is_image_file(img_path):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    resized = cv2.resize(img, size)
                    save_image(resized, dst_folder, img_path.name)
            except Exception as e:
                print(f"Skipping {img_path.name}: {e}") 