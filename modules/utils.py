import os
import cv2
from pathlib import Path
from PIL import Image

def is_image_file(file_path):
    return file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

def save_image(image, dst_folder, filename):
    dst_path = Path(dst_folder) / filename
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        image.save(dst_path)
    else:
        cv2.imwrite(str(dst_path), image)  # OpenCV format