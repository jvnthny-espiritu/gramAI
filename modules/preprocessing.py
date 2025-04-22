import os
import cv2
import numpy as np
import json

def resize_image(image, size=(128, 128)):
    return cv2.resize(image, size)

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def normalize_image(image):
    return image.astype(np.float32) / 255.0

def get_image_metadata(image, filename):
    return {
        "filename": filename,
        "shape": image.shape,
        "dtype": str(image.dtype),
        "mean": float(np.mean(image)),
        "std": float(np.std(image)),
        "min": float(np.min(image)),
        "max": float(np.max(image))
    }

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    image = resize_image(image)
    image = apply_clahe(image)
    image = denoise_image(image)
    image = normalize_image(image)
    
    return image

def save_image(image, save_path):
    image_to_save = (image * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(save_path, image_to_save)

def process_cleaned_files(valid_files, directory_path, output_directory, save_metadata_json=True):
    os.makedirs(output_directory, exist_ok=True)
    processed_images = []
    metadata = []
    folder_name = os.path.basename(os.path.normpath(directory_path))

    for idx, filename in enumerate(valid_files, 1):
        image_path = os.path.join(directory_path, filename)
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            continue
        processed_image = preprocess_image(image_path)
        processed_images.append((filename, processed_image))
        save_filename = f"{folder_name}_{idx:03d}.png"
        save_path = os.path.join(output_directory, save_filename)
        save_image(processed_image, save_path)
        proc_img_uint8 = (processed_image * 255).clip(0, 255).astype(np.uint8)
        metadata.append({
            "before": get_image_metadata(raw_img, filename),
            "after": get_image_metadata(proc_img_uint8, save_filename)
        })
    if save_metadata_json:
        meta_path = os.path.join(output_directory, f"{folder_name}_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
    return processed_images