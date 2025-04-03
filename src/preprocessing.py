import os
import cv2
import numpy as np
import json
from tqdm import tqdm
from typing import Tuple
from pathlib import Path
import matplotlib.pyplot as plt

def preprocess_images(
    input_dir: str, 
    output_dir: str, 
    target_size: Tuple[int, int] = (128, 128), 
    convert_grayscale: bool = False, 
    apply_augmentation: bool = True,
    try_display: bool = False  # New parameter
):
    """
    Preprocess images and apply efficient augmentations using OpenCV.
    Args:
        input_dir (str): Path to raw images.
        output_dir (str): Path to save processed images.
        target_size (Tuple[int, int]): Target size (width, height).
        convert_grayscale (bool): Convert images to grayscale.
        apply_augmentation (bool): Apply data augmentation.
        try_display (bool): Display augmentations for testing.
    """
    metadata = {"processed_images": []}
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing {Path(input_dir).name}"):
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'tif')):
                input_path = os.path.join(root, file)

                # Maintain directory structure
                rel_path = os.path.relpath(root, input_dir)
                save_folder = os.path.join(output_dir, rel_path)
                os.makedirs(save_folder, exist_ok=True)

                try:
                    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE if convert_grayscale else cv2.IMREAD_COLOR)
                    if img is None:
                        print(f"⚠️ Skipping corrupt image: {input_path}")
                        continue

                    # Convert BGR to RGB if color
                    if not convert_grayscale:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    img = resize_image(img, target_size)
                    img = img.astype(np.float32) / 255.0  # Normalize

                    # Save original image
                    output_path = os.path.join(save_folder, file)
                    cv2.imwrite(output_path, (img * 255).astype(np.uint8))

                    metadata["processed_images"].append({
                        "filename": file,
                        "original_path": input_path,
                        "processed_path": output_path,
                        "size": target_size,
                        "grayscale": convert_grayscale,
                        "augmentation": False
                    })

                    if apply_augmentation:
                        augment_and_save(img, file, save_folder, metadata, input_path, try_display)

                except Exception as e:
                    print(f"⚠️ Error processing {input_path}: {e}")

    # Save metadata
    with open(os.path.join(output_dir, "preprocessing_log.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Preprocessing complete! Images saved in: {output_dir}")

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resizes image while preserving aspect ratio and adds padding if necessary."""
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    pad_w, pad_h = target_size[0] - new_w, target_size[1] - new_h
    padded_image = cv2.copyMakeBorder(resized, pad_h//2, pad_h - pad_h//2, 
                                      pad_w//2, pad_w - pad_w//2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def augment_and_save(image: np.ndarray, filename: str, save_folder: str, metadata: dict, original_path: str, try_display: bool = False):
    """Applies unique transformations and saves augmented images."""
    transformations = {
        "flip_h": cv2.flip(image, 1),  # Horizontal Flip
        "flip_v": cv2.flip(image, 0),  # Vertical Flip
        "rot90": cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        "rot180": cv2.rotate(image, cv2.ROTATE_180),
        "rot270": cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }

    # Additional unique augmentations
    transformations["rot90_flip_h"] = cv2.flip(transformations["rot90"], 1)
    transformations["rot180_flip_h"] = cv2.flip(transformations["rot180"], 1)
    transformations["rot270_flip_h"] = cv2.flip(transformations["rot270"], 1)
    transformations["rot90_flip_v"] = cv2.flip(transformations["rot90"], 0)
    transformations["rot270_flip_v"] = cv2.flip(transformations["rot270"], 0)

    for aug_type, aug_img in transformations.items():
        aug_filename = f"{Path(filename).stem}_{aug_type}{Path(filename).suffix}"
        aug_path = os.path.join(save_folder, aug_filename)
        cv2.imwrite(aug_path, (aug_img * 255).astype(np.uint8))

        metadata["processed_images"].append({
            "filename": aug_filename,
            "original_path": original_path,
            "processed_path": aug_path,
            "size": image.shape[:2],
            "grayscale": len(image.shape) == 2,
            "augmentation": True
        })

        if try_display:
            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
            plt.title(f"Augmentation: {aug_type}")
            plt.axis('off')  # Hide axes for better visualization
            plt.tight_layout()  # Adjust layout to avoid clipping
            plt.show()


def save_as_numpy(output_dir: str, npy_filename: str = "preprocessed_dataset.npy", convert_grayscale: bool = False):
    """Saves all preprocessed images as a NumPy array."""
    image_list, image_paths = [], []

    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'tif')):
                image_paths.append(os.path.join(root, file))

    image_paths.sort()  # Ensure consistency
    print(f"🔍 Found {len(image_paths)} images.")

    for img_path in tqdm(image_paths, desc="Loading images"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if convert_grayscale else cv2.IMREAD_COLOR)
        if img is None:
            print(f"⚠️ Skipping unreadable image: {img_path}")
            continue

        if not convert_grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0  
        if convert_grayscale:
            img = np.expand_dims(img, axis=-1)

        image_list.append(img)

    preprocessed_data = np.array(image_list, dtype=np.float32)
    
    if preprocessed_data.size == 0:
        print("⚠️ No valid images to save!")
        return

    np.save(npy_filename, preprocessed_data)
    print(f"✅ Saved dataset: {npy_filename} (Shape: {preprocessed_data.shape})")
