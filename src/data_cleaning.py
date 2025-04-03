import os
import cv2
import hashlib
import shutil
from tqdm import tqdm
from pathlib import Path

def clean_image_dataset(dataset_dir: str, output_dir: str, valid_formats=("png", "jpg", "jpeg", "tif"), remove_duplicates=True):
    """
    Cleans an image dataset by:
    - Skipping corrupt or unreadable images
    - Filtering images by valid file formats
    - Skipping duplicate images (optional)
    
    Args:
        dataset_dir (str): Path to the raw image dataset.
        output_dir (str): Path to save the cleaned dataset.
        valid_formats (tuple): Allowed image formats.
        remove_duplicates (bool): Whether to skip duplicate images.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    unique_hashes = set()
    skipped_files = {"corrupt": 0, "format": 0, "duplicates": 0}

    for root, _, files in os.walk(dataset_dir):
        for file in tqdm(files, desc=f"Cleaning {Path(dataset_dir).name}"):
            file_path = os.path.join(root, file)
            file_ext = file.lower().split('.')[-1]

            # Skip if not a valid image format
            if file_ext not in valid_formats:
                skipped_files["format"] += 1
                continue

            # Try opening the image to detect corruption
            try:
                img = cv2.imread(file_path)
                if img is None or img.shape[0] == 0 or img.shape[1] == 0:
                    skipped_files["corrupt"] += 1
                    continue

                # Skip duplicates based on hash comparison
                if remove_duplicates:
                    file_hash = hashlib.md5(img.tobytes()).hexdigest()
                    if file_hash in unique_hashes:
                        skipped_files["duplicates"] += 1
                        continue
                    else:
                        unique_hashes.add(file_hash)

                # Save cleaned image to output directory while maintaining structure
                rel_path = os.path.relpath(root, dataset_dir)
                save_folder = os.path.join(output_dir, rel_path)
                os.makedirs(save_folder, exist_ok=True)
                shutil.copy(file_path, os.path.join(save_folder, file))

            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")
                skipped_files["corrupt"] += 1

    print(f"Cleaning complete! Skipped corrupt: {skipped_files['corrupt']}, invalid formats: {skipped_files['format']}, duplicates: {skipped_files['duplicates']}")