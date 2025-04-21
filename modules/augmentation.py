import cv2
from pathlib import Path
import json
from modules.utils import is_image_file, save_image

def augment_image(image):
    h, w = image.shape[:2]
    augmented = []

    # Rotation
    for angle in [90, 180, 270]:
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented.append(rotated)

    # Flips
    augmented.append(cv2.flip(image, 1))  # Horizontal
    augmented.append(cv2.flip(image, 0))  # Vertical

    # Zoom Crop
    zoomed = cv2.resize(image[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)], (w, h))
    augmented.append(zoomed)

    return augmented

def augment_and_save(src_folder, dst_folder, mapping_json="augmentation_mapping.json"):
    Path(dst_folder).mkdir(parents=True, exist_ok=True)
    mapping = []
    for img_path in Path(src_folder).rglob("*"):
        if is_image_file(img_path):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    augmented_imgs = augment_image(img)
                    for i, aug in enumerate(augmented_imgs):
                        # Save as PNG, new will be the folder it is in
                        rel_folder = img_path.parent.name
                        new_name = f"{img_path.stem}_aug{i}.png"
                        save_path = Path(dst_folder) / rel_folder
                        save_path.mkdir(parents=True, exist_ok=True)
                        save_image(aug, save_path, new_name)
                        mapping.append({
                            "original": str(img_path),
                            "augmented": str(save_path / new_name)
                        })
            except Exception as e:
                print(f"Augmentation failed for {img_path.name}: {e}")
    # Save mapping to JSON
    with open(Path(dst_folder) / mapping_json, "w") as f:
        json.dump(mapping, f, indent=2)