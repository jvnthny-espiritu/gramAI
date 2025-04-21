import random
import shutil
from pathlib import Path
import cv2
from modules.utils import is_image_file

def resize_image(image_path, output_path, size=(224, 224)):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to read {image_path}")
        return
    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    # Always save as PNG
    output_path = output_path.with_suffix('.png')
    cv2.imwrite(str(output_path), resized)

def split_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    resize: bool = True,
    size=(224, 224)
):
    random.seed(seed)
    images = sorted([p for p in source_dir.iterdir() if is_image_file(p)])
    random.shuffle(images)

    n_train = int(len(images) * train_ratio)
    train_images = images[:n_train]
    val_images = images[n_train:]

    for split_name, split_images in zip(["train", "val"], [train_images, val_images]):
        split_path = output_dir / split_name
        split_path.mkdir(parents=True, exist_ok=True)
        for img_path in split_images:
            out_img_path = split_path / (img_path.stem + ".png")
            if resize:
                resize_image(img_path, out_img_path, size=size)
            else:
                shutil.copy(img_path, out_img_path)

    print(f"Split complete: {len(train_images)} train, {len(val_images)} val")

def move_clinical_to_test(source_dir: Path, output_dir: Path, resize: bool = True, size=(224, 224)):
    test_path = output_dir / "test"
    test_path.mkdir(parents=True, exist_ok=True)

    for img_path in source_dir.iterdir():
        if is_image_file(img_path):
            out_img_path = test_path / (img_path.stem + ".png")
            if resize:
                resize_image(img_path, out_img_path, size=size)
            else:
                shutil.copy(img_path, out_img_path)

    print(f"Moved {len([p for p in source_dir.iterdir() if is_image_file(p)])} images to test set.")