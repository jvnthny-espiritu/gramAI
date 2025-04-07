from PIL import Image
import numpy as np
import os
import glob
import cv2

def preprocess_clinical_image(input_path, output_path, size=(128, 128)):
    # Load image using OpenCV (faster + easier for masking)
    img = cv2.imread(input_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to grayscale and apply threshold to find the bright circular region
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    # Find contours (outer circle)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No contours found in image: {input_path}")

    # Find the largest contour — likely the microscope field
    largest = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest)
    x, y, radius = int(x), int(y), int(radius)

    # Crop the circle and mask outside area
    mask = np.zeros_like(gray)
    cv2.circle(mask, (x, y), radius, 255, -1)
    masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

    # Crop square around the circle
    crop = masked_img[y - radius:y + radius, x - radius:x + radius]

    # Resize to desired size
    final = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)

    # Save with PIL
    Image.fromarray(final).save(output_path)

def batch_preprocess_clinical_images(input_glob, output_dir, size=(128, 128)):
    image_paths = glob.glob(input_glob)

    if not image_paths:
        raise FileNotFoundError(f"No files found for pattern: {input_glob}")

    os.makedirs(output_dir, exist_ok=True)

    for path in image_paths:
        filename = os.path.basename(path)
        output_path = os.path.join(output_dir, filename)
        preprocess_clinical_image(path, output_path, size)

    print(f"✅ Processed {len(image_paths)} images into {output_dir}")
