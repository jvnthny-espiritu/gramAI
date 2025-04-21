import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from skimage.measure import shannon_entropy
from modules.utils import is_image_file

def analyze_image_folder(folder_path, output_path):
    image_paths = [img_path for img_path in Path(folder_path).rglob("*") if is_image_file(img_path)]
    if not image_paths:
        print("No valid images found.")
        return

    print(f"Total images: {len(image_paths)}")

    entropies = []
    laplacian_vars = []
    histograms = []
    mean_values = []
    std_devs = []
    invalid_images = []
    image_metadata = []
    sample_images = []

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skipping invalid image: {img_path}")
            invalid_images.append(str(img_path))
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Basic stats
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        entropy = shannon_entropy(gray)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        min_pixel = int(np.min(gray))
        max_pixel = int(np.max(gray))
        median_pixel = float(np.median(gray))
        percent_zero = float(np.sum(gray == 0) / gray.size * 100)
        percent_255 = float(np.sum(gray == 255) / gray.size * 100)
        skewness = float(((gray - mean_val)**3).mean() / (std_val**3 + 1e-8))
        kurtosis = float(((gray - mean_val)**4).mean() / (std_val**4 + 1e-8))

        # Save data for summary
        mean_values.append(mean_val)
        std_devs.append(std_val)
        entropies.append(entropy)
        laplacian_vars.append(lap_var)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        histograms.append(hist)

        # Sample for display
        if len(sample_images) < 5:
            sample_images.append((img_path.name, img))

        # Metadata
        metadata = {
            "file": str(img_path),
            "filename": img_path.name,
            "relative_path": str(img_path.relative_to(folder_path)),
            "mean": float(mean_val),
            "std": float(std_val),
            "entropy": float(entropy),
            "laplacian_var": float(lap_var),
            "min_pixel": min_pixel,
            "max_pixel": max_pixel,
            "median_pixel": median_pixel,
            "percent_zero_pixels": percent_zero,
            "percent_255_pixels": percent_255,
            "skewness": skewness,
            "kurtosis": kurtosis
        }
        image_metadata.append(metadata)

    # Print Summary
    print(f"\nANALYSIS: {Path(folder_path).name}")
    print(f"Total Samples: {len(image_metadata)}")
    print(f"Mean pixel value: {np.mean(mean_values):.2f} ± {np.std(mean_values):.2f}")
    print(f"Mean std. deviation: {np.mean(std_devs):.2f}")
    print(f"Images with very low mean (<30): {np.sum(np.array(mean_values) < 30)}")
    print(f"Images with very high mean (>220): {np.sum(np.array(mean_values) > 220)}")
    print(f"Images with very low std (<10): {np.sum(np.array(std_devs) < 10)}")
    print(f"Images with very high std (>70): {np.sum(np.array(std_devs) > 70)}")

    # Plots
    plt.hist(entropies, bins=30, color='blue', alpha=0.7)
    plt.title("Image Entropy Distribution")
    plt.xlabel("Entropy")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path / "entropy_distribution.png")
    plt.close()

    plt.hist(laplacian_vars, bins=30, color='green', alpha=0.7)
    plt.title("Laplacian Variance Distribution")
    plt.xlabel("Variance")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path / "blur_distribution.png")
    plt.close()

    avg_hist = np.mean(histograms, axis=0)
    plt.plot(avg_hist, color='black')
    plt.title("Average Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path / "avg_grayscale_histogram.png")
    plt.close()

    # Mean pixel value histogram
    plt.hist(mean_values, bins=20, alpha=0.7, color='purple')
    plt.title(f"{Path(folder_path).name} - Image Mean Pixel Value Distribution")
    plt.xlabel('Mean Pixel Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_path / "mean_pixel_distribution.png")
    plt.close()


    # Sample previews
    fig, axes = plt.subplots(2, len(sample_images), figsize=(4 * len(sample_images), 6))
    for i, (filename, img) in enumerate(sample_images):
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(filename)
        axes[1, i].hist(img.ravel(), bins=32, color='red', alpha=0.7)
        axes[1, i].set_title("Histogram")
        axes[1, i].set_xlabel("Pixel Intensity")
        axes[1, i].set_ylabel("Count")
    plt.suptitle(f"{Path(folder_path).name} - Sample Images and Histograms")
    plt.tight_layout()
    plt.savefig(output_path / "sample_previews.png")
    plt.close()

    # Save Metadata
    with open(output_path / "image_metadata.json", "w") as f:
        json.dump(image_metadata, f, indent=2)

def check_overfitting(train_metrics, val_metrics, metric_name='loss'):
    plt.plot(train_metrics, label='Train')
    plt.plot(val_metrics, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'Train vs Validation {metric_name}')
    plt.legend()
    plt.show()
