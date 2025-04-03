import os
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

# ------------------------------
# Load Image Metadata (Optimized)
# ------------------------------
def load_image_metadata(image_dir: str) -> pd.DataFrame:
    """
    Extract metadata from images such as dimensions, format, and file size.
    Uses multithreading for speed optimization.
    """
    data = []
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif'))]

    def process_image(f):
        path = os.path.join(image_dir, f)
        try:
            with Image.open(path) as img:
                return {
                    "filename": f,
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "size_kb": os.path.getsize(path) / 1024
                }
        except Exception:
            return None

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_image, image_files))

    return pd.DataFrame([res for res in results if res is not None])

# ------------------------------
# Detect Corrupt Images (Optimized)
# ------------------------------
def detect_corrupt_images(image_dir: str) -> List[str]:
    """
    Identify corrupt images that cannot be opened, using cv2 for better handling.
    """
    corrupt_files = []

    def check_image(f):
        path = os.path.join(image_dir, f)
        try:
            img = cv2.imread(path)
            if img is None:
                return f
        except:
            return f
        return None

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif'))]
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(check_image, image_files))

    return [f for f in results if f is not None]

# ------------------------------
# Plot Pixel Intensity Distribution (Optimized)
# ------------------------------
def plot_pixel_intensity_distribution(image_dir: str, num_samples: int = 5) -> None:
    """
    Plot histogram of pixel intensity distribution for randomly selected images.
    """
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif'))]
    if not image_files:
        print("No image files found in the directory.")
        return

    sample_files = random.sample(image_files, min(num_samples, len(image_files)))

    for f in sample_files:
        path = os.path.join(image_dir, f)
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            plt.figure(figsize=(6, 4))
            sns.histplot(img.flatten(), bins=50, color='blue', kde=True)
            plt.title(f"Pixel Intensity Distribution - {f}")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.show()
        except Exception as e:
            print(f"Error processing file {f}: {e}")

# ------------------------------
# Extract Features for t-SNE (Optimized)
# ------------------------------
def extract_features_for_tsne(image_dir: str, num_samples: int = 100) -> np.ndarray:
    """
    Extract image features using PCA for t-SNE visualization.
    """
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif'))]
    if not image_files:
        print("No image files found in the directory.")
        return np.array([])

    sample_files = random.sample(image_files, min(num_samples, len(image_files)))

    def process_image(f):
        path = os.path.join(image_dir, f)
        img = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_2)
        if img is None:
            return None
        return cv2.resize(img, (64, 64)).flatten()

    with ThreadPoolExecutor() as executor:
        images = list(executor.map(process_image, sample_files))

    images = np.array([img for img in images if img is not None])
    print(images.shape)

    if images.size == 0:
        print("No valid images processed.")
        return np.array([])

    min_components = min(images.shape[0], images.shape[1])
    pca = PCA(n_components=min(50, min_components)).fit_transform(images)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(pca)

    return tsne_result

# ------------------------------
# Plot t-SNE Clusters (Optimized)
# ------------------------------
def plot_tsne_clusters(image_dir: str, num_samples: int = 100) -> None:
    """
    Plot t-SNE visualization of image dataset clusters.
    """
    embeddings = extract_features_for_tsne(image_dir, num_samples)

    if embeddings.size == 0:
        print("No embeddings to plot.")
        return

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], alpha=0.7)
    plt.title("t-SNE Visualization of Image Dataset")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()
