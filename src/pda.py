import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

def load_dataset(npy_file):
    """Load preprocessed dataset from .npy file."""
    data = np.load(npy_file)
    print(f"✅ Loaded dataset with shape: {data.shape}")
    return data

def load_metadata(json_file):
    """Load metadata JSON file."""
    with open(json_file, "r") as f:
        metadata = json.load(f)
    return pd.DataFrame(metadata["processed_images"])

def dataset_summary(data):
    """Prints dataset summary statistics."""
    print("✅ Dataset Summary:")
    print(f"Total Images: {len(data)}")
    print(f"Image Shape: {data.shape[1:]}")
    print(f"Mean Pixel Value: {data.mean():.4f}")
    print(f"Std. Dev. of Pixel Values: {data.std():.4f}")

def plot_class_distribution(df):
    """Plot class distribution from metadata."""
    df["label"] = df["original_path"].apply(lambda x: Path(x).parent.name)
    plt.figure(figsize=(10, 5))
    sns.countplot(y=df["label"], order=df["label"].value_counts().index, palette="coolwarm")
    plt.title("Class Distribution")
    plt.xlabel("Count")
    plt.ylabel("Class")
    plt.show()

def plot_brightness_distribution(data):
    """Analyze image brightness distribution."""
    brightness_values = [np.mean(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)) for img in data]
    plt.figure(figsize=(10, 5))
    sns.histplot(brightness_values, bins=30, kde=True, color="blue")
    plt.title("Brightness Distribution")
    plt.xlabel("Brightness (Mean Pixel Value)")
    plt.ylabel("Frequency")
    plt.show()

def plot_edge_intensity(data):
    """Analyze edge intensity distribution."""
    edge_intensity = [np.mean(cv2.Sobel((img * 255).astype(np.uint8), cv2.CV_64F, 1, 1, ksize=5)) for img in data]
    plt.figure(figsize=(10, 5))
    sns.histplot(edge_intensity, bins=30, kde=True, color="red")
    plt.title("Edge Intensity Distribution")
    plt.xlabel("Edge Strength")
    plt.ylabel("Frequency")
    plt.show()

def plot_pixel_distribution(data):
    """Plot histogram of pixel values."""
    plt.hist(data.flatten(), bins=50, color="purple", alpha=0.7)
    plt.title("Pixel Value Distribution")
    plt.xlabel("Pixel Value (Normalized)")
    plt.ylabel("Frequency")
    plt.show()

def visualize_pca(data, labels):
    """Visualize feature space using PCA."""
    flat_images = data.reshape(data.shape[0], -1)
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(flat_images)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=labels, palette="coolwarm")
    plt.title("PCA Feature Space")
    plt.show()

def visualize_tsne(data, labels, sample_size=500):
    """Visualize feature space using t-SNE."""
    flat_images = data.reshape(data.shape[0], -1)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_features = tsne.fit_transform(flat_images[:sample_size])
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_features[:, 0], y=tsne_features[:, 1], hue=labels[:sample_size], palette="coolwarm")
    plt.title("t-SNE Feature Visualization")
    plt.show()

def check_data_leakage(df):
    """Check train/test split balance."""
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, stratify=df["label"], random_state=42)
    plt.figure(figsize=(10, 5))
    sns.histplot(df.loc[train_idx, "label"], label="Train", kde=True, color="blue", alpha=0.6)
    sns.histplot(df.loc[test_idx, "label"], label="Test", kde=True, color="red", alpha=0.6)
    plt.title("Train/Test Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.legend()
    plt.show()