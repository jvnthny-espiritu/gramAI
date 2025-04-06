import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from autoencoder import ConvAutoencoder
from train_autoencoder import load_numpy_dataset

def extract_latents(model_path, dataset_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    data = load_numpy_dataset(dataset_path).to(device)
    with torch.no_grad():
        latents = model.encode_features(data).cpu().numpy()
    latents = latents.reshape(latents.shape[0], -1)  # Flatten for clustering
    return latents

def cluster_and_visualize(latents, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(latents)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(latents)

    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        plt.scatter(reduced[labels == i, 0], reduced[labels == i, 1], label=f"Cluster {i}")
    plt.legend()
    plt.title("K-Means Clusters in PCA-Reduced Latent Space")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.show()
    return labels