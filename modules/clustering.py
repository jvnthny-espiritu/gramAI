import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def extract_latents(model, dataloader, device):
    model.eval()
    latents = []
    for x, _ in dataloader:
        x = x.to(device)
        z = model.encoder(x).detach().cpu().numpy()
        latents.append(z)
    return np.concatenate(latents)

def run_clustering(latents, method="kmeans", n_clusters=5):
    if method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == "dbscan":
        clusterer = DBSCAN()
    elif method == "agglo":
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        raise ValueError("Unknown clustering method")
    labels = clusterer.fit_predict(latents)
    return labels, clusterer

def evaluate_clustering(latents, labels):
    return {
        "silhouette": silhouette_score(latents, labels),
        "davies_bouldin": davies_bouldin_score(latents, labels),
        "calinski_harabasz": calinski_harabasz_score(latents, labels)
    }