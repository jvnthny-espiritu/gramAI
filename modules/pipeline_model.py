# modules/model_pipeline.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, calinski_harabasz_score, davies_bouldin_score


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.latent = nn.Linear(128 * 28 * 28, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, 128 * 28 * 28)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        enc = self.encoder(x)
        flat = self.flatten(enc)
        z = self.latent(flat)
        dec = self.decoder_fc(z).view(-1, 128, 28, 28)
        out = self.decoder(dec)
        return out, z


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2, dropout_p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),  # Dropout after activation
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def perform_kmeans(latent_vectors, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(latent_vectors)
    return kmeans, labels


def compute_clustering_metrics(latent_vectors, labels):
    silhouette = silhouette_score(latent_vectors, labels)
    ari = adjusted_rand_score(labels, labels)  # self-agreement
    nmi = normalized_mutual_info_score(labels, labels)
    ch = calinski_harabasz_score(latent_vectors, labels)
    db = davies_bouldin_score(latent_vectors, labels)
    return {
        "silhouette": silhouette,
        "ARI": ari,
        "NMI": nmi,
        "Calinski-Harabasz": ch,
        "Davies-Bouldin": db
    }

def check_overfitting(train_loss, val_loss, threshold=0.05):
    gap = abs(val_loss - train_loss) / train_loss
    return gap > threshold, gap
