import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import silhouette_score, adjusted_rand_score
import json
from autoencoder import ConvAutoencoder


def compute_reconstruction_loss(model, dataloader, device='cpu'):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch[0].to(device)
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def compute_clustering_metrics(latents, cluster_labels, ground_truth_labels=None):
    metrics = {}

    if ground_truth_labels is not None:
        metrics["adjusted_rand_index"] = adjusted_rand_score(ground_truth_labels, cluster_labels.tolist())

    if len(set(cluster_labels)) > 1:
        metrics["silhouette_score"] = silhouette_score(latents, cluster_labels)
    else:
        metrics["silhouette_score"] = None
    return metrics


def evaluate_unsupervised(
    latents_path,
    cluster_labels_path,
    train_loader,
    val_loader,
    model_weights_path,
    log_path="unsupervised_evaluation.json"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))

    train_loss = compute_reconstruction_loss(model, train_loader, device)
    val_loss = compute_reconstruction_loss(model, val_loader, device)
    generalization_gap = abs(train_loss - val_loss)

    latents = np.load(latents_path)
    cluster_labels = np.load(cluster_labels_path)

    metrics = compute_clustering_metrics(latents, cluster_labels)
    metrics = {k: float(v) if isinstance(v, (np.float32, torch.Tensor)) else v for k, v in metrics.items()}

    metrics.update({
        "train_reconstruction_loss": train_loss,
        "val_reconstruction_loss": val_loss,
        "generalization_gap": generalization_gap
    })

    with open(log_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ Evaluation complete. Metrics saved to {log_path}")
    return metrics
