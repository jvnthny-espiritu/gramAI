import torch
import torch.nn as nn
import numpy as np

class LatentMLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super(LatentMLPClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Binary classification: Gram-positive vs Gram-negative
        )

    def forward(self, x):
        return self.classifier(x).squeeze(1)  # Output shape: (batch_size,)


def load_latents_and_labels(latents_path, labels_path):
    """
    Load and return latent vectors and binary labels from disk.
    """
    latents = np.load(latents_path)
    labels = np.load(labels_path)
    latents = torch.tensor(latents, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    return latents, labels
