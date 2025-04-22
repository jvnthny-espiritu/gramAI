import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def stratified_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def plot_reconstructions(model, dataloader, device, n=5):
    model.eval()
    imgs, _ = next(iter(dataloader))
    imgs = imgs.to(device)
    with torch.no_grad():
        recons = model(imgs)
    imgs = imgs.cpu().numpy()
    recons = recons.cpu().numpy()
    for i in range(n):
        plt.subplot(2, n, i+1)
        plt.imshow(np.transpose(imgs[i], (1,2,0)))
        plt.axis('off')
        plt.subplot(2, n, i+1+n)
        plt.imshow(np.transpose(recons[i], (1,2,0)))
        plt.axis('off')
    plt.show()

def plot_clusters(latents_2d, labels):
    plt.figure(figsize=(6,6))
    plt.scatter(latents_2d[:,0], latents_2d[:,1], c=labels, cmap='tab10', s=10)
    plt.title("Latent space clustering")
    plt.show()