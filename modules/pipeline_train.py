import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as TF
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path = self.imgs[index][0]
        return original_tuple + (path,)


def get_transforms(img_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.70491495, 0.65667014, 0.63810917],
                             std=[0.24834319, 0.27763783, 0.22302949])
    ])


def get_dataloaders(data_dir, batch_size=32, val_split=0.0):
    dataset = ImageFolderWithPaths(data_dir, transform=get_transforms())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, None


def train_autoencoder(model, train_loader, val_loader, device, num_epochs=30, lr=3e-3, 
                      early_stopping_patience=5, early_stopping_delta=1e-4):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, _ = model(x)
            loss = criterion(x_recon, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                x = x.to(device)
                x_recon, _ = model(x)
                loss = criterion(x_recon, x)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss - early_stopping_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break

    return model, history


def extract_latents(model, dataloader, device):
    model.eval()
    latents = []
    paths = []
    with torch.no_grad():
        for x, _, path in tqdm(dataloader, desc="Extracting Latents"):
            x = x.to(device)
            _, z = model(x)
            latents.append(z.cpu())
            paths.extend(path)
    return torch.cat(latents).numpy(), paths
