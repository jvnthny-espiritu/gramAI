import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
from autoencoder import ConvAutoencoder

def load_numpy_dataset(path):
    data = np.load(path)
    data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2)  # NHWC -> NCHW
    return data

def train_autoencoder(dataset_path, save_path='autoencoder.pth', epochs=30, batch_size=64, lr=1e-3, early_stopping_patience=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)

    data = load_numpy_dataset(dataset_path)
    dataloader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    loss_history = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            imgs = batch[0].to(device)
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Early stopping
        if early_stopping_patience is not None:
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save(model.state_dict(), save_path)  # Save best model
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"⏹️ Early stopping at epoch {epoch+1}. Best loss: {best_loss:.4f} (epoch {best_epoch})")
                    break

    if early_stopping_patience is None:
        torch.save(model.state_dict(), save_path)

    print(f"✅ Autoencoder saved to {save_path}")

    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

    print(f"🥇 Best Epoch: {best_epoch}, Best Loss: {best_loss:.4f}")
