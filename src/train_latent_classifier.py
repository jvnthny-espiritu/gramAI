import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from semi_supervised_mlp import LatentMLPClassifier, load_latents_and_labels

def train_latent_classifier(
    latents_path,
    labels_path,
    save_path='mlp_classifier.pth',
    log_path='mlp_training_log.json',
    epochs=30,
    batch_size=64,
    lr=1e-3,
    early_stopping_patience=5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latents, labels = load_latents_and_labels(latents_path, labels_path)

    # Compute class weights
    classes = np.array([0, 1])
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels.numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    dataset = TensorDataset(latents, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LatentMLPClassifier(input_dim=latents.shape[1]).to(device)
    criterion = nn.BCELoss(reduction='none')  # Weighted manually
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            x, y = batch[0].to(device), batch[1].to(device)
            preds = model(x)
            sample_weights = torch.where(y == 1, class_weights[1], class_weights[0])
            loss = (criterion(preds, y) * sample_weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.8f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}. Best loss: {best_loss:.9f}")
                break

    # Save training log
    log = {
        "best_loss": best_loss,
        "epochs_run": epoch + 1,
        "loss_curve": history
    }
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)

    # Plot loss
    plt.plot(range(1, len(history)+1), history, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Latent MLP Classifier Training Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mlp_loss_curve.png")
    plt.close()

    print(f"✅ Model saved to {save_path} | Log: {log_path} | Loss plot: mlp_loss_curve.png")
