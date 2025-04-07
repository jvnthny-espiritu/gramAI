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
    val_data=None,  # (X_val, y_val)
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
    classes = np.unique(labels.numpy())
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels.numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(latents, labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optional validation loader
    val_loader = None
    if val_data is not None:
        X_val, y_val = val_data
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    model = LatentMLPClassifier(input_dim=latents.shape[1]).to(device)
    criterion = nn.BCELoss(reduction='none')  # We'll apply class weights manually
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            sample_weights = torch.where(y_batch == 1, class_weights[1], class_weights[0])
            loss = (criterion(preds, y_batch) * sample_weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.8f}")

        # Validation loss
        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_val_batch, y_val_batch in val_loader:
                    x_val_batch = x_val_batch.to(device)
                    y_val_batch = y_val_batch.to(device)
                    val_preds = model(x_val_batch)
                    val_sample_weights = torch.where(y_val_batch == 1, class_weights[1], class_weights[0])
                    loss = (criterion(val_preds, y_val_batch) * val_sample_weights).mean()
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            print(f"   🔎 Val Loss: {val_loss:.8f}")

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(model, save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"⏹️ Early stopping at epoch {epoch+1}. Best val loss: {best_loss:.9f}")
                    break
        else:
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(model, save_path)
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
    plt.plot(range(1, len(history)+1), history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Latent MLP Classifier Training Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mlp_loss_curve.png")
    plt.close()

    print(f"✅ Model saved to {save_path} | Log: {log_path} | Loss plot: mlp_loss_curve.png")

def evaluate_model(model_path, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LatentMLPClassifier(input_dim=X_test.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    X_test, y_test = X_test.to(device), y_test.to(device)
    with torch.no_grad():
        preds = model(X_test)
        preds_binary = (preds > 0.5).float()
        accuracy = (preds_binary == y_test).float().mean().item()

    print(f"✅ Final Test Accuracy: {accuracy:.4f}")
