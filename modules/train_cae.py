import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_cae(
    model, dataloader, val_loader, optimizer, scheduler, device,
    n_epochs=100, early_stop=10, visualize_fn=None, save_path="cae_best.pth", 
    checkpoint_dir="checkpoints", start_epoch=0, 
):
    import os
    import torch

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float('inf')
    epochs_no_improve = 0
    criterion = nn.MSELoss()
    best_loss = np.inf
    patience = 0
    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, n_epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False)
        for x, _ in pbar:
            x = x.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            pbar.set_postfix({"batch_loss": loss.item()})
        train_loss /= len(dataloader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                out = model(x)
                loss = criterion(out, x)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        checkpoint_path = os.path.join(checkpoint_dir, f"cae_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            # Add more if needed (e.g., loss)
        }, checkpoint_path)

        if visualize_fn and (epoch+1) % 10 == 0:
            visualize_fn(model, val_loader, device)

        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience += 1
            if patience >= early_stop:
                print("Early stopping.")
                break

    # Plot loss curves
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CAE Training Loss')
    plt.legend()
    plt.show()

    # Optionally return stats
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_loss,
        "epochs_ran": epoch + 1
    }