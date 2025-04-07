import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from autoencoder import ConvAutoencoder
from train_autoencoder import load_numpy_dataset  # reuse your loader

def extract_and_save_latents(dataset_path, autoencoder_path, output_path='latents.npy', batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(autoencoder_path, map_location=device))
    model.eval()

    data = load_numpy_dataset(dataset_path)
    dataloader = DataLoader(TensorDataset(data), batch_size=batch_size)

    latents = []

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch[0].to(device)
            encoded = model.encoder(imgs)  # use only encoder part
            flat = encoded.view(encoded.size(0), -1).cpu().numpy()
            latents.append(flat)

    latents = np.concatenate(latents, axis=0)
    np.save(output_path, latents)
    print(f"✅ Latents saved to {output_path} (shape: {latents.shape})")