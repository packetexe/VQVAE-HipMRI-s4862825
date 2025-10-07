import os
import torch
from torch.utils.data import DataLoader
from dataset import HipMRISliceSet
from modules import VQVAE

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dir = r"C:\Users\melvi\Desktop\VQVAE-HipMRI-s4862825\data\train"
    val_dir   = r"C:\Users\melvi\Desktop\VQVAE-HipMRI-s4862825\data\val"
    out_dir   = "readme_images"
    os.makedirs(out_dir, exist_ok=True)

    train_ds = HipMRISliceSet(train_dir, size=128)
    val_ds   = HipMRISliceSet(val_dir, size=128)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = VQVAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    for ep in range(1, 3):  # short test run
        model.train()
        for x in train_dl:
            x = x.to(device)
            xhat, loss, logs = model(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoch {ep} done")

if __name__ == "__main__":
    main()
