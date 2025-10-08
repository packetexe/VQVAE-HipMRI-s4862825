import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import HipMRISliceSet
from modules import VQVAE

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_dir = r"C:\Users\melvi\Desktop\VQVAE-HipMRI-s4862825\data\train"
    val_dir   = r"C:\Users\melvi\Desktop\VQVAE-HipMRI-s4862825\data\val"
    out_dir   = "readme_images"
    os.makedirs(out_dir, exist_ok=True)

    img_sz = 128
    train_ds = HipMRISliceSet(train_dir, size=128)
    val_ds   = HipMRISliceSet(val_dir, size=128)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = VQVAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
 

    epochs = 60
    best_ssim = 0.0
    train_losses, val_ssims, epochs_axis = [], [], []

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {ep}/{epochs}")

        for x in pbar:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                x_hat, loss, logs = model(x)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item() * x.size(0)
            pbar.set_postfix(loss=loss.item(),
                             recon=float(logs["recon"]),
                             vq=float(logs["vq"]),
                             perp=float(logs["perp"]))

        # validation
        val_ssim = eval_ssim(model, val_dl, device)
        avg_train = running / len(train_ds)
        print(f"[ep {ep}] train_loss={avg_train:.4f}  val_ssim={val_ssim:.4f}")
        # track plots
        train_losses.append(avg_train)
        val_ssims.append(val_ssim)
        epochs_axis.append(ep)

        # save plots every 5 epochs
        if ep % 5 == 0:
            plot_curves(epochs_axis, train_losses, val_ssims, out_dir=out_dir)

        # save small sample tensors every few epochs
        if ep % 5 == 0:
            torch.save({"x": x[:8].cpu(), "x_hat": x_hat[:8].cpu()},
                       os.path.join(out_dir, f"e{ep}_batch.pt"))

        # save best checkpoint by SSIM
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save({"model": model.state_dict(),
                        "val_ssim": val_ssim,
                        "epoch": ep},
                       "best.pt")
            print(f"New best SSIM: {val_ssim:.4f} (saved as best.pt)")

    print("Done. Best val SSIM:", best_ssim)

# SSIM utilities (ignore this, basically to avoid extra dependencies)
from skimage.metrics import structural_similarity as ssim
import numpy as np

@torch.no_grad()
def eval_ssim(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for x in loader:
        x = x.to(device)
        x_hat, _, _ = model(x)
        total += batch_ssim(x, x_hat) * x.size(0)
        count += x.size(0)
    return total / max(count, 1)

def batch_ssim(x, y):
    # x,y: (B,1,H,W) in [0,1]
    a = x.clamp(0, 1).cpu().numpy()
    b = y.clamp(0, 1).cpu().numpy()
    vals = [ssim(ai[0], bi[0], data_range=1.0) for ai, bi in zip(a, b)]
    return float(np.mean(vals))

# Plot curves
def plot_curves(epochs, train_losses, val_ssims, out_dir="readme_images"):
    os.makedirs(out_dir, exist_ok=True)

    # Loss curve
    plt.figure()
    plt.plot(epochs, train_losses, label="train_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Training Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    # SSIM curve
    plt.figure()
    plt.plot(epochs, val_ssims, label="val_ssim")
    plt.xlabel("epoch"); plt.ylabel("SSIM"); plt.title("Validation SSIM")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ssim_curve.png"))
    plt.close()

if __name__ == "__main__":
    main()