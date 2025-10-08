import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from dataset import HipMRISliceSet
from modules import VQVAE
from skimage.metrics import structural_similarity as ssim
import numpy as np

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dir = r"C:\Users\melvi\Desktop\VQVAE-HipMRI-s4862825\data\test"
    out_dir  = "readme_images"
    os.makedirs(out_dir, exist_ok=True)

    ds = HipMRISliceSet(test_dir, size=128)
    dl = DataLoader(ds, batch_size=32, shuffle=False)

    model = VQVAE().to(device)
    state = torch.load("best.pt", map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    scores = []
    with torch.no_grad():
        for i, x in enumerate(dl):
            x = x.to(device)
            xhat, _, _ = model(x)
            scores.append(batch_ssim(x, xhat))
            if i == 0:
                grid = interleave_grid(x[:8], xhat[:8])
                save_image(grid, os.path.join(out_dir, "test_recons.png"))
    print(f"Test SSIM: {float(np.mean(scores)):.4f}")

    # Save final SSIM to text file
    test_ssim = float(np.mean(scores)) if scores else 0.0
    print(f"Test SSIM: {test_ssim:.4f}")
    with open("final_ssim.txt", "w") as f:
        f.write(f"Test SSIM: {test_ssim:.4f}\n")
    print("Saved final_ssim.txt")

def batch_ssim(x, y):
    a = x.clamp(0,1).cpu().numpy()
    b = y.clamp(0,1).cpu().numpy()
    vals = [ssim(ai[0], bi[0], data_range=1.0) for ai, bi in zip(a, b)]
    return float(np.mean(vals))

def interleave_grid(x, xhat):
    pairs = []
    for a, b in zip(x, xhat): pairs += [a, b]
    return make_grid(torch.stack(pairs), nrow=2, padding=2)

if __name__ == "__main__":
    main()
