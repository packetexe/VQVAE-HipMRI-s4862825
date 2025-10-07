import os, glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class HipMRISliceSet(Dataset):
    """Loads 2D slice images as grayscale tensors in [0,1]."""
    def __init__(self, root: str, size: int = 128):
        self.paths = []
        for pat in ("*.png", "*.jpg", "*.jpeg"):
            self.paths += glob.glob(os.path.join(root, "**", pat), recursive=True)
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}")
        self.tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("L")
        return self.tf(img)