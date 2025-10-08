import os, glob
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm

src_root = r"C:\Users\melvi\Desktop\nifti" # Input from this folder
dst_root = r"C:\Users\melvi\Desktop\VQVAE-HipMRI-s4862825\data" # Output in this foler
os.makedirs(dst_root, exist_ok=True)

# Recursively find NIfTI files in all subfolders
nii_paths = glob.glob(os.path.join(src_root, "**", "*.nii*"), recursive=True)

for vol_path in tqdm(nii_paths, desc="Converting"):
    # Preserve train/val/test structure automatically
    rel_path = os.path.relpath(vol_path, src_root)
    subdir = os.path.dirname(rel_path)
    vol_name = os.path.splitext(os.path.basename(vol_path))[0]

    # Create matching output directory
    vol_out = os.path.join(dst_root, subdir, vol_name)
    os.makedirs(vol_out, exist_ok=True)

    # Load & normalize
    nii = nib.load(vol_path)
    data = nii.get_fdata()
    data = np.nan_to_num(data)
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    data = (data * 255).astype(np.uint8)

    # Save each slice
    for i in range(data.shape[2]):  # axial slices
        slice_img = data[:, :, i]
        cv2.imwrite(os.path.join(vol_out, f"slice_{i:03d}.png"), slice_img)

print("2D PNGs saved under", dst_root)
