import os, glob
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm
from collections import Counter

src_root = r"C:\Users\melvi\Desktop\nifti" # Input from this folder
dst_root = r"C:\Users\melvi\Desktop\VQVAE-HipMRI-s4862825\data" # Output in this foler
os.makedirs(dst_root, exist_ok=True)

def norm_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr)
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn + 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - mn) / (mx - mn + 1e-8)
    return (arr * 255).astype(np.uint8)

# Recursively find NIfTI files in all subfolders
nii_paths = glob.glob(os.path.join(src_root, "**", "*.nii*"), recursive=True)

skipped = []

for vol_path in tqdm(nii_paths, desc="Converting"):
    try:
        # Preserve train/val/test structure automatically
        rel_path = os.path.relpath(vol_path, src_root)
        subdir = os.path.dirname(rel_path)
        vol_name = os.path.splitext(os.path.basename(vol_path))[0]

        # Create matching output directory
        vol_out_root = os.path.join(dst_root, subdir)
        os.makedirs(vol_out_root, exist_ok=True)

        nii = nib.load(vol_path)
        data = nii.get_fdata()
        ndim = data.ndim

        if ndim == 3:
            # (H, W, D)
            vol_out = os.path.join(vol_out_root, vol_name)
            os.makedirs(vol_out, exist_ok=True)
            data_u8 = norm_uint8(data)
            for i in range(data_u8.shape[2]):
                slice_img = data_u8[:, :, i]
                cv2.imwrite(os.path.join(vol_out, f"slice_{i:03d}.png"), slice_img)

        elif ndim == 4:
            # (H, W, D, T)
            for t in range(data.shape[-1]):
                vol3d = data[..., t]
                if vol3d.ndim != 3:
                    skipped.append((vol_path, f"Unexpected 4D inner shape {vol3d.shape}"))
                    continue
                vol_out = os.path.join(vol_out_root, f"{vol_name}_t{t:02d}")
                os.makedirs(vol_out, exist_ok=True)
                vol3d_u8 = norm_uint8(vol3d)
                for i in range(vol3d_u8.shape[2]):
                    cv2.imwrite(os.path.join(vol_out, f"slice_{i:03d}.png"), vol3d_u8[:, :, i])

        elif ndim == 2:
            # Single 2D slice — save once
            vol_out = os.path.join(vol_out_root, vol_name)
            os.makedirs(vol_out, exist_ok=True)
            img = norm_uint8(data)
            cv2.imwrite(os.path.join(vol_out, "slice_000.png"), img)

        else:
            skipped.append((vol_path, f"Unsupported ndim={ndim}, shape={data.shape}"))

    except Exception as e:
        skipped.append((vol_path, f"Exception: {e}"))

print("2D PNGs saved under", dst_root)

# Summary, using splits
splits = []
for p in nii_paths:
    parts = p.split(os.sep)

    if len(parts) >= 2:
        splits.append(parts[-2])
count = Counter(splits)
print("\nSummary of converted volumes:")
for k, v in count.items():
    print(f"  {k}: {v} volumes converted")

if skipped:
    print("\n Skipped files:")
    for p, msg in skipped[:15]:
        print(f"  - {p} -> {msg}")
    if len(skipped) > 15:
        print(f"  ... and {len(skipped)-15} more")