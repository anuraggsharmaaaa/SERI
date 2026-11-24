import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image


def load_tif_2d(tif_path: str) -> np.ndarray:
    with Image.open(tif_path) as img:
        arr = np.array(img, dtype=np.float32)

    if arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
            arr = arr[0]
        else:
            arr = arr[:, :, 0]

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(np.percentile(arr, 2)), float(np.percentile(arr, 98))
    if hi - lo > 1e-8:
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr.astype(np.float32)


def crop_64x64_around(raster: np.ndarray, row_px: int, col_px: int, size: int = 64) -> np.ndarray:
    h, w = raster.shape
    half = size // 2
    r0, r1 = row_px - half, row_px + half
    c0, c1 = col_px - half, col_px + half

    pad_top = max(0, -r0)
    pad_bottom = max(0, r1 - h)
    pad_left = max(0, -c0)
    pad_right = max(0, c1 - w)

    r0_c, r1_c = max(0, r0), min(h, r1)
    c0_c, c1_c = max(0, c0), min(w, c1)

    crop = raster[r0_c:r1_c, c0_c:c1_c]
    if any(p > 0 for p in (pad_top, pad_bottom, pad_left, pad_right)):
        crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")

    if crop.shape != (size, size):
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_BILINEAR)
    return crop.astype(np.float32)


def extract_patches(
    tif_path: str,
    csv_path: str = "data/karnataka_master_data.csv",
    out_dir: str = "seri_patches",
    patch_size: int = 64,
):
    if not os.path.exists(csv_path) and os.path.exists("SERI_Core/" + csv_path):
        csv_path = "SERI_Core/" + csv_path

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    raster = load_tif_2d(tif_path)
    h, w = raster.shape

    lats = df["Latitude"].to_numpy(dtype=np.float64)
    lons = df["Longitude"].to_numpy(dtype=np.float64)

    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())

    for idx in range(len(df)):
        r_norm = (lat_max - lats[idx]) / (lat_max - lat_min + 1e-8)
        c_norm = (lons[idx] - lon_min) / (lon_max - lon_min + 1e-8)

        row_px = int(np.clip(r_norm * (h - 1), 0, h - 1))
        col_px = int(np.clip(c_norm * (w - 1), 0, w - 1))

        patch = crop_64x64_around(raster, row_px, col_px, size=patch_size)
        np.save(os.path.join(out_dir, f"patch_{idx}.npy"), patch)

    print(f"Extracted {len(df)} patches ({patch_size}x{patch_size}) from {tif_path} into {out_dir}/")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m utils.extract_tif_patches path/to/your_file.tif")
    else:
        extract_patches(sys.argv[1])