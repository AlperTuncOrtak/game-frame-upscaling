"""
Step 1 - Baseline: classical interpolation methods for 2x upscaling.

For every screenshot in data/raw/<game>/:
  1. Crop to even width/height (so 2x downscale -> 2x upscale returns the same size)
  2. Downscale 2x (INTER_AREA, the standard way to simulate a low-resolution render)
  3. Upscale back 2x with nearest neighbor, bilinear and bicubic
  4. Measure PSNR and SSIM on the luminance (Y) channel against the original

Outputs:
  results/baseline_per_image.csv   - scores for every image and method
  results/baseline_summary.csv     - average scores per game and method
  results/crops/                   - zoomed crops for visual comparison

Run from the project root:
  python3 src/baseline.py
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

RAW_DIR = Path("data/raw")
RESULTS_DIR = Path("results")
CROPS_DIR = RESULTS_DIR / "crops"

SCALE = 2
BORDER = 4      # pixels ignored at the edges when measuring (standard practice)
CROP_SIZE = 128  # size of the zoomed comparison crops

METHODS = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
}


def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR, uint8
    if img is None:
        raise ValueError(f"Could not read {path}")
    h, w = img.shape[:2]
    return img[: h - h % SCALE, : w - w % SCALE]


def to_y(img_bgr):
    """Luminance channel as float in [0, 1]."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    return ycrcb[:, :, 0].astype(np.float64) / 255.0


def measure(original_bgr, upscaled_bgr):
    a = to_y(original_bgr)[BORDER:-BORDER, BORDER:-BORDER]
    b = to_y(upscaled_bgr)[BORDER:-BORDER, BORDER:-BORDER]
    psnr = peak_signal_noise_ratio(a, b, data_range=1.0)
    ssim = structural_similarity(a, b, data_range=1.0)
    return psnr, ssim


def most_detailed_crop(img_bgr, size):
    """Top-left corner of the crop with the most edges (best for visual comparison)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200).astype(np.float64)
    density = cv2.boxFilter(edges, -1, (size, size), normalize=True)
    half = size // 2
    h, w = gray.shape
    inner = density[half : h - half, half : w - half]
    y, x = np.unravel_index(np.argmax(inner), inner.shape)
    return y, x  # top-left of the crop in the original image


def save_crops(name, original, upscaled_by_method):
    y, x = most_detailed_crop(original, CROP_SIZE)
    tiles = [original[y : y + CROP_SIZE, x : x + CROP_SIZE]]
    labels = ["original"]
    for method, img in upscaled_by_method.items():
        tiles.append(img[y : y + CROP_SIZE, x : x + CROP_SIZE])
        labels.append(method)

    zoom = 3
    tiles = [cv2.resize(t, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST) for t in tiles]
    for tile, label in zip(tiles, labels):
        cv2.putText(tile, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(tile, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(CROPS_DIR / f"{name}.png"), np.hstack(tiles))


def main():
    paths = sorted(RAW_DIR.glob("*/*.png"))
    if not paths:
        print(f"No PNG files found in {RAW_DIR}/<game>/. Add screenshots first.")
        return

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for path in paths:
        game = path.parent.name
        original = load_image(path)
        h, w = original.shape[:2]
        low = cv2.resize(original, (w // SCALE, h // SCALE), interpolation=cv2.INTER_AREA)

        upscaled_by_method = {}
        for method, flag in METHODS.items():
            start = time.perf_counter()
            up = cv2.resize(low, (w, h), interpolation=flag)
            elapsed_ms = (time.perf_counter() - start) * 1000
            psnr, ssim = measure(original, up)
            upscaled_by_method[method] = up
            rows.append({
                "game": game, "image": path.name, "method": method,
                "psnr": psnr, "ssim": ssim, "time_ms": elapsed_ms,
            })

        save_crops(f"{game}_{path.stem}", original, upscaled_by_method)
        print(f"done: {game}/{path.name}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "baseline_per_image.csv", index=False)

    summary = df.groupby(["game", "method"])[["psnr", "ssim", "time_ms"]].mean().round(4)
    summary.to_csv(RESULTS_DIR / "baseline_summary.csv")

    print("\nAverage scores per game and method:")
    print(summary.to_string())
    print("\nOverall:")
    print(df.groupby("method")[["psnr", "ssim", "time_ms"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
