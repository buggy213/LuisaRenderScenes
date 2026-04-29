"""Compare two EXR files using SSIM.

Usage:
    python ssim.py <reference.exr> <comparison.exr>

Prints per-channel and luminance-weighted SSIM scores.
EXR values are tone-mapped with Reinhard before comparison so that
HDR highlights don't dominate the metric.
"""

import sys
import numpy as np
import imageio.v3 as iio
from skimage.metrics import structural_similarity as ssim


def load_exr(path: str) -> np.ndarray:
    img = iio.imread(path, plugin="opencv")  # float32 BGR or BGRA
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    if img.shape[2] == 4:
        img = img[:, :, :3]
    # opencv loads BGR; convert to RGB
    return img[:, :, ::-1].astype(np.float32)


def reinhard(img: np.ndarray) -> np.ndarray:
    return img / (1.0 + img)


def main():
    if len(sys.argv) != 3:
        print("Usage: python ssim.py <reference.exr> <comparison.exr>")
        sys.exit(1)

    ref = reinhard(load_exr(sys.argv[1]))
    cmp = reinhard(load_exr(sys.argv[2]))

    if ref.shape != cmp.shape:
        print(f"Shape mismatch: {ref.shape} vs {cmp.shape}")
        sys.exit(1)

    channel_names = ["R", "G", "B"]
    scores = []
    for i, name in enumerate(channel_names):
        s = ssim(ref[:, :, i], cmp[:, :, i], data_range=1.0)
        scores.append(s)
        print(f"  {name}: {s:.6f}")

    # Luminance-weighted average (BT.709)
    weights = np.array([0.2126, 0.7152, 0.0722])
    overall = float(np.dot(scores, weights))
    print(f"  Overall (luminance-weighted): {overall:.6f}")


if __name__ == "__main__":
    main()
