#!/usr/bin/env python3
"""Compare the coat's appearance in renders at the capture cameras with the photos.

Over the pixels that the photo mask and every render cover (optionally a vertical band of the
mask, e.g. 0.6,1.0 for the legs), prints the luminance percentiles, the median colour and the
local contrast (median standard deviation of luminance in 9x9 windows). Real fur is fine enough
to read as a soft surface (low local contrast); strands that are too thick or too glossy read as
straw, which shows up as a local contrast several times the photo's long before it is obvious in
a side-by-side. Use it to tune strand radius, pixel filter and shading.

  python patchlocks/qa/coat_texture_stats.py --data_dir <in>/nf_data --views 0029,0040 \\
      --renders <dirA>/render_fur_{view}_raw.png <dirB>/render_fur_{view}_raw.png [--band 0.6,1.0]
"""
import argparse
import os

import cv2
import numpy as np


def local_contrast(img, k=9):
    L = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mu = cv2.blur(L, (k, k))
    return np.sqrt(np.maximum(cv2.blur(L * L, (k, k)) - mu * mu, 0)), L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--renders", nargs="+", required=True, help="paths with {view}; RGBA renders")
    ap.add_argument("--band", default="0,1", help="vertical band of the mask's bounding box (fractions)")
    ap.add_argument("--window", type=int, default=9)
    a = ap.parse_args()
    b0, b1 = [float(x) for x in a.band.split(",")]
    for v in a.views.split(","):
        ph = cv2.imread(os.path.join(a.data_dir, "images_2", f"{v}.png"))
        m = cv2.imread(os.path.join(a.data_dir, "masks_2/body", f"{v}.png"), 0) > 127
        ys = np.nonzero(m)[0]
        band = np.zeros_like(m)
        band[int(ys.min() + b0 * (ys.max() - ys.min())):int(ys.min() + b1 * (ys.max() - ys.min())) + 1] = True
        common = m & band
        rs = []
        for p in a.renders:
            r = cv2.imread(p.format(view=v), cv2.IMREAD_UNCHANGED)
            if r.shape[:2] != m.shape:
                r = cv2.resize(r, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_AREA)
            common &= r[..., 3] > 200
            rs.append(r[..., :3].copy())
        print(f"view {v}: {int(common.sum())} pixels")
        for name, img in [("photo", ph)] + list(zip(a.renders, rs)):
            sd, L = local_contrast(img, a.window)
            rgb = img[common][:, ::-1].astype(np.float64)
            print(f"  luma p10/50/90 {np.round(np.percentile(L[common], [10, 50, 90])).astype(int).tolist()}  "
                  f"RGB p50 {np.round(np.median(rgb, 0)).astype(int).tolist()}  "
                  f"local contrast {np.median(sd[common]):5.1f}  {name}")


if __name__ == "__main__":
    main()
