#!/usr/bin/env python3
"""Side-by-side panels: capture photo | groom rendered at that camera | render over the photo.

  python patchlocks/qa/compose_realcam_compare.py --data_dir <nf_data> --renders <dir> \\
      --tag fur --views 0003,0029 --out compare.jpg

Reads <renders>/render_<tag>_<view>_raw.png (render_groom_realcam.py, RGBA) and
<data_dir>/images_2/<view>.png, crops both to the animal (photo mask bbox, padded) and stacks
one row per view.
"""
import argparse
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--tag", default="fur")
    ap.add_argument("--views", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--row_h", type=int, default=640)
    ap.add_argument("--bg", default="150,150,150")
    a = ap.parse_args()
    bg = np.array([int(x) for x in a.bg.split(",")], np.float32)
    rows = []
    for vid in a.views.split(","):
        photo = cv2.imread(os.path.join(a.data_dir, "images_2", f"{vid}.png"))
        mask = cv2.imread(os.path.join(a.data_dir, "masks_2/body", f"{vid}.png"), 0) > 0
        r = cv2.imread(os.path.join(a.renders, f"render_{a.tag}_{vid}_raw.png"), cv2.IMREAD_UNCHANGED)
        if r is None:
            print("missing render", vid)
            continue
        al = r[..., 3:4].astype(np.float32) / 255.0
        rgb = r[..., :3].astype(np.float32)
        on_gray = (rgb * al + bg * (1 - al)).astype(np.uint8)
        dim = photo.astype(np.float32) * 0.55
        on_photo = (rgb * al + dim * (1 - al)).astype(np.uint8)
        ys, xs = np.nonzero(mask | (al[..., 0] > 0.1))
        p = int(0.08 * max(xs.ptp(), ys.ptp()))
        x0, x1 = max(xs.min() - p, 0), min(xs.max() + p, photo.shape[1])
        y0, y1 = max(ys.min() - p, 0), min(ys.max() + p, photo.shape[0])
        tiles = []
        for im, label in [(photo, f"photo {vid}"), (on_gray, "PatchLocks render"), (on_photo, "render over photo")]:
            c = im[y0:y1, x0:x1]
            c = cv2.resize(c, (int(c.shape[1] * a.row_h / c.shape[0]), a.row_h), interpolation=cv2.INTER_AREA)
            cv2.putText(c, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(c, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(c)
        rows.append(np.concatenate(tiles, 1))
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 8, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
    cv2.imwrite(a.out, np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
