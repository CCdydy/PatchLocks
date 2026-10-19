#!/usr/bin/env python3
"""Trim a groom to the visual hull of the photo masks (fur cannot reach outside the silhouettes).

The visual hull of the body masks of all capture views bounds the fur. Strands that leave it (a
spray of long strands off a cheek, spikes off the outline) are shortened about their root,
uniformly, so the first point outside the hull comes back to its surface; the shape and the tip
taper are kept. Voxels are carved on the GPU; a voxel survives if it projects inside the mask in
all but --misses of the views that see it, and the hull is dilated by --dilate voxels to absorb
mask noise.

  python patchlocks/deploy/carve_groom_with_hull.py --groom <run>/grow/surfpack_v2fur.npz \\
      --data_dir <in>/nf_data --out <run>/grow/surfpack_v2fur_carved.npz
"""
import argparse
import json
import os
import pickle

import cv2
import numpy as np
import torch
from scipy import ndimage


def visual_hull(data_dir, lo, hi, vox, misses, dev):
    M = pickle.load(open(os.path.join(data_dir, "3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl"), "rb"))
    axes = [np.arange(lo[i], hi[i], vox) + vox / 2 for i in range(3)]
    G = torch.as_tensor(np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3), dtype=torch.float32, device=dev)
    homo = torch.cat([G, torch.ones_like(G[:, :1])], 1)
    inside = torch.zeros(len(G), dtype=torch.int16, device=dev)
    seen = torch.zeros(len(G), dtype=torch.int16, device=dev)
    for k in sorted(M):
        m = cv2.imread(os.path.join(data_dir, "masks_2/body", f"{k}.png"), 0)
        H, W = m.shape
        mt = torch.as_tensor(m > 127, device=dev)
        c = homo @ torch.as_tensor(np.asarray(M[k], np.float32), device=dev)
        w = c[:, 3]
        px = (c[:, 0] / w * 0.5 + 0.5) * W
        py = (1 - (c[:, 1] / w * 0.5 + 0.5)) * H
        ok = (w > 0) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        inside += (ok & mt[py.clamp(0, H - 1).long(), px.clamp(0, W - 1).long()]).short()
        seen += ok.short()
    occ = ((inside >= seen - misses) & (seen >= 10)).cpu().numpy().reshape([len(a) for a in axes])
    return occ, len(M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groom", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vox", type=float, default=0.003, help="voxel size (mesh units)")
    ap.add_argument("--misses", type=int, default=2, help="views a voxel may fall outside the mask in")
    ap.add_argument("--dilate", type=int, default=1, help="hull dilation in voxels")
    ap.add_argument("--min_scale", type=float, default=0.25, help="never shorten a strand below this fraction")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    g = dict(np.load(a.groom, allow_pickle=True))
    S = g["strands"].astype(np.float32)
    flat = S.reshape(-1, 3)
    lo = flat.min(0) - 2 * a.vox
    hi = flat.max(0) + 2 * a.vox
    occ, nviews = visual_hull(a.data_dir, lo, hi, a.vox, a.misses, dev)
    if a.dilate > 0:
        occ = ndimage.binary_dilation(occ, iterations=a.dilate)
    shape = np.array(occ.shape)

    seg = np.linalg.norm(np.diff(S, axis=1), axis=2)
    cum = np.concatenate([np.zeros((len(S), 1), np.float32), np.cumsum(seg, axis=1)], 1)
    arc = cum[:, -1]
    scale = np.ones(len(S), np.float32)
    for c0 in range(0, len(S), 500_000):
        s = S[c0:c0 + 500_000]
        idx = np.clip(np.floor((s - lo) / a.vox).astype(np.int64), 0, shape - 1)
        ins = occ[idx[..., 0], idx[..., 1], idx[..., 2]]
        ins[:, 0] = True                                   # the root is on the skin
        out_any = ~ins.all(1)
        first = np.argmin(ins, axis=1)                     # first point outside
        keep_arc = cum[c0:c0 + 500_000][np.arange(len(s)), np.maximum(first - 1, 0)]
        sc = np.where(out_any, keep_arc / np.maximum(arc[c0:c0 + 500_000], 1e-9), 1.0)
        scale[c0:c0 + 500_000] = np.maximum(sc, a.min_scale)
    g["strands"] = S[:, :1] + (S - S[:, :1]) * scale[:, None, None]
    g["hull_carve_scale"] = scale
    np.savez(a.out, **g)
    rep = {"views": nviews, "hull_voxels": int(occ.sum()), "strands": int(len(S)),
           "trimmed": int((scale < 0.999).sum()), "trimmed_frac": round(float((scale < 0.999).mean()), 4),
           "scale_of_trimmed_p10_50_90": np.round(np.percentile(scale[scale < 0.999], [10, 50, 90]), 3).tolist() if (scale < 0.999).any() else []}
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
