#!/usr/bin/env python3
"""Showcase appearance: colour a groom from a photo-baked albedo on a dense furless mesh.

Display layer only; the groom geometry is not changed. The pipeline's own strand colour is one
photo-projected colour per patch (a patch is 3-4 cm across), which flattens the coat's gradients.
Here every root takes the inverse-distance-weighted albedo of its nearest vertices on a dense
mesh in the groom frame. Bake the albedo with bake_mesh_albedo_robust.py; pass
--exclude_dir <mask> there to keep an accessory such as a collar out (segment_accessory_masks.py).

  python patchlocks/deploy/color_groom_from_albedo.py --groom <run>/grow/surfpack_v2fur.npz \\
      --mesh <in>/furless_sub2.obj --albedo <in>/appearance/albedo_sub2_srgb.npy \\
      --out <run>/grow/surfpack_v2fur_photo.npz
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bake_mesh_albedo_robust import load_obj_mesh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groom", required=True)
    ap.add_argument("--mesh", required=True, help="dense furless mesh (OBJ) in the groom frame")
    ap.add_argument("--albedo", required=True, help="per-vertex sRGB albedo .npy for --mesh")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4, help="nearest mesh vertices per root")
    ap.add_argument("--root_gain", type=float, default=1.0, help="multiplies the root colour")
    ap.add_argument("--tip_gain", type=float, default=1.0, help="multiplies the tip colour")
    a = ap.parse_args()

    v, _, _ = load_obj_mesh(a.mesh)
    alb = np.load(a.albedo).astype(np.float64)
    assert len(alb) == len(v), "albedo does not match mesh"
    g = dict(np.load(a.groom, allow_pickle=True))
    roots = g["strands"][:, 0].astype(np.float64)
    d, idx = cKDTree(v).query(roots, k=a.k)
    w = 1.0 / np.maximum(d, 1e-5)
    w /= w.sum(1, keepdims=True)
    col = (alb[idx] * w[..., None]).sum(1)
    g["color"] = np.clip(col * a.root_gain, 0, 1).astype(np.float32)
    g["color_tip"] = np.clip(col * a.tip_gain, 0, 1).astype(np.float32)
    g["appearance_source"] = np.asarray(f"photo-baked albedo {os.path.basename(a.albedo)} on {os.path.basename(a.mesh)} (showcase layer)")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **g)
    print(json.dumps({"strands": int(len(roots)), "root_to_vertex_mm_p50_p99": np.round(np.percentile(d[:, 0], [50, 99]) * 1000, 2).tolist(),
                      "luma_p5_50_95": np.round(np.percentile(col @ [0.2126, 0.7152, 0.0722], [5, 50, 95]), 3).tolist()}))


if __name__ == "__main__":
    main()
