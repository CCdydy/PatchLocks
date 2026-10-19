#!/usr/bin/env python3
"""Showcase appearance: colour a groom from a photo-baked albedo on a dense furless mesh.

Display layer only; the groom geometry is not changed. The pipeline's own strand colour is one
photo-projected colour per patch (a patch is 3-4 cm across), which flattens the coat's gradients.
Here every root takes the inverse-distance-weighted albedo of its nearest vertices on a dense
mesh in the groom frame. Bake the albedo with bake_mesh_albedo_robust.py; pass --exclude_dir
<mask> there to keep an accessory such as a collar out (segment_accessory_masks.py).

Optionally facial features are exposed: strands rooted close to small separate mesh pieces
(eyeballs, tongue) or to given points (the nose) are removed or shortened.

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
    ap.add_argument("--expose_small_components", type=float, default=0.0,
                    help="pieces of --mesh with fewer than this fraction of its faces (eyeballs, tongue) are features")
    ap.add_argument("--expose_point", action="append", default=[], help="x,y,z,radius (metres): a feature sphere, e.g. the nose")
    ap.add_argument("--expose_bald", type=float, default=0.006, help="strands rooted closer than this to a feature are removed")
    ap.add_argument("--expose_soft", type=float, default=0.012, help="... and shortened up to this distance")
    ap.add_argument("--expose_min", type=float, default=0.25, help="length scale at the bald edge")
    a = ap.parse_args()

    v, f, _ = load_obj_mesh(a.mesh)
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

    removed = shortened = 0
    if a.expose_small_components > 0 or a.expose_point:
        dist = np.full(len(roots), np.inf)
        if a.expose_small_components > 0:
            import trimesh
            lab = trimesh.graph.connected_component_labels(trimesh.Trimesh(v, f, process=False).face_adjacency, node_count=len(f))
            small = np.bincount(lab) < a.expose_small_components * len(f)
            fv = np.unique(f[small[lab]])
            if len(fv):
                dist = np.minimum(dist, cKDTree(v[fv]).query(roots)[0])
        for spec in a.expose_point:
            x, y, z_, r = [float(t) for t in spec.split(",")]
            dist = np.minimum(dist, np.maximum(np.linalg.norm(roots - [x, y, z_], axis=1) - r, 0.0))
        keep = dist >= a.expose_bald
        t = np.clip((dist - a.expose_bald) / max(a.expose_soft - a.expose_bald, 1e-9), 0, 1)
        scale = (a.expose_min + (1 - a.expose_min) * t * t * (3 - 2 * t)).astype(np.float32)
        S = g["strands"].astype(np.float32)
        g["strands"] = S[:, :1] + (S - S[:, :1]) * scale[:, None, None]
        removed, shortened = int((~keep).sum()), int(((scale < 0.999) & keep).sum())
        n = len(keep)
        for k_ in list(g):
            if isinstance(g[k_], np.ndarray) and g[k_].ndim >= 1 and len(g[k_]) == n:
                g[k_] = g[k_][keep]
    g["appearance_source"] = np.asarray(f"photo-baked albedo {os.path.basename(a.albedo)} on {os.path.basename(a.mesh)} (showcase layer)")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **g)
    print(json.dumps({"strands": int(len(roots)), "removed": removed, "shortened": shortened,
                      "root_to_vertex_mm_p50_p99": np.round(np.percentile(d[:, 0], [50, 99]) * 1000, 2).tolist(),
                      "luma_p5_50_95": np.round(np.percentile(col @ [0.2126, 0.7152, 0.0722], [5, 50, 95]), 3).tolist()}))


if __name__ == "__main__":
    main()
