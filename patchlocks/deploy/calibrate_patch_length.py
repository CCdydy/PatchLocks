#!/usr/bin/env python3
"""Refine Step 2's per-patch strand length from the fur silhouettes of all capture views.

Step 2 fits length on its 12 fit views with a splat renderer. This compares, over every capture
view, how far the fur reaches beyond the furless silhouette in the photo mask and in the grown
groom (as patchlocks/qa/silhouette_fur_thickness.py does), per patch: each silhouette pixel outside the furless
mesh belongs to the nearest camera-facing patch centre. The per-patch ratio photo / groom of the
90th-percentile thickness, shrunk towards 1 for patches with few samples, smoothed over
neighbouring patches and clipped, multiplies Step 2's length (silhouette pixels near an accessory
mask, e.g. a collar standing off the fur, are ignored with --exclude); grow again with the written
patch_params (the grow normalizes length by its median, so --target_arclen still sets the scale).
run_v2_character_mainline.sh runs this as Step 3b when SILHOUETTE_LENGTH_ITERS > 0.

  python patchlocks/deploy/calibrate_patch_length.py --data_dir <in>/nf_data --mesh <in>/furless.obj \\
      --groom <run>/grow/surfpack_v2fur.npz --surfpack <in>/surfpack/surface_patch_dataset.npz \\
      --patch_params <run>/step2/patch_params.npz --out <run>/step2/patch_params_silcal.npz \\
      [--regions body,legs --region_prior <run>/v2_region_prior.annotated.npz]
"""
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_graph_opt_geo import project_row  # noqa: E402


def fill_mesh(px, py, F, H, W):
    m = np.zeros((H, W), np.uint8)
    for f in F:
        cv2.fillConvexPoly(m, np.stack([px[f], py[f]], 1).round().astype(np.int32), 1)
    return m


def patch_map_for_view(centers, normals, M, W, H):
    """Patch index image: each pixel takes the nearest camera-facing patch centre."""
    P = M.astype(np.float64).T
    Mp, p4 = np.array([P[0], P[1], P[3]])[:, :3], np.array([P[0], P[1], P[3]])[:, 3]
    C = -np.linalg.solve(Mp, p4)
    facing = ((C[None] - centers) * normals).sum(1) > 0
    px, py, ok = project_row(centers, M, W, H)
    sel = np.flatnonzero(facing & ok)
    tree = cKDTree(np.stack([px[sel], py[sel]], 1))
    yy, xx = np.mgrid[0:H:2, 0:W:2]
    _, idx = tree.query(np.stack([xx.ravel(), yy.ravel()], 1))
    small = sel[idx].reshape(yy.shape).astype(np.int32)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--groom", required=True)
    ap.add_argument("--surfpack", required=True)
    ap.add_argument("--patch_params", required=True, help="Step 2 patch_params.npz to refine")
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", default="")
    ap.add_argument("--subsample", type=int, default=300000)
    ap.add_argument("--pct", type=float, default=90.0)
    ap.add_argument("--n0", type=float, default=400.0, help="pixels at which a patch's ratio gets half weight")
    ap.add_argument("--clip", default="0.7,1.6")
    ap.add_argument("--smooth_k", type=int, default=8)
    ap.add_argument("--smooth_iters", type=int, default=2)
    ap.add_argument("--exclude", default="", help="masks_2/<name> (e.g. accessory): ignore silhouette pixels near it")
    ap.add_argument("--exclude_px", type=int, default=25, help="dilation of the excluded mask in pixels")
    ap.add_argument("--regions", default="", help="only change patches of these regions (comma list)")
    ap.add_argument("--region_prior", default="", help="patch_region labels for --regions")
    a = ap.parse_args()
    lo, hi = [float(x) for x in a.clip.split(",")]

    sp = np.load(a.surfpack, mmap_mode="r", allow_pickle=True)
    centers, normals = np.asarray(sp["centers"], np.float64), np.asarray(sp["Nn"], np.float64)
    Pn = len(centers)
    M = pickle.load(open(os.path.join(a.data_dir, "3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl"), "rb"))
    views = a.views.split(",") if a.views else sorted(M)
    mesh = trimesh.load(a.mesh, process=False)
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    S = np.load(a.groom)["strands"]
    S = S[np.random.default_rng(0).choice(len(S), min(a.subsample, len(S)), replace=False)]
    Ppts = S.reshape(-1, 3)
    ph_vals, gr_vals = [[] for _ in range(Pn)], [[] for _ in range(Pn)]
    for vid in views:
        photo = cv2.imread(os.path.join(a.data_dir, "masks_2/body", f"{vid}.png"), 0) > 0
        H, W = photo.shape
        mx, my, _ = project_row(V, M[vid], W, H)
        meshmask = fill_mesh(mx, my, F, H, W)
        sx, sy, ok = project_row(Ppts, M[vid], W, H)
        fur = np.zeros((H, W), np.uint8)
        fur[sy[ok].astype(int), sx[ok].astype(int)] = 1
        fur = cv2.morphologyEx(fur, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        groom_sil = np.maximum(fur, meshmask)
        c, d = V.mean(0), 1e-3
        pts = np.stack([c, c + [d, 0, 0], c + [0, d, 0], c + [0, 0, d]])
        qx, qy, _ = project_row(pts, M[vid], W, H)
        J = np.stack([qx[1:] - qx[0], qy[1:] - qy[0]]) / d
        mpp = 1.0 / max(float(np.linalg.svd(J, compute_uv=False)[0]), 1e-9)
        dist = cv2.distanceTransform((1 - meshmask).astype(np.uint8), cv2.DIST_L2, 5) * mpp
        pmap = patch_map_for_view(centers, normals, M[vid], W, H)
        keep = np.ones((H, W), bool)
        if a.exclude:
            ex = cv2.imread(os.path.join(a.data_dir, "masks_2", a.exclude, f"{vid}.png"), 0)
            if ex is not None:
                k = 2 * a.exclude_px + 1
                keep = cv2.dilate((ex > 127).astype(np.uint8), np.ones((k, k), np.uint8)) == 0
        for sil, store in [(photo, ph_vals), (groom_sil > 0, gr_vals)]:
            ring = sil & (meshmask == 0) & keep
            ids, vals = pmap[ring], dist[ring]
            order = np.argsort(ids, kind="stable")
            ids, vals = ids[order], vals[order]
            cut = np.flatnonzero(np.diff(ids)) + 1
            for grp_ids, grp_vals in zip(np.split(ids, cut), np.split(vals, cut)):
                if len(grp_ids):
                    store[int(grp_ids[0])].append(grp_vals)
        print(vid, end=" ", flush=True)
    print()

    n = np.zeros(Pn)
    ratio = np.ones(Pn)
    tp, tg = np.zeros(Pn), np.zeros(Pn)
    for i in range(Pn):
        if ph_vals[i] and gr_vals[i]:
            p, g = np.concatenate(ph_vals[i]), np.concatenate(gr_vals[i])
            n[i] = min(len(p), len(g))
            tp[i], tg[i] = np.percentile(p, a.pct), np.percentile(g, a.pct)
            ratio[i] = (tp[i] + 1e-3) / (tg[i] + 1e-3)
    w = n / (n + a.n0)
    r = np.exp(w * np.log(np.clip(ratio, lo, hi)))          # shrink towards 1 by sample count
    _, nb = cKDTree(centers).query(centers, k=a.smooth_k + 1)
    for _ in range(a.smooth_iters):
        r = np.exp(0.5 * np.log(r) + 0.5 * np.log(r[nb[:, 1:]]).mean(1))
    r = np.clip(r, lo, hi)
    if a.regions:
        reg = np.load(a.region_prior, allow_pickle=True)["patch_region"].astype(str)
        keep = np.isin(reg, [x.strip() for x in a.regions.split(",")])
        r[~keep] = 1.0
    pp = dict(np.load(a.patch_params, allow_pickle=True))
    pp["length"] = (pp["length"].astype(np.float64) * r).astype(np.float32)
    pp["silhouette_length_scale"] = r.astype(np.float32)
    pp["silhouette_length_source"] = np.asarray(f"calibrate_patch_length.py on {os.path.basename(a.groom)}, {len(views)} views, p{a.pct:g}")
    np.savez(a.out, **pp)
    seen = n > 0
    rep = {"patches_with_samples": int(seen.sum()), "scale_p10_50_90": np.round(np.percentile(r, [10, 50, 90]), 3).tolist(),
           "raw_ratio_p10_50_90_seen": np.round(np.percentile(ratio[seen], [10, 50, 90]), 3).tolist()}
    json.dump(rep, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
