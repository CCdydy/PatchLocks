#!/usr/bin/env python3
"""Compare the fur silhouette of a groom with the photo masks, to calibrate groom length.

For every view, the furless mesh and the groom's strands are rasterized with the capture's
cameras (patch_graph_opt_geo.project_row). Fur thickness = distance from the furless-mesh
silhouette to the outer silhouette, measured the same way on the photo mask and on the groom,
converted to metres with the camera focal length and the mesh depth. Also reports the
silhouette IoU between groom and photo.

  python patchlocks/qa/silhouette_fur_thickness.py --data_dir <nf_data> --mesh furless.obj \\
      --groom outputs/.../grow/surfpack_v2fur.npz [--views 0000,0003,...] [--out report.json] \\
      [--surfpack surface_patch_dataset.npz --region_prior v2_region_prior.npz]

With --surfpack and --region_prior, every silhouette pixel outside the furless mesh is assigned to
the body region of the nearest camera-facing patch centre, and thickness is reported per region
(body, face, ears, legs, paws, tail, tail_base): the ratio photo / groom is the length scale that
region needs.
"""
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deploy"))
from patch_graph_opt_geo import project_row  # noqa: E402


def fill_mesh(px, py, F, H, W):
    m = np.zeros((H, W), np.uint8)
    for f in F:
        cv2.fillConvexPoly(m, np.stack([px[f], py[f]], 1).round().astype(np.int32), 1)
    return m


def thickness(sil, meshmask, metres_per_px, region_map=None, names=()):
    d = cv2.distanceTransform((1 - meshmask).astype(np.uint8), cv2.DIST_L2, 5)
    ring = (sil > 0) & (meshmask == 0)
    v = d[ring] * metres_per_px
    out = (float(np.percentile(v, 50)), float(np.percentile(v, 90))) if len(v) > 50 else (0.0, 0.0)
    per = {}
    if region_map is not None:
        r = region_map[ring]
        for i, nm in enumerate(names):
            vv = v[r == i]
            if len(vv) > 30:
                per[nm] = vv
    return out, per


def region_map_for_view(centers, normals, labels, M, W, H):
    """Region index image: each pixel takes the label of the nearest camera-facing patch centre."""
    from scipy.spatial import cKDTree
    P = M.astype(np.float64).T
    Mp, p4 = np.array([P[0], P[1], P[3]])[:, :3], np.array([P[0], P[1], P[3]])[:, 3]
    C = -np.linalg.solve(Mp, p4)
    facing = ((C[None] - centers) * normals).sum(1) > 0
    px, py, ok = project_row(centers, M, W, H)
    sel = facing & ok
    tree = cKDTree(np.stack([px[sel], py[sel]], 1))
    yy, xx = np.mgrid[0:H:2, 0:W:2]
    _, idx = tree.query(np.stack([xx.ravel(), yy.ravel()], 1))
    small = labels[sel][idx].reshape(yy.shape).astype(np.int16)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--groom", required=True)
    ap.add_argument("--views", default="")
    ap.add_argument("--subsample", type=int, default=300000)
    ap.add_argument("--out", default="")
    ap.add_argument("--surfpack", default="", help="surface_patch_dataset.npz (patch centres, normals)")
    ap.add_argument("--region_prior", default="", help="v2_region_prior.npz (patch_region labels)")
    a = ap.parse_args()
    names, centers = (), None
    if a.surfpack and a.region_prior:
        sp = np.load(a.surfpack, mmap_mode="r", allow_pickle=True)
        centers, normals = np.asarray(sp["centers"], np.float64), np.asarray(sp["Nn"], np.float64)
        reg = np.load(a.region_prior, allow_pickle=True)["patch_region"].astype(str)
        names = tuple(sorted(set(reg)))
        labels = np.array([names.index(x) for x in reg])
    acc = {nm: {"photo": [], "groom": []} for nm in names}

    M = pickle.load(open(os.path.join(a.data_dir, "3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl"), "rb"))
    views = a.views.split(",") if a.views else sorted(M)
    mesh = trimesh.load(a.mesh, process=False)
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    S = np.load(a.groom)["strands"]
    S = S[np.random.default_rng(0).choice(len(S), min(a.subsample, len(S)), replace=False)]
    P = S.reshape(-1, 3)
    rows = []
    for vid in views:
        photo = cv2.imread(os.path.join(a.data_dir, "masks_2/body", f"{vid}.png"), 0) > 0
        H, W = photo.shape
        mx, my, _ = project_row(V, M[vid], W, H)
        meshmask = fill_mesh(mx, my, F, H, W)
        sx, sy, ok = project_row(P, M[vid], W, H)
        fur = np.zeros((H, W), np.uint8)
        fur[sy[ok].astype(int), sx[ok].astype(int)] = 1
        fur = cv2.morphologyEx(fur, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        groom_sil = np.maximum(fur, meshmask)
        # metres per pixel at the mesh centre, for displacements perpendicular to the view ray:
        # largest singular value of the pixel-per-metre Jacobian of the projection
        c, d = V.mean(0), 1e-3
        pts = np.stack([c, c + [d, 0, 0], c + [0, d, 0], c + [0, 0, d]])
        qx, qy, _ = project_row(pts, M[vid], W, H)
        J = np.stack([qx[1:] - qx[0], qy[1:] - qy[0]]) / d
        mpp = 1.0 / max(float(np.linalg.svd(J, compute_uv=False)[0]), 1e-9)
        rmap = region_map_for_view(centers, normals, labels, M[vid], W, H) if centers is not None else None
        tp, per_p = thickness(photo, meshmask, mpp, rmap, names)
        tg, per_g = thickness(groom_sil, meshmask, mpp, rmap, names)
        for nm in names:
            if nm in per_p and nm in per_g:
                acc[nm]["photo"].append(per_p[nm]); acc[nm]["groom"].append(per_g[nm])
        iou = float((photo & (groom_sil > 0)).sum() / max((photo | (groom_sil > 0)).sum(), 1))
        rows.append({"view": vid, "photo_p50_cm": tp[0] * 100, "photo_p90_cm": tp[1] * 100,
                     "groom_p50_cm": tg[0] * 100, "groom_p90_cm": tg[1] * 100, "iou": iou})
    r = {k: float(np.median([x[k] for x in rows])) for k in rows[0] if k != "view"}
    print(f"{len(rows)} views (median): photo fur thickness p50 {r['photo_p50_cm']:.2f} / p90 {r['photo_p90_cm']:.2f} cm | "
          f"groom p50 {r['groom_p50_cm']:.2f} / p90 {r['groom_p90_cm']:.2f} cm | silhouette IoU {r['iou']:.3f}")
    per_region = {}
    for nm in names:
        if acc[nm]["photo"]:
            ph, gr = np.concatenate(acc[nm]["photo"]) * 100, np.concatenate(acc[nm]["groom"]) * 100
            per_region[nm] = {"photo_p50_cm": float(np.percentile(ph, 50)), "photo_p90_cm": float(np.percentile(ph, 90)),
                              "groom_p50_cm": float(np.percentile(gr, 50)), "groom_p90_cm": float(np.percentile(gr, 90)),
                              "ratio_p90": float(np.percentile(ph, 90) / max(np.percentile(gr, 90), 1e-6))}
            q = per_region[nm]
            print(f"  {nm:10s} photo p50/p90 {q['photo_p50_cm']:.2f}/{q['photo_p90_cm']:.2f} cm | groom "
                  f"{q['groom_p50_cm']:.2f}/{q['groom_p90_cm']:.2f} cm | photo/groom (p90) {q['ratio_p90']:.2f}")
    if a.out:
        json.dump({"median": r, "views": rows, "per_region": per_region}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
