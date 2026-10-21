#!/usr/bin/env python3
"""Fur depth per vertex of the furless mesh, in the photos and in a groom.

In every view, the vertices on the outline of the furless mesh's silhouette (as in
patchlocks/deploy/fit_furless_mesh.py) measure how far the fur reaches beyond them: the distance
from their projection to the outline of the photo mask, and to the outline of the groom's
silhouette (all its strand points splatted over the mesh). The median over the views that see a
vertex on the outline is its fur depth. Writes fur_depth.json (photo and groom depth per region,
and how many vertices the groom makes too short or too long), fur_depth.npz (per vertex) and
fur_depth_ratio.jpg (photo / groom on a few views: blue groom too long, green right, red too short).

The outline of the fur is the envelope of all strands near it, so a vertex's depth is not its own
strands' length alone (a tail lying over the rump, a neighbour's feathering hold it out); read the
map as where the coat is too thick or too thin, not as a per-strand correction.

  python patchlocks/qa/fur_depth_map.py --mesh <in>/furless.obj --data_dir <in>/nf_data \\
      --groom <run>/grow/surfpack_v2fur.npz --regions <run>/v2_region_prior.annotated.npz \\
      --out_dir <run>/fur_depth
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as tnf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deploy"))
import fit_furless_mesh as ffm  # noqa: E402
from bake_mesh_albedo_robust import load_obj_mesh  # noqa: E402


def groom_distance(strands, v, S_mesh, chunk=4_000_000):
    """Signed distance (pixels) to the outline of the groom's silhouette joined with the mesh's."""
    H, W = v["H"], v["W"]
    sil = torch.zeros(H * W, device=strands.device)
    for c0 in range(0, len(strands), chunk):
        P2, _, inb = ffm.project(strands[c0:c0 + chunk].float(), v)
        sil[P2[inb, 1].long() * W + P2[inb, 0].long()] = 1.0
    sil = sil.view(1, 1, H, W)
    sil = -tnf.max_pool2d(-tnf.max_pool2d(sil, 3, 1, 1), 3, 1, 1)
    u = ((sil + S_mesh) > 0.5)[0, 0].cpu().numpy().astype(np.uint8)
    D = cv2.distanceTransform(u, cv2.DIST_L2, 5) - cv2.distanceTransform(1 - u, cv2.DIST_L2, 5)
    return torch.as_tensor(D, device=strands.device)[None, None]


def measure(mesh, views, bary, strands=None):
    """Per view and vertex: fur depth (metres) beyond the outline vertices, photo and groom."""
    n = len(mesh.V0)
    gp = np.full((len(views), n), np.nan, np.float32)
    gg = np.full((len(views), n), np.nan, np.float32)
    for k, v in enumerate(views):
        with torch.no_grad():
            P2, s, inb = ffm.project(mesh.V0, v)
            outl, S_mesh = mesh.outline(mesh.V0, P2, inb, v, bary)    # the collar is already left out
            o = outl.cpu().numpy()
            gp[k, o] = (ffm.sample(v["D"], P2, v["H"], v["W"]) / s)[outl].cpu().numpy()
            if strands is not None:
                Dg = groom_distance(strands, v, S_mesh)
                gg[k, o] = (ffm.sample(Dg, P2, v["H"], v["W"]) / s)[outl].cpu().numpy()
    return gp, gg


def per_vertex(g, ok):
    out = np.full(g.shape[1], np.nan)
    out[ok] = np.nanmedian(g[:, ok], axis=0)
    return out


def report(photo, groom, reg, ok, out_dir, tag, views, mesh, data_dir, map_views):
    ratio = (photo + 1e-3) / (groom + 1e-3)
    rep = {}
    for r in np.unique(reg):
        m = ok & (reg == r)
        if m.sum() < 10:
            continue
        rep[r] = {"vertices": int(m.sum()),
                  "photo_mm_p10_50_90": np.round(np.percentile(photo[m], [10, 50, 90]) * 1000, 1).tolist(),
                  "groom_mm_p10_50_90": np.round(np.percentile(groom[m], [10, 50, 90]) * 1000, 1).tolist(),
                  "ratio_p10_50_90": np.round(np.percentile(ratio[m], [10, 50, 90]), 2).tolist(),
                  "groom_too_short_frac": round(float((ratio[m] > 1.25).mean()), 3),
                  "groom_too_long_frac": round(float((ratio[m] < 0.8).mean()), 3)}
    json.dump(rep, open(os.path.join(out_dir, "fur_depth.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(out_dir, "fur_depth.npz"), photo=photo, groom=groom, ratio=ratio,
                        vertex_region=reg)
    names = [v["name"] for v in views]
    tiles = []
    for vn in map_views:
        v = views[names.index(vn)]
        img = cv2.imread(os.path.join(data_dir, "images_2", f"{vn}.png"))
        img = (cv2.resize(img, (v["W"], v["H"]), interpolation=cv2.INTER_AREA) * 0.45).astype(np.uint8)
        with torch.no_grad():
            P2, _, inb = ffm.project(mesh.V0, v)
        P2, inb = P2.cpu().numpy(), inb.cpu().numpy()
        for i in np.flatnonzero(ok & inb):
            t = np.clip((np.log(ratio[i]) / np.log(2.0) + 1) / 2, 0, 1)   # 0.5 blue .. 1 green .. 2 red
            col = (int(255 * (1 - t)), int(255 * (1 - abs(2 * t - 1))), int(255 * t))
            cv2.circle(img, (int(P2[i, 0]), int(P2[i, 1])), 2, col, -1)
        ys, xs = np.nonzero(v["mask"])
        c = img[max(ys.min() - 10, 0):ys.max() + 10, max(xs.min() - 10, 0):xs.max() + 10]
        cv2.putText(c, vn, (8, 24), 0, 0.7, (255, 255, 255), 2)
        tiles.append(cv2.resize(c, (int(c.shape[1] * 480 / c.shape[0]), 480)))
    cv2.imwrite(os.path.join(out_dir, "fur_depth_ratio.jpg"), np.concatenate(tiles, 1))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groom", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--regions", required=True, help="vertex_region npz, or patch_region + patch_pos npz")
    ap.add_argument("--mesh", default="", help="the furless mesh the groom was grown on (default: its mesh_path)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_views", type=int, default=2, help="outline views needed for a vertex's depth")
    ap.add_argument("--work_scale", type=float, default=0.5)
    ap.add_argument("--map_views", default="", help="views for the colour maps (default: 4 spread)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out_dir, exist_ok=True)
    g = np.load(a.groom, allow_pickle=True)
    V, F, _ = load_obj_mesh(a.mesh or str(g["mesh_path"]))
    V = V.astype(np.float64)
    reg = ffm.vertex_regions(a.regions, V)
    views = ffm.load_views(a.data_dir, a.work_scale, 6, dev)
    names = [v["name"] for v in views]
    map_views = a.map_views.split(",") if a.map_views else names[:: max(1, len(names) // 4)][:4]
    mesh = ffm.Mesh(V, F, 0.05, dev)
    mesh.edge_px = ffm.edge_px(mesh, mesh.V0, views)
    bary = ffm.bary_grid(min(max(2, int(np.ceil(mesh.edge_px / 0.7))), 10), dev)
    strands = torch.as_tensor(g["strands"].reshape(-1, 3), dtype=torch.float16, device=dev)
    gp, gg = measure(mesh, views, bary, strands)
    ok = np.sum(~np.isnan(gp), 0) >= a.min_views
    rep = report(per_vertex(gp, ok), per_vertex(gg, ok), reg, ok, a.out_dir, "map", views, mesh, a.data_dir, map_views)
    print(json.dumps(rep, indent=1))
    print("wrote", a.out_dir)


if __name__ == "__main__":
    main()
