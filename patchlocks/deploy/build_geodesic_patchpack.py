"""Build a high-surface patch control field for fur.

This is intentionally pre-diffusion: it validates the surface atlas itself.
Patch centers are control points on a dense body surface; roots are sampled
independently over the whole surface and blend nearby patches by smooth weights.
"""
import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import cKDTree


NO_FUR_PARTS = ("eyes", "nosetip", "paw_pads", "inner_earcanal")
SMALL_PARTS = ("face", "ears", "paws", "leg_front", "leg_rear", "tail")
PART_NAMES = ("body", "face", "ears", "paws", "legs", "tail", "other")


def unit(v, eps=1e-9):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def load_annotations(path, nverts):
    ann = json.load(open(path)) if path and os.path.exists(path) else {}
    sets = {}
    for k, v in ann.items():
        arr = np.asarray(v, np.int64)
        arr = arr[(arr >= 0) & (arr < nverts)]
        sets[k] = arr
    return sets


def mark_vertices(nverts, annotations, keys):
    out = np.zeros(nverts, bool)
    for k in keys:
        if k in annotations:
            out[annotations[k]] = True
    return out


def face_part_ids(faces, annotations, nverts):
    part_v = np.full(nverts, len(PART_NAMES) - 1, np.int32)
    if "body" in annotations:
        part_v[annotations["body"]] = PART_NAMES.index("body")
    if "tail" in annotations:
        part_v[annotations["tail"]] = PART_NAMES.index("tail")
    for key in ("leg_front", "leg_rear"):
        if key in annotations:
            part_v[annotations[key]] = PART_NAMES.index("legs")
    for key in ("paws", "paw_pads", "front_paws"):
        if key in annotations:
            part_v[annotations[key]] = PART_NAMES.index("paws")
    if "ears" in annotations:
        part_v[annotations["ears"]] = PART_NAMES.index("ears")
    if "face" in annotations:
        part_v[annotations["face"]] = PART_NAMES.index("face")
    tri = part_v[faces]
    ids = np.zeros(len(faces), np.int32)
    for i in range(len(faces)):
        vals, cnt = np.unique(tri[i], return_counts=True)
        ids[i] = vals[np.argmax(cnt)]
    return ids


def sample_surface_weighted(mesh, count, face_mask, rng):
    areas = mesh.area_faces.astype(np.float64) * face_mask.astype(np.float64)
    p = areas / max(areas.sum(), 1e-12)
    fid = rng.choice(len(mesh.faces), size=count, replace=True, p=p)
    r1 = np.sqrt(rng.random(count))
    r2 = rng.random(count)
    bary = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1).astype(np.float32)
    pts = (mesh.triangles[fid] * bary[:, :, None]).sum(1).astype(np.float32)
    return pts, fid.astype(np.int32), bary


def radius_for_part(base_radius, part_ids):
    mult = np.ones_like(part_ids, dtype=np.float32)
    idx = {n: i for i, n in enumerate(PART_NAMES)}
    mult[part_ids == idx["body"]] = 1.15
    mult[part_ids == idx["tail"]] = 0.90
    mult[part_ids == idx["legs"]] = 0.76
    mult[part_ids == idx["paws"]] = 0.55
    mult[part_ids == idx["face"]] = 0.58
    mult[part_ids == idx["ears"]] = 0.58
    mult[part_ids == idx["other"]] = 0.75
    return base_radius * mult


def greedy_variable_poisson(cand_pts, cand_faces, cand_bary, cand_radius, target, rng):
    order = rng.permutation(len(cand_pts))
    picked = []
    tree_pts = []
    tree = None
    rebuild_every = 64
    max_r = float(cand_radius.max())
    for ci in order:
        p = cand_pts[ci]
        ok = True
        if tree is not None:
            near = tree.query_ball_point(p, max_r)
            for j in near:
                pj = picked[j]
                d = np.linalg.norm(p - cand_pts[pj])
                if d < 0.72 * min(cand_radius[ci], cand_radius[pj]):
                    ok = False
                    break
        if ok:
            picked.append(int(ci))
            tree_pts.append(p)
            if len(picked) == target:
                break
            if len(tree_pts) == 1 or len(tree_pts) % rebuild_every == 0:
                tree = cKDTree(np.asarray(tree_pts))
    if tree is None and tree_pts:
        tree = cKDTree(np.asarray(tree_pts))
    picked = np.asarray(picked, np.int64)
    return cand_pts[picked], cand_faces[picked], cand_bary[picked], cand_radius[picked]


def compute_root_weights(root_pos, patch_pos, patch_radius, k=6):
    tree = cKDTree(patch_pos)
    kk = min(max(k * 3, k), len(patch_pos))
    d, idx = tree.query(root_pos, k=kk, workers=-1)
    if d.ndim == 1:
        d = d[:, None]
        idx = idx[:, None]
    rr = patch_radius[idx]
    x = d / np.maximum(rr, 1e-8)
    w = np.where(x < 1.0, np.cos(0.5 * np.pi * x) ** 2, 0.0).astype(np.float32)
    count = (w > 1e-5).sum(1).astype(np.int32)
    order = np.argsort(-w, axis=1)[:, :k]
    row = np.arange(len(root_pos))[:, None]
    idxk = idx[row, order].astype(np.int32)
    wk = w[row, order]
    wsum = wk.sum(1, keepdims=True)
    wk = np.where(wsum > 1e-8, wk / np.maximum(wsum, 1e-8), 0.0).astype(np.float32)
    return idxk, wk, count, wsum[:, 0]


def add_backfill_patches(root_pos, root_face, root_bary, root_count, patch_pos, patch_face, patch_bary, patch_radius,
                         face_part, base_radius, target_count, max_add):
    add_pos = []
    add_face = []
    add_bary = []
    add_rad = []
    tree = cKDTree(patch_pos)
    need = np.flatnonzero(root_count < target_count)
    if not len(need):
        return patch_pos, patch_face, patch_bary, patch_radius, 0
    # Farthest low-coverage roots first, so one patch fixes a whole local component.
    dnear, _ = tree.query(root_pos[need], k=1, workers=-1)
    need = need[np.argsort(-dnear)]
    current = patch_pos.copy()
    tree2 = cKDTree(current)
    for ri in need:
        p = root_pos[ri]
        d, _ = tree2.query(p, k=1)
        r = radius_for_part(base_radius, np.asarray([face_part[root_face[ri]]], np.int32))[0]
        if d < 0.45 * r:
            continue
        add_pos.append(p)
        add_face.append(root_face[ri])
        add_bary.append(root_bary[ri])
        add_rad.append(r)
        current = np.concatenate([current, p[None]], 0)
        if len(add_pos) % 64 == 0:
            tree2 = cKDTree(current)
        if len(add_pos) >= max_add:
            break
    if not add_pos:
        return patch_pos, patch_face, patch_bary, patch_radius, 0
    return (
        np.concatenate([patch_pos, np.asarray(add_pos, np.float32)], 0),
        np.concatenate([patch_face, np.asarray(add_face, np.int32)], 0),
        np.concatenate([patch_bary, np.asarray(add_bary, np.float32)], 0),
        np.concatenate([patch_radius, np.asarray(add_rad, np.float32)], 0),
        len(add_pos),
    )


def scatter_panel(ax, pts, val, title, cmap="viridis", s=0.25, vmin=None, vmax=None):
    im = ax.scatter(pts[:, 0], pts[:, 2], c=val, s=s, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def make_debug(out, mesh, root_pos, root_count, root_wsum, dom_patch, patch_pos, patch_radius, nofur_v):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    take = rng.choice(len(root_pos), min(120000, len(root_pos)), replace=False)
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    im0 = scatter_panel(ax[0, 0], root_pos[take], np.clip(root_count[take], 0, 6), "root patch coverage count", "magma", vmin=0, vmax=6)
    plt.colorbar(im0, ax=ax[0, 0], fraction=0.03)
    im1 = scatter_panel(ax[0, 1], root_pos[take], root_wsum[take], "raw weight sum before normalize", "viridis", vmin=0, vmax=np.percentile(root_wsum, 99))
    plt.colorbar(im1, ax=ax[0, 1], fraction=0.03)
    im2 = scatter_panel(ax[1, 0], root_pos[take], dom_patch[take] % 256, "dominant patch id mod 256", "tab20", s=0.35)
    plt.colorbar(im2, ax=ax[1, 0], fraction=0.03)
    ax[1, 1].scatter(root_pos[take, 0], root_pos[take, 2], c="lightgray", s=0.15, linewidths=0)
    nf = np.flatnonzero(nofur_v)
    if len(nf):
        nf_take = nf[rng.choice(len(nf), min(30000, len(nf)), replace=False)]
        vv = np.asarray(mesh.vertices)[nf_take]
        ax[1, 1].scatter(vv[:, 0], vv[:, 2], c="red", s=0.35, linewidths=0)
    ax[1, 1].scatter(patch_pos[:, 0], patch_pos[:, 2], c=np.clip(patch_radius, patch_radius.min(), patch_radius.max()), cmap="cool", s=7, linewidths=0)
    ax[1, 1].set_aspect("equal")
    ax[1, 1].set_title("patch centers + no-fur vertices(red)", fontsize=9)
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "DEBUG_geodesic_patchpack_xz.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, dims, title in zip(ax, [(0, 2), (1, 2), (0, 1)], ["front x-z", "side y-z", "top x-y"]):
        a.scatter(root_pos[take, dims[0]], root_pos[take, dims[1]], c=np.clip(root_count[take], 0, 6), s=0.15, cmap="magma", vmin=0, vmax=6, linewidths=0)
        a.scatter(patch_pos[:, dims[0]], patch_pos[:, dims[1]], c="cyan", s=2, linewidths=0)
        a.set_aspect("equal")
        a.set_title(title, fontsize=9)
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "DEBUG_coverage_3views.png"), dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="patchlocks/datasets/neuralfur/extracted/data/Artemis/panda_processed_GH2/walk/neus_mesh_uv.obj")
    ap.add_argument("--annotations", default="patchlocks/datasets/neuralfur/extracted/data/Artemis/panda_processed_GH2/walk/annotations_neus.json")
    ap.add_argument("--out", default="patchlocks/deploy/neuralfur_geopatch_neus")
    ap.add_argument("--npatch", type=int, default=720)
    ap.add_argument("--nroots", type=int, default=220000)
    ap.add_argument("--candidate_mult", type=int, default=35)
    ap.add_argument("--base_radius", type=float, default=0.055, help="patch radius as fraction of bbox diagonal")
    ap.add_argument("--root_k", type=int, default=4)
    ap.add_argument("--min_cover", type=int, default=3)
    ap.add_argument("--max_backfill", type=int, default=300)
    ap.add_argument("--cover_iters", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--nofur_small_components", type=float, default=0.0,
                    help="faces of connected components smaller than this fraction of all faces grow no fur "
                         "(eyes, teeth, tongue modelled as separate pieces); 0 = off")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    mesh = trimesh.load(args.mesh, process=False)
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces, np.int64)
    diag = float(np.linalg.norm(mesh.extents))
    annotations = load_annotations(args.annotations, len(V))
    nofur_v = mark_vertices(len(V), annotations, NO_FUR_PARTS)
    nofur_face = nofur_v[F].mean(1) > 0.34
    if args.nofur_small_components > 0:
        comp = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(F))
        small = np.bincount(comp) < args.nofur_small_components * len(F)
        nofur_face |= small[comp]
        print(f"no fur on {int(small.sum())} small components ({int(small[comp].sum())} faces)", flush=True)
    grow_face = ~nofur_face
    face_part = face_part_ids(F, annotations, len(V))
    base_radius = args.base_radius * diag

    cand_n = max(args.npatch * args.candidate_mult, args.npatch + 1000)
    cand_pts, cand_face, cand_bary = sample_surface_weighted(mesh, cand_n, grow_face, rng)
    cand_part = face_part[cand_face]
    cand_radius = radius_for_part(base_radius, cand_part)
    patch_pos, patch_face, patch_bary, patch_radius = greedy_variable_poisson(
        cand_pts, cand_face, cand_bary, cand_radius, args.npatch, rng
    )

    root_pos, root_face, root_bary = sample_surface_weighted(mesh, args.nroots, grow_face, rng)
    added_total = 0
    root_idx, root_w, root_count, root_wsum = compute_root_weights(root_pos, patch_pos, patch_radius, args.root_k)
    for _ in range(args.cover_iters):
        if int((root_count < args.min_cover).sum()) == 0 or added_total >= args.max_backfill:
            break
        budget = args.max_backfill - added_total
        patch_pos, patch_face, patch_bary, patch_radius, added = add_backfill_patches(
            root_pos, root_face, root_bary, root_count, patch_pos, patch_face, patch_bary, patch_radius,
            face_part, base_radius, args.min_cover, budget
        )
        added_total += added
        root_idx, root_w, root_count, root_wsum = compute_root_weights(root_pos, patch_pos, patch_radius, args.root_k)
        if added == 0:
            break
    dom_patch = root_idx[:, 0]

    # Patch neighbor graph for smoothness losses.
    ptree = cKDTree(patch_pos)
    nd, ni = ptree.query(patch_pos, k=min(9, len(patch_pos)), workers=-1)
    neigh = ni[:, 1:].astype(np.int32)
    neigh_dist = nd[:, 1:].astype(np.float32)

    np.savez(
        os.path.join(args.out, "geodesic_patchpack.npz"),
        mesh_path=os.path.abspath(args.mesh),
        annotation_path=os.path.abspath(args.annotations),
        patch_pos=patch_pos.astype(np.float32),
        patch_face=patch_face.astype(np.int32),
        patch_bary=patch_bary.astype(np.float32),
        patch_radius=patch_radius.astype(np.float32),
        patch_part=face_part[patch_face].astype(np.int32),
        patch_neighbors=neigh,
        patch_neighbor_dist=neigh_dist,
        root_pos=root_pos.astype(np.float32),
        root_face=root_face.astype(np.int32),
        root_bary=root_bary.astype(np.float32),
        root_patch_idx=root_idx,
        root_patch_w=root_w,
        root_cover_count=root_count,
        root_weight_sum=root_wsum.astype(np.float32),
        nofur_vertices=np.flatnonzero(nofur_v).astype(np.int32),
        grow_face=grow_face.astype(np.bool_),
        part_names=np.asarray(PART_NAMES),
        base_radius=np.float32(base_radius),
        bbox_diag=np.float32(diag),
        min_cover=np.int32(args.min_cover),
        backfill_added=np.int32(added_total),
    )
    make_debug(args.out, mesh, root_pos, root_count, root_wsum, dom_patch, patch_pos, patch_radius, nofur_v)
    print(
        f"DONE {args.out}/geodesic_patchpack.npz | mesh V={len(V)} F={len(F)} diag={diag:.3f} "
        f"patches={len(patch_pos)} (+{added_total}) roots={len(root_pos)} nofur_faces={int(nofur_face.sum())} "
        f"coverage: zero={int((root_count==0).sum())} <{args.min_cover}={int((root_count<args.min_cover).sum())} "
        f"mean={root_count.mean():.2f} p10/p50/p90={np.percentile(root_count,[10,50,90]).round(2).tolist()} "
        f"weight_sum min/p50={float(root_wsum.min()):.4f}/{float(np.median(root_wsum)):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
