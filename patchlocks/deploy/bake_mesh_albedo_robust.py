#!/usr/bin/env python3
"""Robust per-vertex MESH albedo from multi-view photos (Tier-1 color solve).

Projects the body-mesh vertices into the calibrated views and samples the photo RGB; strands
then take their colour from the mesh (color_groom_from_albedo.py). Robust against:

  1. NAIVE MEAN -> ghosting:   per-vertex MAD outlier rejection across views drops
     mislabeled-visible views, then a view-facing-weighted mean of the survivors.
  2. FLAT FALLBACK -> blotches: vertices seen by < --min_views get filled by mesh-graph
     Laplacian diffusion from well-observed neighbors (no dead flat patches).
  3. SILHOUETTE BLEED:         body mask is eroded so grazing/edge pixels (background
     halo) do not leak in; occlusion + z-buffer are kept from front_visible.
  4. ACCESSORIES:              --exclude_dir masks (e.g. a collar, segment_accessory_masks.py)
     are never sampled.

Output is a per-vertex (V,3) sRGB albedo .npy.

  python patchlocks/deploy/bake_mesh_albedo_robust.py \
    --mesh <furless.obj> --data_dir <nf_data> --views all \
    --out <run>/mesh_albedo_robust.npy [--exclude_dir accessory]
"""
import argparse
import glob
import os
import pickle

import cv2
import numpy as np
def load_image(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def bilinear_sample(img, x, y):
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = np.clip(x0 + 1, 0, w - 1), np.clip(y0 + 1, 0, h - 1)
    wx, wy = (x - x0)[..., None], (y - y0)[..., None]
    return (img[y0, x0] * (1 - wx) * (1 - wy)
            + img[y0, x1] * wx * (1 - wy)
            + img[y1, x0] * (1 - wx) * wy
            + img[y1, x1] * wx * wy)


def load_obj_mesh(path):
    """Load vertices/triangles and compute smooth vertex normals without trimesh."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                ids = [int(x.split("/")[0]) - 1 for x in line.split()[1:]]
                for i in range(1, len(ids) - 1):
                    faces.append([ids[0], ids[i], ids[i + 1]])
    v = np.asarray(verts, np.float32)
    f = np.asarray(faces, np.int64)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for corner in range(3):
        np.add.at(vn, f[:, corner], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)
    return v, f, vn


def latest_matrices(data_dir):
    pref = os.path.join(data_dir, "3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl")
    if os.path.exists(pref):
        return pref
    c = sorted(glob.glob(os.path.join(data_dir, "3d_gaussian_splatting/GS_/cameras/*matrices*.pkl")))
    if not c:
        raise FileNotFoundError(f"no matrices pkl under {data_dir}")
    return c[-1]


def camera_center(M):
    """Recover the camera center for the repository's row-vector clip convention."""
    Minv = np.linalg.inv(M.astype(np.float64))
    def unproject(nx, ny, nz):
        h = np.array([nx, ny, nz, 1.0]) @ Minv
        return h[:3] / h[3]
    a1, b1 = unproject(0.0, 0.0, -0.5), unproject(0.0, 0.0, 0.5)
    a2, b2 = unproject(0.55, 0.35, -0.5), unproject(0.55, 0.35, 0.5)
    d1 = (b1 - a1) / np.linalg.norm(b1 - a1)
    d2 = (b2 - a2) / np.linalg.norm(b2 - a2)
    t = np.linalg.lstsq(np.stack([d1, -d2], 1), a2 - a1, rcond=None)[0]
    return (0.5 * ((a1 + t[0] * d1) + (a2 + t[1] * d2))).astype(np.float32)


def front_visible_vertices(v, vn, M, width, height, diag, zbuf_w):
    """Front-face plus conservative point z-buffer visibility, dependency-free."""
    homo = np.concatenate([v, np.ones((len(v), 1), np.float32)], 1)
    clip = homo @ M
    depth = clip[:, 3]
    ok = np.abs(depth) > 1e-6
    ndc = np.zeros((len(v), 3), np.float32)
    ndc[ok] = clip[ok, :3] / depth[ok, None]
    px = (ndc[:, 0] * 0.5 + 0.5) * width
    py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height
    front = ((camera_center(M)[None] - v) * vn).sum(1) > 0
    inb = ok & (px >= 0) & (px < width) & (py >= 0) & (py < height)

    rw = int(zbuf_w)
    rh = max(1, int(round(height * rw / width)))
    xi = np.clip((px / width * rw).astype(np.int64), 0, rw - 1)
    yi = np.clip((py / height * rh).astype(np.int64), 0, rh - 1)
    zbuf = np.full(rw * rh, np.inf, np.float32)
    ids = np.flatnonzero(inb)
    np.minimum.at(zbuf, yi[ids] * rw + xi[ids], depth[ids])
    nearest = zbuf[yi * rw + xi]
    tol = max(0.004 * float(diag), 0.006 * float(np.nanmax(depth[ok]) - np.nanmin(depth[ok])))
    visible = inb & front & (depth <= nearest + tol)
    return visible, px.astype(np.float32), py.astype(np.float32)


def erode_mask(mask, px):
    if px <= 0:
        return mask
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.erode((mask > 0.5).astype(np.uint8), k, iterations=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--views", default="all", help="comma-separated view ids, or 'all'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask_dir", default="body", choices=["body", "hair"],
                    help="which mask gates sampled pixels (body=whole silhouette, hair=fur only)")
    ap.add_argument("--exclude_dir", default="",
                    help="masks_2/<name> of pixels never to sample (e.g. a collar from segment_accessory_masks.py)")
    ap.add_argument("--erode_px", type=int, default=6, help="erode mask to kill silhouette background bleed")
    ap.add_argument("--mask_min", type=float, default=0.5)
    ap.add_argument("--zbuf_w", type=int, default=1600)
    ap.add_argument("--facing_pow", type=float, default=2.0, help="weight = max(0,cos)^p; trust frontal views more")
    ap.add_argument("--mad_k", type=float, default=2.5, help="reject views whose luma deviates > k*MAD from the per-vertex median")
    ap.add_argument("--min_views", type=int, default=2, help="vertices with fewer surviving views are filled from neighbors")
    ap.add_argument("--fill_iters", type=int, default=40, help="graph-Laplacian diffusion iters to fill unobserved vertices")
    ap.add_argument("--smooth_iters", type=int, default=2, help="light Laplacian denoise over observed albedo")
    ap.add_argument("--smooth_lambda", type=float, default=0.35)
    ap.add_argument("--init_color", default="0.6,0.55,0.5")
    ap.add_argument("--srgb_to_linear", type=int, default=0)
    a = ap.parse_args()

    v, faces, vn = load_obj_mesh(a.mesh)
    V = len(v)
    diag = float(np.linalg.norm(v.max(0) - v.min(0)))
    init = np.asarray([float(x) for x in a.init_color.split(",")], np.float32)
    cam = latest_matrices(a.data_dir)
    mats = pickle.load(open(cam, "rb"))

    if a.views.strip() == "all":
        vids = sorted(mats.keys())
    else:
        vids = [x.strip() for x in a.views.split(",") if x.strip()]

    # dense per-vertex, per-view sample buffers (NaN = not observed in that view)
    samp = np.full((V, len(vids), 3), np.nan, np.float32)
    wts = np.zeros((V, len(vids)), np.float32)

    for j, vid in enumerate(vids):
        ipath = os.path.join(a.data_dir, "images_2", f"{vid}.png")
        if not os.path.exists(ipath):
            print(f"view {vid}: no image", flush=True)
            continue
        img = load_image(ipath)
        h, w = img.shape[:2]
        mp = os.path.join(a.data_dir, f"masks_2/{a.mask_dir}", f"{vid}.png")
        mk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        mask = (mk.astype(np.float32) / 255) if mk is not None else np.ones((h, w), np.float32)
        if a.exclude_dir:
            ex = cv2.imread(os.path.join(a.data_dir, "masks_2", a.exclude_dir, f"{vid}.png"), cv2.IMREAD_GRAYSCALE)
            if ex is not None:
                mask = mask * (ex < 128)
        mask = erode_mask(mask, a.erode_px)
        m = mats[vid]
        mat = (m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)).astype(np.float32)
        vis, px, py = front_visible_vertices(v, vn, mat, w, h, diag, a.zbuf_w)
        bx = np.clip(px, 0, w - 1).astype(np.int64)
        by = np.clip(py, 0, h - 1).astype(np.int64)
        keep = np.flatnonzero(vis & (mask[by, bx] > a.mask_min))
        if len(keep) == 0:
            print(f"view {vid}: 0 visible", flush=True)
            continue
        rgb = bilinear_sample(img, px[keep], py[keep]).astype(np.float32)
        # view-facing weight: cos between vertex normal and vertex->camera
        C = camera_center(mat)
        vd = C[None] - v[keep]
        vd /= np.maximum(np.linalg.norm(vd, axis=1, keepdims=True), 1e-9)
        cos = np.clip((vn[keep] * vd).sum(1), 0, 1)
        wj = (cos ** a.facing_pow).astype(np.float32)
        samp[keep, j] = rgb
        wts[keep, j] = wj
        print(f"view {vid}: visible verts {len(keep)}", flush=True)

    # ---- robust per-vertex combine: MAD reject on luminance, then facing-weighted mean ----
    lum = 0.2126 * samp[..., 0] + 0.7152 * samp[..., 1] + 0.0722 * samp[..., 2]  # (V, nv)
    med = np.nanmedian(lum, axis=1, keepdims=True)                                # (V,1)
    mad = np.nanmedian(np.abs(lum - med), axis=1, keepdims=True)
    mad = np.where(mad < 1e-4, 1e-4, mad)
    valid = np.isfinite(lum)
    inlier = valid & (np.abs(lum - med) <= a.mad_k * mad)
    inlier = np.where(np.isfinite(lum), inlier, False)

    w = wts * inlier
    wsum = w.sum(1)                                                               # (V,)
    col = np.full((V, 3), np.nan, np.float32)
    good = wsum > 1e-8
    sc = np.nansum(np.where(inlier[..., None], samp * w[..., None], 0.0), axis=1)
    col[good] = (sc[good] / wsum[good, None]).astype(np.float32)
    nobs = inlier.sum(1)
    observed = good & (nobs >= a.min_views)
    col[~observed] = np.nan
    print(f"observed verts (>= {a.min_views} views): {int(observed.sum())}/{V}", flush=True)

    # ---- build mesh vertex adjacency ----
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
    e = np.concatenate([e, e[:, ::-1]], 0)
    order = np.argsort(e[:, 0], kind="stable")
    e = e[order]
    starts = np.searchsorted(e[:, 0], np.arange(V))
    ends = np.searchsorted(e[:, 0], np.arange(V), side="right")
    nbr = e[:, 1]

    def neighbor_mean(x, mask_valid):
        acc = np.zeros((V, 3), np.float64)
        cnt = np.zeros(V, np.float64)
        vn_ = np.where(mask_valid[:, None], x, 0.0)
        contrib = vn_[nbr] * mask_valid[nbr, None]
        cok = mask_valid[nbr].astype(np.float64)
        np.add.at(acc, e[:, 0], contrib)
        np.add.at(cnt, e[:, 0], cok)
        out = np.where(cnt[:, None] > 0, acc / np.maximum(cnt[:, None], 1e-9), 0.0)
        return out.astype(np.float32), cnt > 0

    # ---- fill unobserved via Laplacian diffusion from observed ----
    filled = np.where(observed[:, None], col, init[None]).astype(np.float32)
    fixed = observed.copy()
    # Propagate outward ONE ring per iteration, averaging ONLY over already-known
    # (fixed) neighbours. Using all-neighbours here would let unfilled default
    # verts average each other and freeze the whole hole at `init` in one step.
    it = 0
    while not fixed.all():
        nm, has = neighbor_mean(filled, fixed)   # mean over KNOWN neighbours only
        upd = ~fixed & has
        if not upd.any():
            break  # remaining verts are in components with no observed vertex
        filled[upd] = nm[upd]
        fixed = fixed | upd
        it += 1
        if it >= max(a.fill_iters, V):
            break
    print(f"fill: propagated to {int(fixed.sum())}/{V} verts in {it} rings", flush=True)

    # ---- light Laplacian denoise on originally-observed verts (edges preserved by low lambda/iters) ----
    for _ in range(a.smooth_iters):
        nm, has = neighbor_mean(filled, np.ones(V, bool))
        sm = observed & has
        filled[sm] = (1 - a.smooth_lambda) * filled[sm] + a.smooth_lambda * nm[sm]

    col = np.clip(filled, 0, 1).astype(np.float32)
    if a.srgb_to_linear:
        col = np.where(col <= 0.04045, col / 12.92, ((col + 0.055) / 1.055) ** 2.4).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out, col)
    L = 0.2126 * col[:, 0] + 0.7152 * col[:, 1] + 0.0722 * col[:, 2]
    print(f"saved {a.out} | verts {V} observed {int(observed.sum())} "
          f"lum q[5,50,95]={np.percentile(L, [5, 50, 95]).round(3).tolist()}", flush=True)


if __name__ == "__main__":
    main()
