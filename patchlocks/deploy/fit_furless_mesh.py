#!/usr/bin/env python3
"""Fit the furless body mesh to a multi-view capture, view by view (before the patch cover).

The photos see the coat, not the body under it. The body must therefore lie inside the animal's
silhouette in every view, under a fur depth that is plausible for its region, keep the anatomy
of the template (an artist's furless model in the capture frame) and keep its features where the
photos see them. For a mesh point x in view v, with D_v the signed distance to the outline of the
whole-animal mask (pixels, positive inside) and s the image scale at x (pixels per metre),
g = D_v(proj_v x) / s is the fur margin that photo leaves around x:

  containment, every vertex:              g >= d_min(region)   under at least d_min of fur
  coverage, vertices on the body outline: g <= d_max(region)   at most d_max of fur beyond the body

Silhouettes also hold long fringes, a tail lying against a hip, shadows and segmentation errors,
so the mesh only moves where the photos agree: containment tolerates --cont_tol (the mask
accuracy), and a vertex is pushed outwards only if most of the views that see it on the outline
(--cov_consensus) find it too thin; the consensus is renewed every --consensus_every steps.

The outline vertices are the contour-generator vertices on the outline of the mesh's own
silhouette in v (the union of its projected triangles, so no depth test is needed). The collar
(masks_2/accessory) is left out of the coverage term. With --up, no vertex may sink below the
floor (the template's lowest point: the paws stand on it), and cameras below the floor, which
look at the paws past the edge of the stage, do not test containment for points near the floor. Shape prior: as-rigid-as-possible energy
to the template plus Laplacian smoothness of the displacement; pieces smaller than
--small_components of the faces (eyeballs, a tongue) follow the surface around them rigidly.
Landmarks pull groups of vertices by a given offset (e.g. to triangulated eyes and nose).

All views constrain one shared mesh: each step visits a random mini-batch of views (the worst
fitting more often) and moves the mesh a little, at two levels: the template's own resolution
(vertex order kept, so its UVs and labels still apply) and once Loop-subdivided (the fine mesh).
Per-view overlays (photo, fur outline green, template body outline blue, fitted body outline
red) and a residual report show how every photo is met.

  python patchlocks/deploy/fit_furless_mesh.py --mesh <in>/furless.obj --data_dir <in>/nf_data \\
      --regions <vertex_region.npz | v2_region_prior.npz> --out_dir <in>/furless_fit

Regions: an npz with `vertex_region` (one label per vertex, label_patch_regions.py --mesh), or
with `patch_region` and `patch_pos` (labels of the nearest patch centre). --iters 0 only reports.
"""
import argparse
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as tnf
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bake_mesh_albedo_robust import load_obj_mesh  # noqa: E402

DEFAULT_BAND = ("body:0.004:0.035,tail_base:0.004:0.040,tail:0.006:0.050,legs:0.002:0.025,"
                "paws:0.0:0.012,face:0.002:0.025,ears:0.0:0.012")


# ---------------------------------------------------------------- inputs
def load_views(data_dir, ws, acc_dilate, dev):
    mats = pickle.load(open(os.path.join(data_dir, "3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl"), "rb"))
    views = []
    for k in sorted(mats):
        m = cv2.imread(os.path.join(data_dir, "masks_2/body", f"{k}.png"), 0)
        if m is None:
            continue
        H0, W0 = m.shape
        H, W = int(round(H0 * ws)), int(round(W0 * ws))
        u = (cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA) > 127).astype(np.uint8)
        D = cv2.distanceTransform(u, cv2.DIST_L2, 5) - cv2.distanceTransform(1 - u, cv2.DIST_L2, 5)
        acc = None
        ap = os.path.join(data_dir, "masks_2/accessory", f"{k}.png")
        if os.path.exists(ap):
            a = cv2.resize(cv2.imread(ap, 0), (W, H), interpolation=cv2.INTER_AREA) > 127
            if a.any():
                a = cv2.dilate(a.astype(np.uint8), np.ones((2 * acc_dilate + 1,) * 2, np.uint8)) > 0
                acc = torch.as_tensor(a, device=dev)
        P = np.asarray(mats[k], np.float64).T                     # clip = P @ [x, 1]
        A = np.array([P[0, :3], P[1, :3], P[3, :3]])
        cam = np.linalg.solve(A, -np.array([P[0, 3], P[1, 3], P[3, 3]]))
        views.append({"name": k, "H": H, "W": W, "mask": u,
                      "D": torch.as_tensor(D, dtype=torch.float32, device=dev)[None, None],
                      "acc": acc, "M": torch.as_tensor(np.asarray(mats[k], np.float32), device=dev),
                      "cam": torch.as_tensor(cam, dtype=torch.float32, device=dev)})
    return views


def vertex_regions(path, V):
    d = np.load(path, allow_pickle=True)
    if "vertex_region" in d.files:
        reg = d["vertex_region"].astype(str)
        assert len(reg) == len(V), "vertex_region does not match the mesh"
        return reg
    return d["patch_region"].astype(str)[cKDTree(d["patch_pos"]).query(V)[1]]


def parse_band(spec):
    band = {}
    for tok in spec.split(","):
        name, lo, hi = tok.split(":")
        band[name.strip()] = (float(lo), float(hi))
    return band


# ---------------------------------------------------------------- geometry
def project(X, v):
    M = v["M"]
    c = X @ M[:3] + M[3]
    w = c[:, 3]
    px = (c[:, 0] / w * 0.5 + 0.5) * v["W"]
    py = (1.0 - (c[:, 1] / w * 0.5 + 0.5)) * v["H"]
    # pixels per metre: sqrt of the area scale of the 2x3 projection Jacobian
    jx = 0.5 * v["W"] * (w[:, None] * M[:3, 0] - c[:, :1] * M[:3, 3]) / (w[:, None] ** 2)
    jy = 0.5 * v["H"] * (w[:, None] * M[:3, 1] - c[:, 1:2] * M[:3, 3]) / (w[:, None] ** 2)
    s = torch.linalg.norm(torch.cross(jx, jy, dim=1), dim=1).sqrt()
    inb = (w > 1e-6) & (px >= 0) & (px < v["W"]) & (py >= 0) & (py < v["H"])
    return torch.stack([px, py], 1), s.detach(), inb


def sample(img, P2, H, W):
    g = torch.stack([P2[:, 0] / W * 2 - 1, P2[:, 1] / H * 2 - 1], 1)[None, None]
    return tnf.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=False)[0, 0, 0]


def bary_grid(n, dev):
    pts = [(i / n, j / n) for i in range(n + 1) for j in range(n + 1 - i)]
    return torch.tensor([[a, b, 1 - a - b] for a, b in pts], dtype=torch.float32, device=dev)


def silhouette(P2, F, H, W, bary):
    """Union of the projected triangles (splatted barycentric samples, pinholes closed)."""
    S = torch.zeros(H * W, device=P2.device)
    for c0 in range(0, len(F), 200_000):
        pts = torch.einsum("kb,fbc->fkc", bary, P2[F[c0:c0 + 200_000]]).reshape(-1, 2)
        ix, iy = pts[:, 0].floor().long(), pts[:, 1].floor().long()
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        S[iy[ok] * W + ix[ok]] = 1.0
    S = S.view(1, 1, H, W)
    S = -tnf.max_pool2d(-tnf.max_pool2d(S, 3, 1, 1), 3, 1, 1)
    return S


class Mesh:
    """Deformable vertices, faces, edges, and the rigid small pieces of a (multi-piece) mesh."""

    def __init__(self, V, F, small_frac, dev):
        self.dev = dev
        self.V0 = torch.as_tensor(V, dtype=torch.float32, device=dev)
        self.F_np = F
        self.F = torch.as_tensor(F, dtype=torch.long, device=dev)
        lab = trimesh.graph.connected_component_labels(trimesh.Trimesh(V, F, process=False).face_adjacency,
                                                       node_count=len(F))
        small = np.bincount(lab) < small_frac * len(F)
        vlab = np.full(len(V), -1)
        vlab[F.ravel()] = np.repeat(lab, 3)
        self.rigid = np.zeros(len(V), bool)
        self.rigid[F[small[lab]].ravel()] = True
        self.Fd = torch.as_tensor(F[~small[lab]], dtype=torch.long, device=dev)   # deformable faces
        e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
        e = np.unique(e, axis=0)
        e = e[~self.rigid[e[:, 0]]]
        self.ei = torch.as_tensor(np.r_[e[:, 0], e[:, 1]], device=dev)
        self.ej = torch.as_tensor(np.r_[e[:, 1], e[:, 0]], device=dev)
        self.deg = torch.zeros(len(V), device=dev).index_add_(0, self.ei, torch.ones(len(self.ei), device=dev))
        self.e0 = self.V0[self.ei] - self.V0[self.ej]
        self.fdeg = torch.zeros(len(V), device=dev).index_add_(
            0, self.Fd.reshape(-1), torch.ones(self.Fd.numel(), device=dev))
        # rigid pieces follow the 30 nearest deformable vertices
        self.pieces = []
        tree = cKDTree(V[~self.rigid])
        dv = np.flatnonzero(~self.rigid)
        for k in np.flatnonzero(small):
            vid = np.unique(F[lab == k])
            anc = dv[tree.query(V[vid].mean(0), k=30)[1]]
            self.pieces.append((torch.as_tensor(vid, device=dev), torch.as_tensor(anc, device=dev)))
        self.mean_edge = float(self.e0.norm(dim=1).mean())

    def arap(self, X):
        e = X[self.ei] - X[self.ej]
        S = torch.zeros(len(X), 3, 3, device=self.dev).index_add_(0, self.ei, self.e0[:, :, None] * e[:, None, :])
        U, _, Vh = torch.linalg.svd(S.detach())
        R = Vh.transpose(1, 2) @ U.transpose(1, 2)
        flip = torch.det(R) < 0
        if flip.any():
            Vh2 = Vh.clone()
            Vh2[flip, 2] *= -1
            R = Vh2.transpose(1, 2) @ U.transpose(1, 2)
        r = e - (R[self.ei] @ self.e0[:, :, None])[..., 0]
        return (r ** 2).sum()

    def lap(self, X):
        u = X - self.V0
        nb = torch.zeros_like(u).index_add_(0, self.ei, u[self.ej]) / self.deg.clamp(min=1)[:, None]
        return (((u - nb) ** 2).sum(1) * (self.deg > 0)).sum()

    def place_rigid(self, X):
        X = X.clone()
        for vid, anc in self.pieces:
            a0, a1 = self.V0[anc], X[anc]
            c0, c1 = a0.mean(0), a1.mean(0)
            U, _, Vh = torch.linalg.svd((a0 - c0).T @ (a1 - c1))
            d = torch.sign(torch.det(Vh.T @ U.T))
            R = Vh.T @ torch.diag(torch.tensor([1.0, 1.0, float(d)], device=self.dev)) @ U.T
            X[vid] = (self.V0[vid] - c0) @ R.T + c1
        return X

    def outline(self, X, P2, inb, v, bary):
        """Contour-generator vertices on the outline of the mesh's silhouette in view v."""
        with torch.no_grad():
            T = X[self.Fd]
            n = torch.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0], dim=1)
            front = ((n * (v["cam"] - T.mean(1))).sum(1) > 0).float()
            cf = torch.zeros(len(X), device=self.dev).index_add_(0, self.Fd.reshape(-1), front.repeat_interleave(3))
            contour = (cf > 0) & (cf < self.fdeg)
            S = silhouette(P2, self.Fd, v["H"], v["W"], bary)
            border = S - (-tnf.max_pool2d(-S, 3, 1, 1))
            r = max(2, int(round(self.edge_px)))
            near = tnf.max_pool2d(border, 2 * r + 1, 1, r)
            on = sample(near, P2, v["H"], v["W"]) > 0.5
            out = contour & on & inb & ~torch.as_tensor(self.rigid, device=self.dev)
            if v["acc"] is not None:
                ix = P2[:, 0].long().clamp(0, v["W"] - 1)
                iy = P2[:, 1].long().clamp(0, v["H"] - 1)
                out &= ~v["acc"][iy, ix]
        return out, S


# ---------------------------------------------------------------- fitting
class Floor:
    """The floor the animal stands on (--up): its height, and the views that look up at it."""

    def __init__(self, up, V0, views, occl_height, dev):
        s = -1.0 if up.startswith("-") else 1.0
        self.u = torch.zeros(3, device=dev)
        self.u["xyz".index(up.lstrip("+-"))] = s
        self.h0 = float((V0 @ self.u).min())
        self.occl = occl_height
        self.low = {v["name"]: float(v["cam"] @ self.u) < self.h0 + 0.02 for v in views}

    def height(self, X):
        return X @ self.u - self.h0


def residuals(mesh, X, v, dmin, dmax, bary, floor=None, cov_ok=None, tol=0.0):
    P2, s, inb = project(X, v)
    g = sample(v["D"], P2, v["H"], v["W"]) / s                       # fur margin in metres
    deform = inb & ~torch.as_tensor(mesh.rigid, device=mesh.dev)
    outl, S = mesh.outline(X, P2, inb, v, bary)
    tested = deform
    if floor is not None and floor.low[v["name"]]:
        tested = deform & (floor.height(X).detach() > floor.occl)     # the stage edge hides the paws
    rc = torch.relu(dmin - g - tol) * tested
    rv = torch.relu(g - dmax) * outl
    if cov_ok is not None:
        rv = rv * cov_ok
    return g, rc, rv, deform, outl, S


def huber(x, d=1.0):
    a = x.abs()
    return torch.where(a < d, 0.5 * a * a, d * (a - 0.5 * d))


def edge_px(mesh, X, views):
    vals = []
    for v in views[:: max(1, len(views) // 8)]:
        _, s, inb = project(X, v)
        vals.append(float((s[inb]).median()) * mesh.mean_edge)
    return float(np.median(vals))


def consensus(mesh, X, views, dmax, bary, frac):
    """Vertices that most of the views seeing them on the outline find too thin."""
    n_out = torch.zeros(len(X), device=mesh.dev)
    n_thin = torch.zeros(len(X), device=mesh.dev)
    with torch.no_grad():
        for v in views:
            P2, s, inb = project(X, v)
            g = sample(v["D"], P2, v["H"], v["W"]) / s
            outl, _ = mesh.outline(X, P2, inb, v, bary)
            n_out += outl.float()
            n_thin += (outl & (g > dmax)).float()
    return (n_out >= 4) & (n_thin >= frac * n_out)


def fit_level(mesh, views, dmin, dmax, lmk, a, iters, log, tag, floor=None):
    X = mesh.V0.clone().requires_grad_(True)
    if iters <= 0:
        return mesh.V0.clone()
    opt = torch.optim.Adam([X], lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=a.lr * 0.1)
    tau = a.tau
    ema = np.ones(len(views))
    rng = np.random.default_rng(0)
    nb = max(2, int(np.ceil(mesh.edge_px / 0.7)))
    bary = bary_grid(min(nb, 10), mesh.dev)
    t0 = time.time()
    for it in range(iters):
        if it % a.consensus_every == 0:
            cov_ok = consensus(mesh, X.detach(), views, dmax, bary, a.cov_consensus)
            if it == 0:
                log(f"[{tag}] {int(cov_ok.sum())} vertices too thin by consensus")
        p = 0.5 / len(views) + 0.5 * ema / ema.sum()
        batch = rng.choice(len(views), size=min(a.batch, len(views)), replace=False, p=p / p.sum())
        e_band = 0.0
        for b in batch:
            g, rc, rv, deform, outl, _ = residuals(mesh, X, views[b], dmin, dmax, bary, floor, cov_ok, a.cont_tol)
            lc = huber(rc / tau).sum()
            lv = huber(rv / tau).sum()
            e_band = e_band + (a.w_cont * lc + a.w_cov * lv) / len(batch)
            ema[b] = 0.8 * ema[b] + 0.2 * (lc + lv).item() / max(1.0, float(deform.sum())) * 1e3 + 1e-6
        e_arap = mesh.arap(X) / tau ** 2
        e_lap = mesh.lap(X) / tau ** 2
        e_lmk = 0.0
        for grp, off in lmk:
            e_lmk = e_lmk + ((X[grp].mean(0) - mesh.V0[grp].mean(0) - off) ** 2).sum() / tau ** 2
        e_floor = huber(torch.relu(-floor.height(X)) / tau).sum() if floor is not None else 0.0
        loss = e_band + a.w_arap * e_arap + a.w_lap * e_lap + a.w_lmk * e_lmk + a.w_floor * e_floor
        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            X.grad[torch.as_tensor(mesh.rigid, device=mesh.dev)] = 0
        opt.step()
        sched.step()
        if it % 100 == 0 or it == iters - 1:
            lm = float(e_lmk.detach()) if torch.is_tensor(e_lmk) else e_lmk
            mv = float((X.detach() - mesh.V0).norm(dim=1).max()) * 1000
            log(f"[{tag}] it {it:4d} band {float(e_band.detach()):9.3f} arap {e_arap.item():8.3f} "
                f"lap {e_lap.item():8.3f} lmk {lm:7.3f} too thin {int(cov_ok.sum())} move max {mv:5.1f} mm "
                f"({time.time() - t0:.0f}s)")
    return mesh.place_rigid(X.detach())


def evaluate(mesh, X, views, dmin, dmax, reg, a, floor=None):
    bary = bary_grid(min(max(2, int(np.ceil(mesh.edge_px / 0.7))), 10), mesh.dev)
    per_view, gaps, sils = [], {r: [] for r in np.unique(reg)}, {}
    for v in views:
        with torch.no_grad():
            g, rc, rv, deform, outl, S = residuals(mesh, X, v, dmin, dmax, bary, floor)
        nd, no = max(1, int(deform.sum())), max(1, int(outl.sum()))
        per_view.append({"view": v["name"],
                         "outside_frac": round(float(((rc > 0.002) & deform).sum()) / nd, 4),
                         "outside_mm_max": round(float(rc.max()) * 1000, 1),
                         "too_thin_frac": round(float(((rv > 0.003) & outl).sum()) / no, 4),
                         "outline_gap_mm_p50": round(float(g[outl].median()) * 1000, 1) if outl.any() else None})
        go = g[outl].cpu().numpy()
        ro = reg[outl.cpu().numpy()]
        for r in gaps:
            gaps[r].append(go[ro == r])
        sils[v["name"]] = S[0, 0].cpu().numpy() > 0.5
    gap_stats = {r: np.round(np.percentile(np.concatenate(x) * 1000, [10, 50, 90]), 1).tolist()
                 for r, x in gaps.items() if sum(len(y) for y in x) > 20}
    summ = {"outside_frac_mean": round(float(np.mean([p["outside_frac"] for p in per_view])), 4),
            "below_floor_mm": round(float(torch.relu(-floor.height(X)).max()) * 1000, 1) if floor is not None else None,
            "too_thin_frac_mean": round(float(np.mean([p["too_thin_frac"] for p in per_view])), 4),
            "worst_views_outside": sorted(per_view, key=lambda p: -p["outside_frac"])[:5],
            "outline_gap_mm_p10_50_90_by_region": gap_stats}
    return per_view, summ, sils


def overlays(views, data_dir, sil_before, sil_after, out_dir, grid_views):
    os.makedirs(os.path.join(out_dir, "qa"), exist_ok=True)
    tiles = []
    for v in views:
        img = cv2.imread(os.path.join(data_dir, "images_2", f"{v['name']}.png"))
        img = cv2.resize(img, (v["W"], v["H"]), interpolation=cv2.INTER_AREA)
        img = (0.75 * img).astype(np.uint8)
        for m, col, th in [(v["mask"], (0, 220, 0), 2), (sil_before[v["name"]], (255, 120, 0), 2),
                           (sil_after.get(v["name"]), (0, 0, 255), 2)]:
            if m is None:
                continue
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, col, th)
        cv2.putText(img, v["name"], (12, 40), 0, 1.2, (255, 255, 255), 3)
        small = cv2.resize(img, (v["W"] // 2, v["H"] // 2), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out_dir, "qa", f"{v['name']}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if v["name"] in grid_views:
            tiles.append(cv2.resize(small, (300, int(300 * small.shape[0] / small.shape[1]))))
    if tiles:
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
        rows = [np.concatenate(tiles[i:i + 6] + [np.zeros_like(tiles[0])] * (6 - len(tiles[i:i + 6])), 1)
                for i in range(0, len(tiles), 6)]
        cv2.imwrite(os.path.join(out_dir, "qa_grid.jpg"), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])


def write_obj_like(src, X, dst):
    """Rewrite the vertex lines of src with X (keeps faces, UVs, groups)."""
    it = iter(X)
    with open(src) as fi, open(dst, "w") as fo:
        for ln in fi:
            if ln.startswith("v "):
                p = next(it)
                fo.write(f"v {p[0]:.7f} {p[1]:.7f} {p[2]:.7f}\n")
            else:
                fo.write(ln)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="template furless mesh (OBJ) in the capture frame")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--band", default=DEFAULT_BAND, help="region:d_min:d_max,... fur depth range in metres")
    ap.add_argument("--landmarks", default="", help='json: [{"name", "source": [x,y,z], "target": [x,y,z], "k": 12}]')
    ap.add_argument("--work_scale", type=float, default=0.5, help="mask resolution relative to images_2")
    ap.add_argument("--iters", default="800,400", help="iterations at the template level and the subdivided level")
    ap.add_argument("--batch", type=int, default=4, help="views per step")
    ap.add_argument("--lr", type=float, default=3e-4, help="Adam step (metres)")
    ap.add_argument("--tau", type=float, default=0.005, help="residual scale (metres)")
    ap.add_argument("--w_cont", type=float, default=1.0)
    ap.add_argument("--w_cov", type=float, default=1.0)
    ap.add_argument("--w_arap", type=float, default=0.3)
    ap.add_argument("--w_lap", type=float, default=0.5)
    ap.add_argument("--w_lmk", type=float, default=10.0)
    ap.add_argument("--up", default="", help="axis pointing up (e.g. -y): floor constraint and stage occlusion")
    ap.add_argument("--occl_height", type=float, default=0.05,
                    help="cameras below the floor skip containment for points up to this height above it")
    ap.add_argument("--w_floor", type=float, default=10.0)
    ap.add_argument("--cont_tol", type=float, default=0.003, help="containment tolerance (metres): mask accuracy")
    ap.add_argument("--cov_consensus", type=float, default=0.5,
                    help="fraction of a vertex's outline views that must find it too thin")
    ap.add_argument("--consensus_every", type=int, default=100)
    ap.add_argument("--small_components", type=float, default=0.05)
    ap.add_argument("--accessory_dilate", type=int, default=6, help="pixels at --work_scale")
    ap.add_argument("--grid_views", default="", help="views for qa_grid.jpg (default: 12 spread over all)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logf = open(os.path.join(a.out_dir, "fit.log"), "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    V, F, _ = load_obj_mesh(a.mesh)
    V = V.astype(np.float64)
    views = load_views(a.data_dir, a.work_scale, a.accessory_dilate, dev)
    band = parse_band(a.band)
    iters = [int(x) for x in a.iters.split(",")]
    log(f"{len(views)} views at {views[0]['W']}x{views[0]['H']}, mesh {len(V)} vertices {len(F)} faces")

    def setup(Vl, Fl, regl):
        mesh = Mesh(Vl, Fl, a.small_components, dev)
        mesh.edge_px = edge_px(mesh, mesh.V0, views)
        lo = np.array([band.get(r, (0.0, 0.03))[0] for r in regl], np.float32)
        hi = np.array([band.get(r, (0.0, 0.03))[1] for r in regl], np.float32)
        lmk = []
        if a.landmarks:
            dv = np.flatnonzero(~mesh.rigid)
            tree = cKDTree(Vl[dv])
            for L in json.load(open(a.landmarks)):
                grp = dv[tree.query(L["source"], k=int(L.get("k", 12)))[1]]
                lmk.append((torch.as_tensor(grp, device=dev),
                            torch.as_tensor(np.subtract(L["target"], L["source"]), dtype=torch.float32, device=dev)))
        return mesh, torch.as_tensor(lo, device=dev), torch.as_tensor(hi, device=dev), lmk

    reg0 = vertex_regions(a.regions, V)
    names, counts = np.unique(reg0, return_counts=True)
    log("regions " + json.dumps(dict(zip(names.tolist(), counts.tolist()))))
    mesh, dmin, dmax, lmk = setup(V, F, reg0)
    log(f"template edge ~{mesh.edge_px:.1f} px; {len(mesh.pieces)} rigid pieces; {len(lmk)} landmarks")
    floor = Floor(a.up, mesh.V0, views, a.occl_height, dev) if a.up else None
    if floor is not None:
        log(f"floor at {floor.h0:.4f} along {a.up}; {sum(floor.low.values())} views look up at it: "
            + ",".join(k for k, lo in floor.low.items() if lo))
    pv0, s0, sil0 = evaluate(mesh, mesh.V0, views, dmin, dmax, reg0, a, floor)
    log("template: " + json.dumps({k: v for k, v in s0.items() if k != "worst_views_outside"}))
    report = {"band": band, "template": {"summary": s0, "per_view": pv0}}

    X1 = fit_level(mesh, views, dmin, dmax, lmk, a, iters[0], log, "level0", floor)
    pv1, s1, sil1 = evaluate(mesh, X1, views, dmin, dmax, reg0, a, floor)
    log("level0: " + json.dumps({k: v for k, v in s1.items() if k != "worst_views_outside"}))
    report["level0"] = {"summary": s1, "per_view": pv1}
    X1n = X1.cpu().numpy().astype(np.float64)
    write_obj_like(a.mesh, X1n, os.path.join(a.out_dir, "furless_fit.obj"))
    final_sil = sil1

    if len(iters) > 1 and iters[1] > 0:
        Vs, Fs = trimesh.remesh.subdivide_loop(X1n, F, iterations=1)
        reg1 = reg0[cKDTree(X1n).query(Vs)[1]]
        mesh2, dmin2, dmax2, lmk2 = setup(Vs, Fs, reg1)
        X2 = fit_level(mesh2, views, dmin2, dmax2, lmk2, a, iters[1], log, "level1", floor)
        pv2, s2, sil2 = evaluate(mesh2, X2, views, dmin2, dmax2, reg1, a, floor)
        log("level1: " + json.dumps({k: v for k, v in s2.items() if k != "worst_views_outside"}))
        report["level1"] = {"summary": s2, "per_view": pv2}
        trimesh.Trimesh(X2.cpu().numpy(), Fs, process=False).export(os.path.join(a.out_dir, "furless_fit_fine.obj"))
        np.savez_compressed(os.path.join(a.out_dir, "furless_fit_fine_regions.npz"), vertex_region=reg1)
        final_sil = sil2
    json.dump(report, open(os.path.join(a.out_dir, "report.json"), "w"), indent=1)
    names = [v["name"] for v in views]
    grid = a.grid_views.split(",") if a.grid_views else names[:: max(1, len(names) // 12)][:12]
    overlays(views, a.data_dir, sil0, final_sil if iters[0] > 0 else {}, a.out_dir, set(grid))
    log(f"wrote {a.out_dir}")


if __name__ == "__main__":
    main()
