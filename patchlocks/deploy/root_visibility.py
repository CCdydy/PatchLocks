"""Front-visible root filter for high-surface patch optimization.

Default mode is robust front-face culling from smooth root normals. Optional
low-resolution vertex z-buffer occlusion can be enabled, but it is deliberately
off by default because point-splat depth is conservative and can over-delete.

Drop-in for patch_graph_opt_geodesic.py's multi-view loop:
    vis, ff, occ, (px, py, w) = front_visible(root_pos, root_normals, mesh_verts, matrix, W, H, diag)
    keep = vis & (body_mask[by, bx] > 0.05)
"""
import numpy as np, torch
from numba import njit
dev = "cuda" if torch.cuda.is_available() else "cpu"

def _project(points, M):
    """world->clip via points@M (matches patch_graph_opt_geodesic.project_row). Returns px,py (pixel, in the
    matrix's native W,H NDC mapping is applied by caller) plus w=clip-w as a monotone DEPTH, and the front-flag."""
    homo = np.concatenate([points.astype(np.float32), np.ones((len(points), 1), np.float32)], 1)
    clip = homo @ M; w = clip[:, 3]; ok = np.abs(w) > 1e-6
    ndc = np.zeros((len(points), 3), np.float32); ndc[ok] = clip[ok, :3]/w[ok, None]
    return ndc, w, ok

def _camera_center(M):
    """world->clip is clip = homo @ M. Recover the camera center by intersecting two back-projected rays."""
    Minv = np.linalg.inv(M.astype(np.float64))
    def unp(nx, ny, nz): h = np.array([nx, ny, nz, 1.0]) @ Minv; return (h[:3]/h[3])
    def ray(nx, ny): a = unp(nx, ny, -0.5); b = unp(nx, ny, 0.5); d = b-a; return a, d/np.linalg.norm(d)
    a1, d1 = ray(0.0, 0.0); a2, d2 = ray(0.55, 0.35)
    A = np.stack([d1, -d2], 1); t = np.linalg.lstsq(A, a2-a1, rcond=None)[0]
    return (0.5*((a1+t[0]*d1)+(a2+t[1]*d2))).astype(np.float32)


@njit(cache=True)
def _rasterize_tri_zbuf(px, py, depth, faces, rw, rh):
    zbuf = np.full((rh, rw), np.inf, np.float32)
    nf = faces.shape[0]
    for fi in range(nf):
        i0 = faces[fi, 0]
        i1 = faces[fi, 1]
        i2 = faces[fi, 2]
        x0 = px[i0]
        y0 = py[i0]
        z0 = depth[i0]
        x1 = px[i1]
        y1 = py[i1]
        z1 = depth[i1]
        x2 = px[i2]
        y2 = py[i2]
        z2 = depth[i2]
        if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(z0) and np.isfinite(x1) and np.isfinite(y1) and np.isfinite(z1) and np.isfinite(x2) and np.isfinite(y2) and np.isfinite(z2)):
            continue
        minx = int(np.floor(min(x0, min(x1, x2))))
        maxx = int(np.ceil(max(x0, max(x1, x2))))
        miny = int(np.floor(min(y0, min(y1, y2))))
        maxy = int(np.ceil(max(y0, max(y1, y2))))
        if maxx < 0 or maxy < 0 or minx >= rw or miny >= rh:
            continue
        if minx < 0:
            minx = 0
        if miny < 0:
            miny = 0
        if maxx >= rw:
            maxx = rw - 1
        if maxy >= rh:
            maxy = rh - 1
        area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        if abs(area) < 1e-10:
            continue
        inv_area = 1.0 / area
        for yy in range(miny, maxy + 1):
            cy = yy + 0.5
            for xx in range(minx, maxx + 1):
                cx = xx + 0.5
                w0 = ((x1 - cx) * (y2 - cy) - (y1 - cy) * (x2 - cx)) * inv_area
                w1 = ((x2 - cx) * (y0 - cy) - (y2 - cy) * (x0 - cx)) * inv_area
                w2 = 1.0 - w0 - w1
                if w0 >= -1e-5 and w1 >= -1e-5 and w2 >= -1e-5:
                    zz = w0 * z0 + w1 * z1 + w2 * z2
                    if zz < zbuf[yy, xx]:
                        zbuf[yy, xx] = zz
    return zbuf


def _mesh_zbuffer(mesh_verts, mesh_faces, M, W, H, zbuf_w):
    rw = int(zbuf_w)
    rh = max(1, int(round(H * zbuf_w / W)))
    ndcV, wV, okV = _project(mesh_verts, M)
    pxV = (ndcV[:, 0] * 0.5 + 0.5) * rw
    pyV = (1.0 - (ndcV[:, 1] * 0.5 + 0.5)) * rh
    pxV = np.where(okV, pxV, np.nan).astype(np.float32)
    pyV = np.where(okV, pyV, np.nan).astype(np.float32)
    depth = np.where(okV, wV, np.nan).astype(np.float32)
    return _rasterize_tri_zbuf(pxV, pyV, depth, mesh_faces.astype(np.int64), rw, rh), rw, rh


def mesh_silhouette(mesh_verts, mesh_faces, M, W, H, zbuf_w=None):
    """Rasterize a mesh silhouette in the image plane.

    This is for camera/mesh QA and hard alignment gating. It uses the same
    row-vector projection convention as front_visible.
    """
    if zbuf_w is None:
        zbuf_w = int(W)
    zbuf, rw, rh = _mesh_zbuffer(mesh_verts, mesh_faces, M, W, H, zbuf_w)
    sil = np.isfinite(zbuf).astype(np.float32)
    if rw != W or rh != H:
        import cv2
        sil = cv2.resize(sil, (int(W), int(H)), interpolation=cv2.INTER_AREA)
    return sil


def silhouette_iou(pred, target, thr=0.5):
    p = np.asarray(pred) > thr
    t = np.asarray(target) > thr
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return float(inter) / max(float(union), 1.0)

def front_visible(root_pos, root_normals, mesh_verts, M, W, H, diag, zbuf_w=900, foot=1, tol_gross=0.07, occlusion=False, mesh_faces=None):
    # ---- (1) FRONT-FACE cull (exact geometric, ROBUST): camera on the same side as the SMOOTH outward normal ----
    C = _camera_center(M)
    frontface = ((C[None]-root_pos)*root_normals.astype(np.float32)).sum(1) > 0
    ndcR, wR, okR = _project(root_pos, M)
    px_native = (ndcR[:, 0]*0.5+0.5)*W
    py_native = (1.0-(ndcR[:, 1]*0.5+0.5))*H
    inR_native = okR & (px_native >= 0) & (px_native < W) & (py_native >= 0) & (py_native < H)
    if not occlusion:
        occluded = np.zeros(len(root_pos), bool)
        vis = inR_native & frontface
        return vis, frontface, occluded, (px_native.astype(np.float32), py_native.astype(np.float32), wR)

    if mesh_faces is not None:
        zbuf, rw, rh = _mesh_zbuffer(mesh_verts, mesh_faces, M, W, H, zbuf_w)
        pxR = (ndcR[:, 0] * 0.5 + 0.5) * rw
        pyR = (1.0 - (ndcR[:, 1] * 0.5 + 0.5)) * rh
        inR = okR & (pxR >= 0) & (pxR < rw) & (pyR >= 0) & (pyR < rh)
        xi = np.clip(pxR, 0, rw - 1).astype(np.int64)
        yi = np.clip(pyR, 0, rh - 1).astype(np.int64)
        bw = zbuf[yi, xi]
        finite = np.isfinite(bw)
        tol = max(0.004 * float(diag), 0.006 * (float(np.nanmax(wR)) - float(np.nanmin(wR))))
        occluded = inR & finite & (wR > bw + tol)
        vis = inR & frontface & ~occluded
        return vis, frontface, occluded, (pxR * W / rw, pyR * H / rh, wR)

    # ---- (2) LENIENT vertex z-buffer fallback (low res, holes -> NOT occluded) ----
    rw = int(zbuf_w); rh = max(1, int(round(H*zbuf_w/W)))
    ndcV, wV, okV = _project(mesh_verts, M); pxV = (ndcV[:, 0]*0.5+0.5)*rw; pyV = (1.0-(ndcV[:, 1]*0.5+0.5))*rh
    inV = okV & (pxV >= 0) & (pxV < rw) & (pyV >= 0) & (pyV < rh)
    pxv = torch.tensor(pxV[inV], device=dev); pyv = torch.tensor(pyV[inV], device=dev); wv = torch.tensor(wV[inV], device=dev)
    buf = torch.full((rh*rw,), float("inf"), device=dev)
    for dy in range(-foot, foot+1):
        for dx in range(-foot, foot+1):
            xi = (pxv+dx).long().clamp(0, rw-1); yi = (pyv+dy).long().clamp(0, rh-1); buf.scatter_reduce_(0, yi*rw+xi, wv, reduce="amin", include_self=True)
    buf = buf.reshape(rh, rw)
    pxR = (ndcR[:, 0]*0.5+0.5)*rw; pyR = (1.0-(ndcR[:, 1]*0.5+0.5))*rh
    inR = okR & (pxR >= 0) & (pxR < rw) & (pyR >= 0) & (pyR < rh)
    xi = np.clip(pxR, 0, rw-1).astype(np.int64); yi = np.clip(pyR, 0, rh-1).astype(np.int64)
    bw = buf[torch.tensor(yi, device=dev), torch.tensor(xi, device=dev)].cpu().numpy()
    occluded = (np.isfinite(bw) & (wR > bw + tol_gross*(wV[inV].max()-wV[inV].min()))) if occlusion else np.zeros(len(root_pos), bool)
    vis = inR & frontface & ~occluded                                                  # front-face is the robust filter; occlusion is the conservative add-on
    return vis, frontface, occluded, (pxR*W/rw, pyR*H/rh, wR)

if __name__ == "__main__":
    import cv2, pickle, glob, trimesh, os
    NF = "patchlocks/datasets/neuralfur/extracted/data/Artemis/panda_processed_GH2/walk"
    d = np.load("patchlocks/deploy/neuralfur_geopatch_furless/geodesic_patchpack.npz", allow_pickle=True)
    root_pos = d["root_pos"].astype(np.float32); diag = float(d["bbox_diag"])
    mats = pickle.load(open(f"{NF}/3d_gaussian_splatting/GS_/cameras/30000_matrices.pkl", "rb")); view = sorted(mats.keys())[0]
    M = np.asarray(mats[view].detach().cpu().numpy() if hasattr(mats[view], "detach") else mats[view], np.float32)
    img = cv2.cvtColor(cv2.imread(f"{NF}/images_2/{view}.png"), cv2.COLOR_BGR2RGB); H0, W0 = img.shape[:2]
    body = cv2.imread(f"{NF}/masks_2/body/{view}.png", 0).astype(np.float32)/255
    mesh = trimesh.load(str(d["mesh_path"]), process=False)
    V = np.asarray(mesh.vertices, np.float32); faces = np.asarray(mesh.faces); vn = np.asarray(mesh.vertex_normals, np.float32)   # SMOOTH normals (MC face normals are noisy)
    rb = d["root_bary"].astype(np.float32); rn = (vn[faces[d["root_face"]]]*rb[:, :, None]).sum(1); root_normals = rn/np.maximum(np.linalg.norm(rn, axis=1, keepdims=True), 1e-9)
    # naive keep (inb + body mask) vs +visibility
    vis, ff, occ, (pxR, pyR, wR) = front_visible(root_pos, root_normals, V, M, W0, H0, diag)
    inb = (pxR >= 0) & (pxR < W0) & (pyR >= 0) & (pyR < H0)
    bx = np.clip(pxR, 0, W0-1).astype(int); by = np.clip(pyR, 0, H0-1).astype(int); mval = body[by, bx]
    naive = inb & (mval > 0.05); full = naive & vis
    print(f"view {view} | naive(proj+mask) {naive.sum()} | front-face {(naive&ff).sum()} ({100*(naive&ff).sum()/naive.sum():.0f}%) | +occlusion removes {(naive&ff&occ).sum()} more | front-VISIBLE {full.sum()} ({100*full.sum()/naive.sum():.0f}%)")
    can = img.copy()
    for x, y in zip(bx[naive & ~full][::20], by[naive & ~full][::20]): cv2.circle(can, (x, y), 2, (255, 0, 0), -1)   # removed = RED
    for x, y in zip(bx[full][::20], by[full][::20]): cv2.circle(can, (x, y), 2, (0, 220, 0), -1)                     # kept = GREEN
    cv2.imwrite("patchlocks/deploy/_root_visibility.png", cv2.cvtColor(cv2.resize(can, (W0//2, H0//2)), cv2.COLOR_RGB2BGR))
    print("saved _root_visibility.png (GREEN=front-visible kept, RED=removed backface/occluded)")
