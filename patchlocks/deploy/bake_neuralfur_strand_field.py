"""Bake NeuralFur explicit strands into per-face surface fields.

The official NeuralFur reconstruction results provide final strands plus a
furless mesh, but not the training-time orientation maps. This script projects
strand roots back to the mesh and estimates the fields our patch controller
needs: tangent flow, length, density, color, and distance-to-fur.
"""
import argparse
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def unit(v, eps=1e-9):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fur", required=True, help="surfpack NPZ with strands")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--fur_dist", type=float, default=0.009, help="fur-mask threshold as fraction of bbox diagonal")
    args = ap.parse_args()

    mesh = trimesh.load(args.mesh, process=False)
    F = len(mesh.faces)
    fn = np.asarray(mesh.face_normals, np.float32)
    centers = np.asarray(mesh.triangles_center, np.float32)
    diag = float(np.linalg.norm(mesh.extents))

    d = np.load(args.fur)
    S = d["strands"].astype(np.float32)
    roots = S[:, 0]
    tip = S[:, -1]
    arc = np.linalg.norm(np.diff(S, axis=1), axis=-1).sum(1).astype(np.float32)
    col = d["color"].astype(np.float32) if "color" in d.files else np.full((len(S), 3), 0.55, np.float32)

    root_face = np.empty(len(S), np.int32)
    root_dist = np.empty(len(S), np.float32)
    for lo in range(0, len(S), args.chunk):
        hi = min(len(S), lo + args.chunk)
        _, dist, fid = trimesh.proximity.closest_point(mesh, roots[lo:hi])
        root_face[lo:hi] = fid.astype(np.int32)
        root_dist[lo:hi] = dist.astype(np.float32)

    cnt = np.bincount(root_face, minlength=F).astype(np.float32)
    sum_len = np.bincount(root_face, weights=arc, minlength=F).astype(np.float32)
    face_len = np.divide(sum_len, np.maximum(cnt, 1), out=np.zeros(F, np.float32), where=cnt > 0)

    v = tip - roots
    n = fn[root_face]
    flow = v - (v * n).sum(1, keepdims=True) * n
    flow = unit(flow)
    sum_flow = np.zeros((F, 3), np.float32)
    np.add.at(sum_flow, root_face, flow * arc[:, None])
    face_flow = unit(sum_flow)

    sum_col = np.zeros((F, 3), np.float32)
    np.add.at(sum_col, root_face, col)
    face_color = np.divide(sum_col, np.maximum(cnt[:, None], 1), out=np.zeros((F, 3), np.float32), where=cnt[:, None] > 0)

    observed = np.flatnonzero(cnt > 0)
    if len(observed):
        tree = cKDTree(centers[observed])
        dist_face, nn = tree.query(centers, k=1, workers=-1)
        src = observed[nn]
        empty = cnt <= 0
        face_len[empty] = face_len[src[empty]]
        face_flow[empty] = face_flow[src[empty]]
        face_color[empty] = face_color[src[empty]]
    else:
        dist_face = np.full(F, np.inf, np.float32)
        face_flow[:] = np.array([1.0, 0.0, 0.0], np.float32)
        face_color[:] = np.array([0.55, 0.50, 0.43], np.float32)

    nz = cnt[cnt > 0]
    rho_scale = float(np.percentile(nz, 95)) if len(nz) else 1.0
    face_rho = np.clip(cnt / max(rho_scale, 1e-6), 0, 1).astype(np.float32)
    face_furmask = (dist_face < args.fur_dist * diag).astype(np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(
        args.out,
        face_flow=face_flow.astype(np.float32),
        face_len=face_len.astype(np.float32),
        face_rho=face_rho,
        face_color=face_color.astype(np.float32),
        face_count=cnt.astype(np.float32),
        face_furmask=face_furmask.astype(np.float32),
        face_dist_to_root=dist_face.astype(np.float32),
        root_face=root_face,
        root_dist=root_dist,
        bbox_diag=np.float32(diag),
        source_fur=os.path.abspath(args.fur),
        source_mesh=os.path.abspath(args.mesh),
    )
    np.save(os.path.splitext(args.out)[0] + "_flow.npy", face_flow.astype(np.float32))
    print(
        f"saved {args.out} | faces={F} roots={len(S)} observed_faces={int((cnt>0).sum())} "
        f"root_dist mean/max={root_dist.mean():.2e}/{root_dist.max():.2e} "
        f"len med={np.median(arc):.5f} face_len med={np.median(face_len):.5f} "
        f"furmask={face_furmask.mean()*100:.1f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
