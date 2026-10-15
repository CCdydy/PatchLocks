"""Build V13 pose sidecars for an existing surface_patch_dataset.npz.

The surface pack already contains geometry-aligned image projection inputs
(`uv`, `vis`) and local patch frames (`Tt`, `Bb`, `Nn`).  V13 additionally
needs the observation direction expressed in the same local TBN coordinate
system as the z64 strand latent.

Outputs are sidecar arrays, not a rewritten surfpack:
  raydir_tbn.npy  (P,K,H,H,3): unit ray from surface texel to camera in TBN
  pose_view.npy   (P,K,12):    patch/view pose token features
  v13_pose_meta.npz / qa_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np


POSE_NAMES = np.asarray(
    [
        "dir_t",
        "dir_b",
        "dir_n",
        "abs_dir_n",
        "asin_dir_n",
        "cam_t_over_psize",
        "cam_b_over_psize",
        "cam_n_over_psize",
        "log_dist_over_psize",
        "azimuth_cos",
        "azimuth_sin",
        "bias",
    ]
)


def _scalar(x, default=None):
    if x is None:
        return default
    a = np.asarray(x)
    if a.shape == ():
        return a.item()
    if a.size == 1:
        return a.reshape(-1)[0].item()
    return a


def _pack_string(pack, key, default=""):
    if key not in pack.files:
        return default
    v = _scalar(pack[key], default)
    return str(v)


def _unit(x, axis=-1, eps=1e-9):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def load_matrices(path):
    if not path:
        return None
    path = str(path)
    if not os.path.exists(path):
        return None
    mm = pickle.load(open(path, "rb"))
    return mm["matrices"] if isinstance(mm, dict) and "matrices" in mm else mm


def matrix_for_view(mats, view_id, index):
    if mats is None:
        return None
    keys = [view_id, str(view_id), int(view_id) if str(view_id).isdigit() else None, index]
    for key in keys:
        if key is not None and isinstance(mats, dict) and key in mats:
            return np.asarray(mats[key], np.float64)
    if not isinstance(mats, dict) and index < len(mats):
        return np.asarray(mats[index], np.float64)
    return None


def camera_center_from_clip(M):
    """Row-vector homogeneous camera center: C @ M has x=y=w=0."""
    A = M[:, [0, 1, 3]].T
    _u, _s, vh = np.linalg.svd(A)
    c = vh[-1]
    den = c[3] if abs(c[3]) > 1e-8 else (1e-8 if c[3] >= 0 else -1e-8)
    return (c[:3] / den).astype(np.float32)


def auto_matrices_path(pack, explicit):
    if explicit:
        return explicit
    data_dir = _pack_string(pack, "data_dir", "")
    if not data_dir:
        return ""
    cand = os.path.join(data_dir, "3d_gaussian_splatting", "GS_", "cameras", "30000_matrices.pkl")
    return cand if os.path.exists(cand) else ""


def build_texel_tbn(normg, t_hint, b_hint):
    n = _unit(normg)
    th = np.asarray(t_hint, np.float32).reshape(1, 1, 3)
    bh = np.asarray(b_hint, np.float32).reshape(1, 1, 3)

    t = th - n * np.sum(n * th, axis=-1, keepdims=True)
    bad = np.linalg.norm(t, axis=-1) < 1e-6
    if np.any(bad):
        tb = bh - n * np.sum(n * bh, axis=-1, keepdims=True)
        t[bad] = tb[bad]

    bad = np.linalg.norm(t, axis=-1) < 1e-6
    if np.any(bad):
        ax = np.zeros_like(n)
        use_y = np.abs(n[..., 0]) > 0.8
        ax[..., 0] = 1.0
        ax0 = ax[..., 0]
        ax1 = ax[..., 1]
        ax0[use_y] = 0.0
        ax1[use_y] = 1.0
        ax[..., 0] = ax0
        ax[..., 1] = ax1
        tf = ax - n * np.sum(n * ax, axis=-1, keepdims=True)
        t[bad] = tf[bad]

    t = _unit(t)
    b = _unit(np.cross(n, t))
    t = _unit(np.cross(b, n))
    return t, b, n


def pose_features(cam, center, t, b, n, psize):
    rel = np.asarray(cam, np.float32) - np.asarray(center, np.float32)
    dist = float(np.linalg.norm(rel) + 1e-9)
    d = rel / dist
    ct = float(np.dot(d, t))
    cb = float(np.dot(d, b))
    cn = float(np.dot(d, n))
    az = float(np.arctan2(cb, ct))
    denom = max(float(psize), 1e-9)
    return np.asarray(
        [
            ct,
            cb,
            cn,
            abs(cn),
            np.arcsin(np.clip(cn, -1.0, 1.0)),
            float(np.dot(rel, t)) / denom,
            float(np.dot(rel, b)) / denom,
            float(np.dot(rel, n)) / denom,
            np.log(dist / denom + 1e-9),
            np.cos(az),
            np.sin(az),
            1.0,
        ],
        np.float32,
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="surface_patch_dataset.npz")
    ap.add_argument("--out", default="", help="output directory; default: <pack_dir>/v13_pose")
    ap.add_argument("--matrices", default="", help="baked 30000_matrices.pkl; default reads pack data_dir")
    ap.add_argument("--chunk-patches", type=int, default=32)
    ap.add_argument("--max-patches", type=int, default=0, help="debug subset; 0 means all")
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    pack_path = Path(args.pack)
    out_dir = Path(args.out) if args.out else pack_path.with_name("v13_pose")
    out_dir.mkdir(parents=True, exist_ok=True)

    ray_path = out_dir / "raydir_tbn.npy"
    pose_path = out_dir / "pose_view.npy"
    if not args.overwrite and (ray_path.exists() or pose_path.exists()):
        raise FileExistsError(f"{out_dir} already has V13 arrays; pass --overwrite to replace")

    pack = np.load(pack_path, allow_pickle=True, mmap_mode="r")
    required = ["surf", "normg", "centers", "Tt", "Nn", "uv", "vis"]
    missing = [k for k in required if k not in pack.files]
    if missing:
        raise KeyError(f"pack is missing required keys: {missing}")

    surf = pack["surf"]
    normg = pack["normg"]
    centers = pack["centers"].astype(np.float32)
    Tt = pack["Tt"].astype(np.float32)
    Nn = pack["Nn"].astype(np.float32)
    Bb = pack["Bb"].astype(np.float32) if "Bb" in pack.files else _unit(np.cross(Nn, Tt))
    valid = pack["valid"].astype(np.float32) if "valid" in pack.files else np.ones(surf.shape[:3], np.float32)
    vis = pack["vis"]
    psize = float(_scalar(pack["psize"], 1.0)) if "psize" in pack.files else 1.0

    P_full, H, W = surf.shape[:3]
    K = int(vis.shape[1])
    if H != W:
        raise ValueError(f"expected square surf grid, got {H}x{W}")
    P = min(P_full, args.max_patches) if args.max_patches > 0 else P_full

    if "view_ids" in pack.files:
        view_ids = [_scalar(x) for x in np.asarray(pack["view_ids"]).reshape(-1)]
    else:
        view_ids = list(range(K))
    if len(view_ids) < K:
        view_ids = view_ids + list(range(len(view_ids), K))

    matrices_path = auto_matrices_path(pack, args.matrices)
    mats = load_matrices(matrices_path)
    cam_centers = []
    cam_valid = []
    for k in range(K):
        M = matrix_for_view(mats, view_ids[k], k)
        if M is None:
            cam_centers.append(np.zeros(3, np.float32))
            cam_valid.append(False)
        else:
            cam_centers.append(camera_center_from_clip(M))
            cam_valid.append(True)
    cam_centers = np.asarray(cam_centers, np.float32)
    cam_valid = np.asarray(cam_valid, bool)
    if not np.all(cam_valid):
        bad = [str(view_ids[i]) for i, ok in enumerate(cam_valid) if not ok][:8]
        raise RuntimeError(f"missing camera centers for {np.sum(~cam_valid)}/{K} views: {bad}")

    out_dtype = np.float16 if args.dtype == "float16" else np.float32
    ray = np.lib.format.open_memmap(ray_path, mode="w+", dtype=out_dtype, shape=(P, K, H, W, 3))
    pose = np.lib.format.open_memmap(pose_path, mode="w+", dtype=np.float32, shape=(P, K, len(POSE_NAMES)))

    visible_z_sum = 0.0
    visible_z_count = 0
    all_z_min = 1e9
    all_z_max = -1e9
    chunk = max(1, int(args.chunk_patches))
    for s0 in range(0, P, chunk):
        s1 = min(P, s0 + chunk)
        for p in range(s0, s1):
            t_map, b_map, n_map = build_texel_tbn(normg[p].astype(np.float32), Tt[p], Bb[p])
            for k in range(K):
                rw = cam_centers[k].reshape(1, 1, 3) - surf[p].astype(np.float32)
                rw = _unit(rw)
                rloc = np.stack(
                    [
                        np.sum(rw * t_map, axis=-1),
                        np.sum(rw * b_map, axis=-1),
                        np.sum(rw * n_map, axis=-1),
                    ],
                    axis=-1,
                ).astype(np.float32)
                ray[p, k] = rloc.astype(out_dtype)
                pose[p, k] = pose_features(cam_centers[k], centers[p], Tt[p], Bb[p], Nn[p], psize)

                z = rloc[..., 2]
                m = (valid[p] > 0.5) & (vis[p, k].astype(np.float32) > 0.5)
                if np.any(m):
                    visible_z_sum += float(z[m].sum())
                    visible_z_count += int(m.sum())
                all_z_min = min(all_z_min, float(z.min()))
                all_z_max = max(all_z_max, float(z.max()))
        print(f"  pose {s1}/{P}", flush=True)

    ray.flush()
    pose.flush()

    qa = {
        "source_pack": str(pack_path),
        "matrices": matrices_path or "",
        "P": int(P),
        "P_full": int(P_full),
        "K": int(K),
        "H": int(H),
        "raydir_tbn": str(ray_path),
        "raydir_dtype": args.dtype,
        "pose_view": str(pose_path),
        "pose_names": POSE_NAMES.tolist(),
        "camera_centers_valid": int(cam_valid.sum()),
        "raydir_tbn_z_min": all_z_min,
        "raydir_tbn_z_max": all_z_max,
        "visible_mean_dot_ray_n": visible_z_sum / max(visible_z_count, 1),
        "visible_samples": int(visible_z_count),
    }
    (out_dir / "qa_summary.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    np.savez(
        out_dir / "v13_pose_meta.npz",
        source_pack=np.asarray(str(pack_path)),
        matrices=np.asarray(matrices_path or ""),
        view_ids=np.asarray(view_ids[:K]),
        camera_centers=cam_centers,
        camera_valid=cam_valid,
        pose_names=POSE_NAMES,
        raydir_tbn_path=np.asarray(str(ray_path)),
        raydir_tbn_shape=np.asarray([P, K, H, W, 3], np.int64),
        raydir_tbn_dtype=np.asarray(args.dtype),
        pose_view_path=np.asarray(str(pose_path)),
        pose_view_shape=np.asarray([P, K, len(POSE_NAMES)], np.int64),
        psize=np.float32(psize),
    )
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    main()
