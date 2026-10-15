"""Prepare/check synthetic multi-view data for V13.

This script does not train a model.  It organizes render dirs into a manifest,
reports missing training targets, writes the commands needed to bake those
targets, and can add V13 view-geometry sidecars for flat synthetic patches.

Example:
  python3 patchlocks/data_gen/prepare_v13_mv_dataset.py \
    --renders patchlocks/datasets/v11_longfur/renders_mv \
    --out patchlocks/datasets/v11_longfur/v13_mv_prep \
    --latent-name target_latent64_v6codec.npz \
    --write-pose --write-commands
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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


def unit(x, axis=-1, eps=1e-9):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def pose_features(cam, psize):
    rel = np.asarray(cam, np.float32)
    dist = float(np.linalg.norm(rel) + 1e-9)
    d = rel / dist
    ct, cb, cn = float(d[0]), float(d[1]), float(d[2])
    az = float(np.arctan2(cb, ct))
    denom = max(float(psize), 1e-9)
    return np.asarray(
        [
            ct,
            cb,
            cn,
            abs(cn),
            np.arcsin(np.clip(cn, -1.0, 1.0)),
            float(rel[0]) / denom,
            float(rel[1]) / denom,
            float(rel[2]) / denom,
            np.log(dist / denom + 1e-9),
            np.cos(az),
            np.sin(az),
            1.0,
        ],
        np.float32,
    )


def iter_render_dirs(root):
    root = Path(root)
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "cameras.json").exists() and (d / "patch.npz").exists())


def count_rgb(d):
    n = 0
    while (d / f"rgb_{n}.png").exists():
        n += 1
    return n


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def shell0_grid(meta, H):
    size = float(meta.get("size", 1.0))
    ii, jj = np.meshgrid(np.arange(H), np.arange(H))
    ug = (ii + 0.5) / H
    vg = (jj + 0.5) / H
    surf = np.stack([(ug - 0.5) * size, (vg - 0.5) * size, np.zeros_like(ug)], axis=-1).astype(np.float32)
    return surf, size


def view_camera_location(view):
    if "location" in view:
        return np.asarray(view["location"], np.float32)
    if "matrix_world" in view:
        M = np.asarray(view["matrix_world"], np.float64)
        return M[:3, 3].astype(np.float32)
    raise KeyError("view has neither location nor matrix_world")


def frustum_vis(meta, surf):
    K = len(meta["views"])
    H = surf.shape[0]
    vis = np.zeros((K, H, H), np.float16)
    surf_h = np.concatenate([surf.reshape(-1, 3), np.ones((H * H, 1), np.float32)], axis=1)
    for k, cam in enumerate(meta["views"]):
        res = int(cam.get("res", 256))
        lens = float(cam.get("lens", 75.0))
        sensor_width = float(cam.get("sensor_width", 36.0))
        Minv = np.linalg.inv(np.asarray(cam["matrix_world"], np.float64))
        fpx = lens / sensor_width * res
        Xc = (surf_h @ Minv.T)[:, :3].reshape(H, H, 3)
        z = Xc[..., 2]
        front = z < 0
        zz = np.where(front, z, -1e-6)
        px = res / 2.0 + fpx * (Xc[..., 0] / -zz)
        py = res / 2.0 - fpx * (Xc[..., 1] / -zz)
        vis[k] = (front & (px >= 0) & (px < res) & (py >= 0) & (py < res)).astype(np.float16)
    return vis


def write_v13_pose(d, H, overwrite=False, dtype="float16"):
    out = d / f"v13_pose_{H}.npz"
    if out.exists() and not overwrite:
        return False
    meta = load_json(d / "cameras.json")
    surf, size = shell0_grid(meta, H)
    cams = np.stack([view_camera_location(v) for v in meta["views"]]).astype(np.float32)
    K = len(cams)
    ray = np.zeros((K, H, H, 3), np.float32)
    pose = np.zeros((K, len(POSE_NAMES)), np.float32)
    for k, cam in enumerate(cams):
        rw = unit(cam.reshape(1, 1, 3) - surf)
        ray[k] = rw
        pose[k] = pose_features(cam, size)
    vis = frustum_vis(meta, surf)
    rdtype = np.float16 if dtype == "float16" else np.float32
    np.savez(
        out,
        raydir_tbn=ray.astype(rdtype),
        pose_view=pose,
        pose_names=POSE_NAMES,
        vis_frustum=vis,
        camera_centers=cams,
        view_names=np.asarray([v.get("name", f"view{k}") for k, v in enumerate(meta["views"])]),
        H=np.int32(H),
        size=np.float32(size),
        visibility_source=np.asarray("frustum_only"),
        frame=np.asarray("flat_patch_world_equals_TBN"),
    )
    return True


def stable_split(paths, val_frac, seed):
    names = np.asarray([p.name for p in paths])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(names))
    val_n = int(round(len(names) * val_frac))
    val = set(names[order[:val_n]].tolist())
    return {name: ("val" if name in val else "train") for name in names}


def write_commands(out_dir, renders, latent_name, res, codec, python_cmd):
    repo_root = Path(__file__).resolve().parents[2]
    cmd = out_dir / "commands.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {repo_root}",
        "",
        f"{python_cmd} patchlocks/models/bake_latent_batch.py {renders} {res} --codec {codec} --out-name {latent_name}",
        f"{python_cmd} patchlocks/data_gen/build_field_gt.py {renders}",
        (
            f"{python_cmd} patchlocks/data_gen/prepare_v13_mv_dataset.py "
            f"--renders {renders} --out {out_dir} --latent-name {latent_name} "
            f"--res {res} --write-pose --write-commands"
        ),
        "",
    ]
    cmd.write_text("\n".join(lines), encoding="utf-8")
    cmd.chmod(0o755)
    return cmd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True, help="multi-view render directory")
    ap.add_argument("--out", required=True, help="prep output directory")
    ap.add_argument("--latent-name", default="target_latent64_v6codec.npz")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--codec", default="patchlocks/models/fur_strand_codec_v6upright.pt")
    ap.add_argument("--python-cmd", default="conda run -n difflocks python")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=0, help="debug subset for scan/write-pose")
    ap.add_argument("--write-pose", action="store_true")
    ap.add_argument("--pose-dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--overwrite-pose", action="store_true")
    ap.add_argument("--write-commands", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    renders = Path(args.renders).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dirs = iter_render_dirs(renders)
    if args.limit > 0:
        dirs = dirs[: args.limit]
    if not dirs:
        raise FileNotFoundError(f"no render dirs with patch.npz+cameras.json under {renders}")

    split = stable_split(dirs, args.val_frac, args.seed)
    records = []
    missing_lines = []
    pose_written = 0
    for i, d in enumerate(dirs):
        meta = load_json(d / "cameras.json")
        K = len(meta.get("views", []))
        rgb_n = count_rgb(d)
        latent_path = d / args.latent_name
        field_path = d / f"field_gt_{args.res}.npz"
        pose_path = d / f"v13_pose_{args.res}.npz"
        has_latent = latent_path.exists()
        has_field = field_path.exists()
        has_pose = pose_path.exists()
        if args.write_pose:
            wrote = write_v13_pose(d, args.res, overwrite=args.overwrite_pose, dtype=args.pose_dtype)
            pose_written += int(wrote)
            has_pose = pose_path.exists()
        missing = []
        if rgb_n < K:
            missing.append(f"rgb:{rgb_n}/{K}")
        if not has_latent:
            missing.append(args.latent_name)
        if not has_field:
            missing.append(f"field_gt_{args.res}.npz")
        if not has_pose:
            missing.append(f"v13_pose_{args.res}.npz")
        if missing:
            missing_lines.append(f"{d}\t{','.join(missing)}")
        records.append(
            {
                "name": d.name,
                "path": str(d),
                "split": split[d.name],
                "views": K,
                "rgb_count": rgb_n,
                "has_latent": bool(has_latent),
                "has_field_gt": bool(has_field),
                "has_v13_pose": bool(has_pose),
                "ready": bool(rgb_n >= K and has_latent and has_field and has_pose),
                "missing": missing,
            }
        )
        if (i + 1) % 100 == 0:
            print(f"  scanned {i+1}/{len(dirs)}", flush=True)

    manifest = {
        "renders": str(renders),
        "latent_name": args.latent_name,
        "res": int(args.res),
        "count": len(records),
        "ready_count": int(sum(r["ready"] for r in records)),
        "latent_count": int(sum(r["has_latent"] for r in records)),
        "field_count": int(sum(r["has_field_gt"] for r in records)),
        "pose_count": int(sum(r["has_v13_pose"] for r in records)),
        "pose_written": int(pose_written),
        "records": records,
    }
    (out_dir / "v13_mv_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with open(out_dir / "v13_mv_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["name", "path", "split", "views", "rgb_count", "has_latent", "has_field_gt", "has_v13_pose", "ready", "missing"],
        )
        w.writeheader()
        for r in records:
            row = dict(r)
            row["missing"] = ";".join(r["missing"])
            w.writerow(row)
    (out_dir / "missing_targets.txt").write_text("\n".join(missing_lines) + ("\n" if missing_lines else ""), encoding="utf-8")

    cmd = None
    if args.write_commands:
        cmd = write_commands(out_dir, renders, args.latent_name, args.res, args.codec, args.python_cmd)

    summary = {
        "count": manifest["count"],
        "ready_count": manifest["ready_count"],
        "latent_count": manifest["latent_count"],
        "field_count": manifest["field_count"],
        "pose_count": manifest["pose_count"],
        "pose_written": manifest["pose_written"],
        "manifest": str(out_dir / "v13_mv_manifest.json"),
        "missing": str(out_dir / "missing_targets.txt"),
        "commands": str(cmd) if cmd else "",
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
