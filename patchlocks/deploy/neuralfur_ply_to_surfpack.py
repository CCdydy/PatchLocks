"""Convert official NeuralFur strand PLY files to the local surfpack NPZ format.

NeuralFur stores explicit strands as a binary PLY point cloud where vertices are
ordered strand-major. Their README states 100 samples per strand.
"""
import argparse
import os

import numpy as np
import trimesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", type=int, default=100)
    ap.add_argument("--max_strands", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radius", type=float, default=0.0, help="0 = bbox_diag * radius_frac")
    ap.add_argument("--radius_frac", type=float, default=0.00032)
    args = ap.parse_args()

    pc = trimesh.load(args.ply, process=False)
    verts = np.asarray(pc.vertices, np.float32)
    if len(verts) % args.points:
        raise ValueError(f"{len(verts)} vertices is not divisible by --points {args.points}")

    strands = verts.reshape(-1, args.points, 3)
    colors = None
    if hasattr(pc, "colors") and len(pc.colors) == len(verts):
        colors = np.asarray(pc.colors, np.float32).reshape(-1, args.points, 4)[..., :3] / 255.0

    if args.max_strands and len(strands) > args.max_strands:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(len(strands), args.max_strands, replace=False))
        strands = strands[idx]
        if colors is not None:
            colors = colors[idx]

    if colors is None:
        color = np.tile(np.array([0.55, 0.42, 0.30], np.float32), (len(strands), 1))
        color_tip = np.clip(color * 1.05, 0, 1)
    else:
        color = colors[:, 0].astype(np.float32)
        color_tip = colors[:, -1].astype(np.float32)

    bounds = strands.reshape(-1, 3)
    diag = float(np.linalg.norm(bounds.max(0) - bounds.min(0)))
    radius = args.radius if args.radius > 0 else diag * args.radius_frac
    arc = np.linalg.norm(np.diff(strands, axis=1), axis=-1).sum(1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(
        args.out,
        strands=strands.astype(np.float32),
        color=color.astype(np.float32),
        color_tip=color_tip.astype(np.float32),
        radius=np.full(len(strands), radius, np.float32),
        source_ply=os.path.abspath(args.ply),
    )
    print(
        f"saved {args.out} | strands={len(strands)} points={args.points} "
        f"diag={diag:.4f} radius={radius:.6f} arc_med={np.median(arc):.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
