#!/usr/bin/env python3
"""Give grown strands a natural shape: a gentle curve back towards the skin, sideways sway and
length variation (body and leg fur reads as straw when every strand is a straight stick).

The grow stamps nearly straight strands (chord/arc ~0.99). Real guard hairs leave the skin at an
angle and curve over towards it along their length, each a little differently. Per strand:
  * --droop: the tip is pulled towards the skin by this fraction of the length (quadratic along
    the strand, so the root angle is kept), drawn per strand from [0.5, 1] x droop;
  * --sway: a sideways quadratic offset (tangent, across the chord), std as a fraction of length;
  * --length_jitter: length scale drawn from 1 +- this.
  * --radius_scale: thinner strands (the grow's radius covers the skin; thinner reads softer).
Arc length is preserved (before the jitter), and points never go below the root's tangent plane.

  python patchlocks/deploy/naturalize_strands.py --groom <run>/grow/surfpack_v2fur.npz \\
      --region_prior <run>/v2_region_prior.annotated.npz --regions body,legs \\
      --out <run>/grow/surfpack_v2fur_nat.npz
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bake_mesh_albedo_robust import load_obj_mesh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groom", required=True)
    ap.add_argument("--region_prior", required=True)
    ap.add_argument("--mesh", default="", help="furless mesh (default: the groom's mesh_path)")
    ap.add_argument("--regions", default="body,legs", help="comma-separated regions; 'all' for every strand")
    ap.add_argument("--droop", type=float, default=0.35)
    ap.add_argument("--sway", type=float, default=0.12)
    ap.add_argument("--length_jitter", type=float, default=0.15)
    ap.add_argument("--radius_scale", type=float, default=1.0, help="multiplies the radius (thinner strands read softer)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=400_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    g = dict(np.load(a.groom, allow_pickle=True))
    V, F, _ = load_obj_mesh(a.mesh or str(g["mesh_path"]))
    V = V.astype(np.float64)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-15)
    if a.regions == "all":
        idx = np.arange(len(g["strands"]))
    else:
        reg = np.load(a.region_prior, allow_pickle=True)["patch_region"].astype(str)
        row = g["root_prior_row"] if "root_prior_row" in g else g["root_patch"]
        idx = np.flatnonzero(np.isin(reg[np.clip(row, 0, len(reg) - 1)], a.regions.split(",")))
    S = g["strands"]
    rng = np.random.default_rng(a.seed)
    t = np.linspace(0, 1, S.shape[1])[None, :, None]
    straight = []
    for c0 in range(0, len(idx), a.chunk):
        ii = idx[c0:c0 + a.chunk]
        P = S[ii].astype(np.float64)
        off = P - P[:, :1]
        n = fn[g["root_face"][ii]]
        arc0 = np.linalg.norm(np.diff(off, axis=1), axis=2).sum(1)
        ch = off[:, -1]
        cu = ch / np.maximum(np.linalg.norm(ch, axis=1, keepdims=True), 1e-12)
        side = np.cross(n, cu)
        side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-12)
        k = len(ii)
        dr = a.droop * rng.uniform(0.5, 1.0, k) * arc0
        sw = a.sway * rng.standard_normal(k) * arc0
        off = off - (dr[:, None, None] * t ** 2) * n[:, None] + (sw[:, None, None] * t ** 2) * side[:, None]
        h = (off * n[:, None]).sum(-1, keepdims=True)                      # stay above the skin plane
        off -= np.minimum(h, 0.0) * n[:, None]
        arc1 = np.linalg.norm(np.diff(off, axis=1), axis=2).sum(1)
        ls = 1.0 + a.length_jitter * rng.uniform(-1, 1, k)
        off *= (ls * arc0 / np.maximum(arc1, 1e-12))[:, None, None]
        S[ii] = (P[:, :1] + off).astype(S.dtype)
        straight.append(np.linalg.norm(off[:, -1], axis=1) / np.maximum(ls * arc0, 1e-12))
    g["strands"] = S
    if a.radius_scale != 1.0:
        g["radius"] = g["radius"].astype(np.float32).copy()
        g["radius"][idx] *= a.radius_scale
    g["naturalize"] = np.asarray(f"regions={a.regions} droop={a.droop} sway={a.sway} length_jitter={a.length_jitter} radius_scale={a.radius_scale}")
    np.savez(a.out, **g)
    st = np.concatenate(straight) if straight else np.zeros(1)
    print(json.dumps({"strands": int(len(idx)), "chord_over_arc_p10_50_90": np.round(np.percentile(st, [10, 50, 90]), 3).tolist()}))


if __name__ == "__main__":
    main()
