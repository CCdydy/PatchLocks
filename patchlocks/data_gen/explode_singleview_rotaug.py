#!/usr/bin/env python3
"""Explode multi-view dirs into single-view dirs WITH IN-PLANE ROTATION AUGMENTATION (no re-render).

The (image, target) pair can be cheaply multiplied by rotating the patch's tangent frame: rotate the strand
geometry + the camera by an angle phi around the patch normal (RIGID, so the rendered IMAGE is reused as-is),
which yields a NEW consistent pair (rotated strands -> rotated z64 after re-bake; rotated camera -> the shell
back-projection / RGBPROJ comes out rotated to match). z64 is a codec latent (NOT rotation-equivariant), so a
free 2D image-rotate is wrong -- the strands must be physically rotated and z64 RE-BAKED. This script only writes
the rotated geometry (patch.npz) + rotated camera + reused rgb; run bake_latent_batch on the OUTPUT dirs to get
the per-copy z64. Per view it emits the camera-canonical copy (phi=0 vs the view azimuth) plus --naug random
small extra rotations. Roots cropped to the inscribed disk so rotation never leaves the UV square.

Run: python explode_singleview_rotaug.py <mv> <out> --naug 2 --max_deg 35
"""
import os, sys, json, glob, shutil, argparse
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--naug", type=int, default=2, help="extra random-rotation copies per view (0 = aligned-only)")
ap.add_argument("--max_deg", type=float, default=35.0, help="max |extra rotation| in degrees")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.dst, exist_ok=True); rng = np.random.default_rng(a.seed)

def Rz(ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)

dirs = sorted(d for d in glob.glob(a.src + "/*") if os.path.isdir(d) and os.path.exists(f"{d}/cameras.json") and os.path.exists(f"{d}/patch.npz"))
n_out = 0
for d in dirs:
    meta = json.load(open(f"{d}/cameras.json")); views = meta["views"]; size = float(meta["size"])
    p = np.load(f"{d}/patch.npz", allow_pickle=True)
    S0 = p["strands"].astype(np.float32); col = p["color"].astype(np.float32)
    ctip = p["color_tip"].astype(np.float32) if "color_tip" in p.files else col
    rad = p["radius"].astype(np.float32).reshape(-1); base = os.path.basename(d)
    if len(rad) == 1: rad = np.full(len(S0), float(rad[0]), np.float32)
    params = p["params"] if "params" in p.files else None
    disk = np.linalg.norm(S0[:, 0, :2], axis=1) <= (size * 0.5)                  # inscribed-disk crop
    for k, v in enumerate(views):
        if not os.path.exists(f"{d}/rgb_{k}.png"): continue
        MW = np.array(v["matrix_world"], np.float64); loc = MW[:3, 3]
        theta0 = float(np.arctan2(loc[1], loc[0]))                              # camera azimuth (canonical copy = phi 0)
        offsets = [0.0] + list(np.deg2rad(rng.uniform(-a.max_deg, a.max_deg, a.naug)))
        for ai, off in enumerate(offsets):
            theta = theta0 + off; R = Rz(-theta); R4 = np.eye(4); R4[:3, :3] = R
            S = (S0[disk] @ R.T).astype(np.float32)                             # rotate strand geometry around Z
            roots = S[:, 0]; uv = (roots[:, :2] / size + 0.5).astype(np.float32)
            inb = (uv[:, 0] >= 0) & (uv[:, 0] < 1) & (uv[:, 1] >= 0) & (uv[:, 1] < 1)
            S, uvv = S[inb], uv[inb]; cc, ct, rr = col[disk][inb], ctip[disk][inb], rad[disk][inb]
            MW_can = R4 @ MW
            od = os.path.join(a.dst, f"{base}_v{k}_a{ai}"); os.makedirs(od, exist_ok=True)
            np.savez(f"{od}/patch.npz", strands=S, root_uv=uvv, color=cc, color_tip=ct, radius=rr,
                     size=np.float32(size), **({"params": params} if params is not None else {}))
            json.dump({"views": [dict(name=f"v{k}a{ai}", location=MW_can[:3, 3].tolist(),
                                      matrix_world=[list(map(float, row)) for row in MW_can],
                                      lens=v["lens"], sensor_width=v["sensor_width"], res=v["res"])],
                       "size": size}, open(f"{od}/cameras.json", "w"))
            shutil.copy(f"{d}/rgb_{k}.png", f"{od}/rgb_0.png")                  # reuse the render (rigid rotation)
            n_out += 1
print(f"rot-aug exploded {len(dirs)} patches -> {n_out} single-view dirs (x{1 + a.naug}/view) in {a.dst}", flush=True)
print("NEXT: bake_latent_batch + build_field_gt on the OUTPUT dirs (per-copy z64/field), then train.", flush=True)
