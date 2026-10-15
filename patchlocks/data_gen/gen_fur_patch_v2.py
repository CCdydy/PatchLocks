#!/usr/bin/env python3
"""PatchLocks fur generator v2 — bucket + continuous-field driven.

Each patch belongs to a BUCKET (short_sparse, curly_dense, swirl_patchy, ...) that sets coarse
ranges. Within a patch, per-strand parameters are NOT independent random — every strand reads its
length / direction / curl / colour from smooth continuous fields evaluated at its own root_uv, so
variation is spatially coherent (like real fur). Local frame: x=tangent, y=bitangent, z=normal(out);
flat surface z=0; root_uv in [0,1] <-> ((u-.5)*size,(v-.5)*size,0). Output format matches v1
(strands (N,L,3), root_uv, color, radius, size, params) + bucket + field snapshots for QA.
"""
import argparse, json, os
import numpy as np


# ----------------------------- smooth continuous fields -----------------------------
def scalar_field(rng, n_freq=4, max_freq=2.2):
    """Random low-frequency field f:[0,1]^2 -> ~[0,1] (random Fourier features)."""
    F = rng.uniform(-max_freq, max_freq, (n_freq, 2)); ph = rng.uniform(0, 2 * np.pi, n_freq)
    am = rng.uniform(0.4, 1.0, n_freq); s = am.sum()
    def f(uv):
        ang = 2 * np.pi * (uv @ F.T) + ph
        return 0.5 * (1.0 + (np.sin(ang) * am).sum(-1) / s)        # [0,1]
    return f


def _unit(v, eps=1e-9):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


# ----------------------------- buckets -----------------------------
# length (local units, patch size=1), density (#roots), curl_amp, clump_strength, elevation(deg), swirl
BUCKETS = {
    "short_sparse":  dict(length=(0.12, 0.25), density=(150, 300), curl=(0.00, 0.02), clump=(0.0, 0.10), elev=(40, 75), swirl=0),
    "short_dense":   dict(length=(0.12, 0.25), density=(500, 800), curl=(0.00, 0.03), clump=(0.0, 0.20), elev=(40, 75), swirl=0),
    "medium_normal": dict(length=(0.25, 0.55), density=(300, 500), curl=(0.02, 0.05), clump=(0.1, 0.35), elev=(30, 70), swirl=0),
    "medium_dense":  dict(length=(0.25, 0.55), density=(500, 800), curl=(0.02, 0.07), clump=(0.1, 0.35), elev=(30, 70), swirl=0),
    "long_sparse":   dict(length=(0.55, 0.95), density=(150, 300), curl=(0.02, 0.07), clump=(0.1, 0.40), elev=(20, 55), swirl=0),
    "long_dense":    dict(length=(0.55, 0.95), density=(500, 800), curl=(0.03, 0.10), clump=(0.2, 0.60), elev=(20, 55), swirl=0),
    "straight_flat": dict(length=(0.30, 0.70), density=(300, 600), curl=(0.00, 0.02), clump=(0.0, 0.10), elev=(10, 30), swirl=0),
    "curly_dense":   dict(length=(0.30, 0.60), density=(500, 800), curl=(0.07, 0.12), clump=(0.2, 0.50), elev=(30, 70), swirl=0),
    "clumped":       dict(length=(0.30, 0.60), density=(400, 700), curl=(0.02, 0.06), clump=(0.4, 0.70), elev=(30, 65), swirl=0),
    "swirl_patchy":  dict(length=(0.30, 0.60), density=(300, 600), curl=(0.03, 0.08), clump=(0.1, 0.40), elev=(30, 65), swirl=1),
}
BUCKET_NAMES = list(BUCKETS.keys())


def generate_patch(seed=0, bucket="medium_normal", size=1.0):
    rng = np.random.default_rng(seed)
    b = BUCKETS[bucket]
    N_target = int(rng.integers(*b["density"]))
    L = 32

    # ---- continuous fields ----
    f_density = scalar_field(rng); f_length = scalar_field(rng)
    f_az = scalar_field(rng); f_elev = scalar_field(rng); f_curl = scalar_field(rng)
    f_val = scalar_field(rng, n_freq=3)                                # spatial lightness
    base_az = rng.uniform(0, 2 * np.pi)                                # patch grooming heading
    az_spread = rng.uniform(0.3, 1.4)                                  # how much direction varies
    swirl_c = rng.uniform(0.2, 0.8, 2)                                 # vortex centre (uv)
    color_mode = rng.choice(["dark", "light", "mixed"])
    base_col = {"dark": rng.uniform(0.04, 0.25, 3), "light": rng.uniform(0.55, 0.92, 3),
                "mixed": rng.uniform(0.15, 0.8, 3)}[color_mode]

    # ---- roots sampled with rejection on the density field ----
    cand = rng.uniform(0, 1, (N_target * 3, 2)).astype(np.float32)
    dval = f_density(cand)
    keep = rng.uniform(0, 1, len(cand)) < (0.25 + 0.75 * dval)
    uv = cand[keep][:N_target]; N = len(uv)
    roots = np.stack([(uv[:, 0] - 0.5) * size, (uv[:, 1] - 0.5) * size, np.zeros(N)], 1).astype(np.float32)

    # ---- per-strand params from fields ----
    lo, hi = b["length"]; length = (lo + (hi - lo) * f_length(uv)).astype(np.float32)
    e_lo, e_hi = b["elev"]; elev = np.deg2rad(e_lo + (e_hi - e_lo) * f_elev(uv))
    if b["swirl"]:
        d = uv - swirl_c; az = np.arctan2(d[:, 1], d[:, 0]) + np.pi / 2 + (f_az(uv) - 0.5) * 0.6
    else:
        az = base_az + (f_az(uv) - 0.5) * 2 * az_spread
    d0 = _unit(np.stack([np.cos(az) * np.cos(elev), np.sin(az) * np.cos(elev), np.sin(elev)], 1).astype(np.float32))
    c_lo, c_hi = b["curl"]; curl_amp = (c_lo + (c_hi - c_lo) * f_curl(uv)).astype(np.float32)
    curl_freq = float(rng.uniform(2.0, 6.0)); curl_phase = rng.uniform(0, 2 * np.pi, N).astype(np.float32)
    clump_strength = float(rng.uniform(*b["clump"]))
    gravity = float(rng.uniform(0.0, 0.5)); noise_amp = float(rng.uniform(0.0, 0.04))

    # ---- clumps ----
    n_clump = int(rng.integers(3, 14)); cuv = rng.uniform(0.05, 0.95, (n_clump, 2))
    nearest = np.argmin(((uv[:, None] - cuv[None]) ** 2).sum(-1), 1)
    clump_xyz = np.concatenate([(cuv[nearest] - 0.5) * size, np.zeros((N, 1))], 1).astype(np.float32)

    # ---- integrate centreline ----
    seg = (length / (L - 1)).astype(np.float32)
    pts = np.zeros((N, L, 3), np.float32); pts[:, 0] = roots
    dirs = _unit(d0 + rng.normal(0, 0.05, (N, 3)).astype(np.float32))
    gz = np.array([0, 0, -1.0], np.float32)
    for k in range(1, L):
        t = k / (L - 1)
        to_c = _unit(clump_xyz - pts[:, k - 1])
        dirs = _unit(dirs + clump_strength * t * to_c + gravity * t * gz
                     + noise_amp * rng.normal(0, 1, (N, 3)).astype(np.float32))
        pts[:, k] = pts[:, k - 1] + seg[:, None] * dirs
        pts[:, k, 2] = np.maximum(pts[:, k, 2], 0.0)

    # ---- helical curl (per-strand amplitude) ----
    axis = _unit(d0); ref = np.tile(np.array([0, 0, 1.0], np.float32), (N, 1))
    ref[np.abs((axis * ref).sum(1)) > 0.95] = np.array([1, 0, 0.0], np.float32)
    e1 = _unit(np.cross(axis, ref)); e2 = _unit(np.cross(axis, e1))
    tt = np.linspace(0, 1, L)[None]
    ang = 2 * np.pi * curl_freq * tt + curl_phase[:, None]
    amp = curl_amp[:, None] * tt
    offs = amp[..., None] * (np.cos(ang)[..., None] * e1[:, None] + np.sin(ang)[..., None] * e2[:, None])
    pts = pts + offs.astype(np.float32)
    pts[..., 2] = np.maximum(pts[..., 2], 0.0); pts[:, 0] = roots

    # ---- colour from spatial lightness field ----
    val = (0.5 + f_val(uv))[:, None]                                   # ~[0.5,1.5]
    color = (base_col[None] * val + rng.normal(0, 0.04, (N, 3))).clip(0, 1).astype(np.float32)
    radius = np.full(N, float(rng.uniform(0.004, 0.013)), np.float32)

    params = dict(bucket=bucket, seed=int(seed), n_strands=int(N), n_points=L,
                  length_range=b["length"], density_target=N_target, curl_range=b["curl"],
                  clump_strength=clump_strength, elev_range=b["elev"], swirl=b["swirl"],
                  gravity=gravity, color_mode=color_mode, base_color=base_col.tolist())
    # small field snapshots (16x16) for QA
    gg = np.stack(np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 16), indexing="ij"), -1).reshape(-1, 2)
    fields = {n: f(gg).reshape(16, 16).astype(np.float32) for n, f in
              [("density", f_density), ("length", f_length), ("azimuth", f_az), ("elev", f_elev), ("curl", f_curl)]}
    return dict(strands=pts, root_uv=uv.astype(np.float32), color=color, radius=radius,
                size=np.float32(size), params=params, fields=fields)


def save_patch(d, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, strands=d["strands"], root_uv=d["root_uv"], color=d["color"], radius=d["radius"],
             size=d["size"], params=json.dumps(d["params"]),
             **{f"field_{k}": v for k, v in d["fields"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bucket", default="medium_normal", choices=BUCKET_NAMES)
    ap.add_argument("--out", default="patchlocks/datasets/debug/v2_patch.npz")
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    d = generate_patch(a.seed, a.bucket)
    save_patch(d, a.out)
    print(f"bucket={a.bucket} N={len(d['strands'])} L={d['strands'].shape[1]} "
          f"zmax={d['strands'][...,2].max():.3f}")
    print("saved:", a.out)
    if a.preview:
        import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from gen_fur_patch import preview
        preview(d, a.out.replace(".npz", "_preview.png")); print("preview saved")
