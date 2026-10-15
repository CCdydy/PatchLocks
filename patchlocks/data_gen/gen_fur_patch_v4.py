#!/usr/bin/env python3
"""Fur generator v4 — STRUCTURE pass. Field-regression showed per-texel targets were near-pure
stochastic realizations (even ORACLE correspondence got density r_occ~0.15 and latent cos~floor):
the deterministic, appearance-recoverable signal was swamped by per-strand noise. v4 fixes the
DATA so there is a strong deterministic field to predict (and later generate):
  + named multi-scale within-patch STRUCTURE: length gradient, az flow/parting/swirl/crown,
    density tuft/bald, colour gradient/spots/stripes  (all smooth, deterministic, position-keyed)
  - per-strand noise turned DOWN so the field dominates: tighter dir jitter, smaller growth
    random-walk, curl phase made a POSITIONAL field (was random per strand)
Keeps the v1/v2/v3 return contract so the rest of the pipeline is unchanged.
Doublecoat bucket delegates to v3 (already bimodal/structured).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gen_fur_patch_v2 import scalar_field, _unit
from gen_fur_patch_v3 import NATURAL, NAT_NAMES, BUCKETS, _palette, _doublecoat

AZ_MODES = ["flow", "parting", "swirl", "crown"]
DENS_MODES = ["smooth", "tuft", "bald"]
COL_MODES = ["gradient", "spots", "stripes"]


def _az_field(mode, uv, rng, base_az, az_spread, f_az):
    """Return per-strand azimuth (N,) following a deterministic structure + small jitter."""
    if mode == "flow":
        return base_az + (f_az(uv) - 0.5) * 2 * az_spread
    if mode == "parting":
        pc = rng.uniform(0.35, 0.65, 2); th = rng.uniform(0, np.pi)
        n = np.array([-np.sin(th), np.cos(th)])
        side = np.sign((uv - pc) @ n); side[side == 0] = 1
        v = side[:, None] * n                                   # flow away from the parting line
        return np.arctan2(v[:, 1], v[:, 0]) + (f_az(uv) - 0.5) * 0.4
    if mode == "swirl":
        c = rng.uniform(0.3, 0.7, 2); d = uv - c
        return np.arctan2(d[:, 1], d[:, 0]) + np.pi / 2 + (f_az(uv) - 0.5) * 0.4
    # crown / cowlick: radial fan-out
    c = rng.uniform(0.3, 0.7, 2); d = uv - c
    return np.arctan2(d[:, 1], d[:, 0]) + (f_az(uv) - 0.5) * 0.4


def _keep_prob(mode, cand, rng, f_density):
    base = 0.30 + 0.70 * f_density(cand)
    if mode == "tuft":
        nt = int(rng.integers(1, 3)); p = np.zeros(len(cand))
        for _ in range(nt):
            c = rng.uniform(0.2, 0.8, 2); s = rng.uniform(0.10, 0.22)
            p += np.exp(-((cand - c) ** 2).sum(1) / (2 * s * s))
        return np.clip(0.20 + 0.95 * np.clip(p, 0, 1) + 0.3 * f_density(cand), 0, 1)
    if mode == "bald":
        c = rng.uniform(0.3, 0.7, 2); s = rng.uniform(0.10, 0.20)
        hole = np.exp(-((cand - c) ** 2).sum(1) / (2 * s * s))
        return np.clip(base * (1 - 0.92 * hole), 0, 1)
    return base


def _colour(mode, uv, rng, pal_rgb, f_col, f_val):
    n = len(pal_rgb)
    if mode == "spots" and n >= 2:
        ns = int(rng.integers(2, 5)); sel = np.zeros(len(uv))
        for _ in range(ns):
            c = rng.uniform(0.15, 0.85, 2); s = rng.uniform(0.06, 0.14)
            sel = np.maximum(sel, np.exp(-((uv - c) ** 2).sum(1) / (2 * s * s)))
        fr = (sel > 0.5)[:, None]
        base = np.where(fr, pal_rgb[min(1, n - 1)], pal_rgb[0])
    elif mode == "stripes" and n >= 2:
        th = rng.uniform(0, np.pi); dirv = np.array([np.cos(th), np.sin(th)])
        band = 0.5 * (1 + np.sin(2 * np.pi * rng.uniform(3, 6) * (uv @ dirv)))
        idx = (band * (n - 1)); lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, n - 1); fr = (idx - lo)[:, None]
        base = pal_rgb[lo] * (1 - fr) + pal_rgb[hi] * fr
    else:  # gradient (smooth palette mix along a field)
        fc = f_col(uv) * (n - 1); lo = np.floor(fc).astype(int)
        hi = np.minimum(lo + 1, n - 1); fr = (fc - lo)[:, None]
        base = pal_rgb[lo] * (1 - fr) + pal_rgb[hi] * fr
    base = (base * (0.78 + 0.42 * f_val(uv))[:, None] + rng.normal(0, 0.015, base.shape)).clip(0, 1).astype(np.float32)
    return base


def generate_patch(seed=0, bucket="medium_normal", size=1.0):
    b = BUCKETS[bucket]
    if b.get("doublecoat"):
        return _doublecoat(seed, b, size)
    rng = np.random.default_rng(seed)
    N_t = int(rng.integers(*b["density"])); L = 32
    f_density = scalar_field(rng); f_length = scalar_field(rng)
    f_az = scalar_field(rng); f_elev = scalar_field(rng); f_curl = scalar_field(rng)
    f_phase = scalar_field(rng); f_col = scalar_field(rng, n_freq=3); f_val = scalar_field(rng, n_freq=3)

    # ---- pick within-patch STRUCTURE (deterministic, recorded) ----
    az_mode = b["swirl"] and "swirl" or str(rng.choice(AZ_MODES))
    dens_mode = str(rng.choice(DENS_MODES, p=[0.45, 0.35, 0.20]))
    col_mode = str(rng.choice(COL_MODES, p=[0.5, 0.25, 0.25]))
    base_az = rng.uniform(0, 2 * np.pi); az_spread = float(rng.uniform(0.12, 0.40))   # tighter than v3
    grad_dir = _unit(rng.normal(0, 1, 2).astype(np.float64)); grad_mag = float(rng.uniform(0.5, 1.4))

    cand = rng.uniform(0, 1, (N_t * 3, 2)).astype(np.float32)
    keep = rng.uniform(0, 1, len(cand)) < _keep_prob(dens_mode, cand, rng, f_density)
    uv = cand[keep][:N_t]; N = len(uv)
    roots = np.stack([(uv[:, 0] - 0.5) * size, (uv[:, 1] - 0.5) * size, np.zeros(N)], 1).astype(np.float32)

    # length: patch-level base (between-patch) x STRONG spatial multiplier (within-patch 0.5x..1.5x)
    lo, hi = b["length"]; base_len = float(rng.uniform(lo, hi))
    ramp = np.clip(0.5 + grad_mag * ((uv - 0.5) @ grad_dir), 0.0, 1.0)
    length = (base_len * (0.5 + 1.0 * (0.6 * ramp + 0.4 * f_length(uv)))).clip(0.04, None).astype(np.float32)

    e_lo, e_hi = b["elev"]; elev = np.deg2rad(e_lo + (e_hi - e_lo) * f_elev(uv))
    az = _az_field(az_mode, uv, rng, base_az, az_spread, f_az)
    d0 = _unit(np.stack([np.cos(az) * np.cos(elev), np.sin(az) * np.cos(elev), np.sin(elev)], 1).astype(np.float32))

    # curl: patch-level base x strong spatial multiplier (within-patch 0.3x..1.7x)
    c_lo, c_hi = b["curl"]; base_curl = float(rng.uniform(c_lo, c_hi))
    curl_amp = (base_curl * (0.3 + 1.4 * f_curl(uv))).astype(np.float32)
    curl_freq = float(rng.uniform(0.8, 1.8))
    curl_phase = (2 * np.pi * f_phase(uv) + rng.normal(0, 0.25, N)).astype(np.float32)   # POSITIONAL phase
    clump_strength = float(rng.uniform(*b["clump"])); gravity = float(rng.uniform(0.05, 0.45))
    noise_amp = float(rng.uniform(0.003, 0.012))                  # << v3 (was 0.01-0.05): field dominates

    n_clump = int(rng.integers(4, 12)); cuv = rng.uniform(0.05, 0.95, (n_clump, 2))
    near = np.argmin(((uv[:, None] - cuv[None]) ** 2).sum(-1), 1)
    clump_xyz = np.concatenate([(cuv[near] - 0.5) * size, np.zeros((N, 1))], 1).astype(np.float32)

    seg = (length / (L - 1)).astype(np.float32)
    pts = np.zeros((N, L, 3), np.float32); pts[:, 0] = roots
    dirs = _unit(d0 + rng.normal(0, 0.015, (N, 3)).astype(np.float32)); gz = np.array([0, 0, -1.0], np.float32)
    for k in range(1, L):
        t = k / (L - 1); to_c = _unit(clump_xyz - pts[:, k - 1])
        dirs = _unit(dirs + clump_strength * t * to_c + gravity * t * gz
                     + noise_amp * rng.normal(0, 1, (N, 3)).astype(np.float32))
        pts[:, k] = pts[:, k - 1] + seg[:, None] * dirs; pts[:, k, 2] = np.maximum(pts[:, k, 2], 0.0)

    axis = _unit(d0); ref = np.tile(np.array([0, 0, 1.0], np.float32), (N, 1))
    ref[np.abs((axis * ref).sum(1)) > 0.95] = np.array([1, 0, 0.0], np.float32)
    e1 = _unit(np.cross(axis, ref)); e2 = _unit(np.cross(axis, e1)); tt = np.linspace(0, 1, L)[None]
    ang = 2 * np.pi * curl_freq * tt + curl_phase[:, None]; amp = curl_amp[:, None] * tt
    pts = pts + (amp[..., None] * (np.cos(ang)[..., None] * e1[:, None] + np.sin(ang)[..., None] * e2[:, None])).astype(np.float32)
    pts[..., 2] = np.maximum(pts[..., 2], 0.0); pts[:, 0] = roots

    mode = rng.choice(["solid", "two_tone", "two_tone", "tabby"]); pal = _palette(rng, mode)
    pal_rgb = np.array([NATURAL[c] for c in pal], np.float32)
    base = _colour(col_mode, uv, rng, pal_rgb, f_col, f_val)
    tip = np.clip(base * 1.08 + 0.02, 0, 1).astype(np.float32)
    dense = b["density"][1] > 1500
    radius = np.full(N, float(rng.uniform(0.0012, 0.0028) if dense else rng.uniform(0.0020, 0.0035)), np.float32)

    params = dict(bucket=bucket, seed=int(seed), n_strands=int(N), color_mode=mode, palette=pal, version=4,
                  structure=dict(az=az_mode, density=dens_mode, colour=col_mode,
                                 grad_dir=[float(grad_dir[0]), float(grad_dir[1])], grad_mag=grad_mag),
                  length_range=b["length"], curl_range=b["curl"], clump_strength=clump_strength,
                  elev_range=b["elev"], swirl=b["swirl"], radius=float(radius[0]))
    return dict(strands=pts, root_uv=uv.astype(np.float32), color=base, color_tip=tip,
                radius=radius, size=np.float32(size), params=params)


def save_patch(d, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    extra = {"color_tip": d["color_tip"]} if "color_tip" in d else {}
    np.savez(out_path, strands=d["strands"], root_uv=d["root_uv"], color=d["color"], radius=d["radius"],
             size=d["size"], params=json.dumps(d["params"]), **extra)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bucket", default="medium_dense")
    ap.add_argument("--out", default="patchlocks/datasets/debug/v4_patch.npz")
    a = ap.parse_args(); d = generate_patch(a.seed, a.bucket); save_patch(d, a.out)
    s = d["params"]["structure"]
    print(f"v4 bucket={a.bucket} N={len(d['strands'])} struct(az={s['az']},dens={s['density']},col={s['colour']}) "
          f"grad_mag={s['grad_mag']:.2f}")
    print("saved:", a.out)
