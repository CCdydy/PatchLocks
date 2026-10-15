#!/usr/bin/env python3
"""Fur generator v3 — realism pass (per user feedback): thinner + many more strands + far less
spiral + natural pet-fur colour palette + root->tip colour gradient. Geometry/field machinery is
inherited from v2; only the visual-domain knobs change. Output adds `color_tip` (N,3) for the
renderer's root->tip gradient; everything else matches v1/v2 so the rest of the pipeline is unchanged.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gen_fur_patch_v2 import scalar_field, _unit

# natural cat/dog fur base colours (linear-ish RGB)
NATURAL = {
    "black": (0.02, 0.02, 0.02), "dark_brown": (0.09, 0.055, 0.035), "brown": (0.17, 0.10, 0.055),
    "tan": (0.42, 0.28, 0.15), "cream": (0.78, 0.66, 0.48), "white": (0.86, 0.84, 0.80),
    "gray": (0.33, 0.32, 0.31), "orange_tabby": (0.55, 0.27, 0.09),
    "off_white": (0.82, 0.80, 0.75), "pale_gray": (0.56, 0.56, 0.55),
}
NAT_NAMES = list(NATURAL)

# v3 buckets: density x~3 (more strands), curl amplitude ~halved (less coil)
BUCKETS = {
    "short_sparse":  dict(length=(0.12, 0.25), density=(450, 900),   curl=(0.00, 0.010), clump=(0.0, 0.10), elev=(40, 75), swirl=0),
    "short_dense":   dict(length=(0.12, 0.25), density=(1600, 2600), curl=(0.00, 0.015), clump=(0.0, 0.20), elev=(40, 75), swirl=0),
    "medium_normal": dict(length=(0.25, 0.55), density=(1000, 1800), curl=(0.01, 0.025), clump=(0.1, 0.35), elev=(30, 70), swirl=0),
    "medium_dense":  dict(length=(0.25, 0.55), density=(1800, 2800), curl=(0.01, 0.035), clump=(0.1, 0.35), elev=(30, 70), swirl=0),
    "long_sparse":   dict(length=(0.55, 0.95), density=(450, 900),   curl=(0.01, 0.035), clump=(0.1, 0.40), elev=(20, 55), swirl=0),
    "long_dense":    dict(length=(0.55, 0.95), density=(1600, 2600), curl=(0.015, 0.05), clump=(0.2, 0.55), elev=(20, 55), swirl=0),
    "straight_flat": dict(length=(0.30, 0.70), density=(1000, 2000), curl=(0.00, 0.010), clump=(0.0, 0.10), elev=(10, 30), swirl=0),
    "wavy_dense":    dict(length=(0.30, 0.60), density=(1800, 2800), curl=(0.028, 0.048), clump=(0.2, 0.45), elev=(30, 65), swirl=0),
    "clumped":       dict(length=(0.30, 0.60), density=(1400, 2400), curl=(0.01, 0.030), clump=(0.4, 0.70), elev=(30, 65), swirl=0),
    "swirl_patchy":  dict(length=(0.30, 0.60), density=(900, 1800),  curl=(0.015, 0.040), clump=(0.1, 0.40), elev=(30, 65), swirl=1),
    # short-haired DOUBLE COAT (shiba/corgi/lab/short-hair cat): dense undercoat + aligned stiff guard hairs
    "shiba_short_doublecoat": dict(doublecoat=True, density=(2500, 5000), uc_ratio=(0.70, 0.85),
                                   uc_len=(0.06, 0.16), guard_len=(0.18, 0.35), curl=(0.0, 0.008),
                                   elev=(20, 45), az_spread=(0.10, 0.30), clump=(0.0, 0.12),
                                   uc_rad=(0.0008, 0.0018), guard_rad=(0.0015, 0.0030)),
}
BUCKET_NAMES = list(BUCKETS)


def _doublecoat(seed, b, size):
    """Bimodal short-hair double coat: many soft short undercoat + fewer stiffer aligned guard hairs."""
    rng = np.random.default_rng(seed); L = 32
    N_t = int(rng.integers(*b["density"]))
    f_density, f_az, f_elev, f_val = scalar_field(rng), scalar_field(rng), scalar_field(rng), scalar_field(rng, n_freq=3)
    base_az = rng.uniform(0, 2 * np.pi); az_spread = float(rng.uniform(*b["az_spread"]))
    cand = rng.uniform(0, 1, (N_t * 3, 2)).astype(np.float32)
    keep = rng.uniform(0, 1, len(cand)) < (0.35 + 0.65 * f_density(cand))
    uv = cand[keep][:N_t]; N = len(uv)
    roots = np.stack([(uv[:, 0] - 0.5) * size, (uv[:, 1] - 0.5) * size, np.zeros(N)], 1).astype(np.float32)

    is_uc = rng.uniform(0, 1, N) < float(rng.uniform(*b["uc_ratio"]))    # undercoat vs guard
    length = np.where(is_uc, rng.uniform(*b["uc_len"], N), rng.uniform(*b["guard_len"], N)).astype(np.float32)
    length *= (0.85 + 0.3 * f_val(uv))                                  # mild spatial modulation
    radius = np.where(is_uc, rng.uniform(*b["uc_rad"], N), rng.uniform(*b["guard_rad"], N)).astype(np.float32)
    e_lo, e_hi = b["elev"]; elev = np.deg2rad(e_lo + (e_hi - e_lo) * f_elev(uv))
    az = base_az + (f_az(uv) - 0.5) * 2 * az_spread                     # aligned, small spread
    d0 = _unit(np.stack([np.cos(az) * np.cos(elev), np.sin(az) * np.cos(elev), np.sin(elev)], 1).astype(np.float32))
    d0 = _unit(d0 + rng.normal(0, 0.04, (N, 3)).astype(np.float32))
    curl_amp = np.where(is_uc, rng.uniform(*b["curl"], N), rng.uniform(0.0, 0.003, N)).astype(np.float32)  # guard stiffer/straighter
    curl_freq = float(rng.uniform(0.6, 1.4)); curl_phase = rng.uniform(0, 2 * np.pi, N).astype(np.float32)
    clump_strength = float(rng.uniform(*b["clump"])); gravity = float(rng.uniform(0.05, 0.30)); noise_amp = float(rng.uniform(0.005, 0.02))
    n_clump = int(rng.integers(3, 9)); cuv = rng.uniform(0.05, 0.95, (n_clump, 2))
    near = np.argmin(((uv[:, None] - cuv[None]) ** 2).sum(-1), 1)
    clump_xyz = np.concatenate([(cuv[near] - 0.5) * size, np.zeros((N, 1))], 1).astype(np.float32)

    seg = (length / (L - 1)).astype(np.float32); pts = np.zeros((N, L, 3), np.float32); pts[:, 0] = roots
    dirs = d0.copy(); gz = np.array([0, 0, -1.0], np.float32)
    for k in range(1, L):
        t = k / (L - 1); to_c = _unit(clump_xyz - pts[:, k - 1])
        dirs = _unit(dirs + clump_strength * t * to_c + gravity * t * gz + noise_amp * rng.normal(0, 1, (N, 3)).astype(np.float32))
        pts[:, k] = pts[:, k - 1] + seg[:, None] * dirs; pts[:, k, 2] = np.maximum(pts[:, k, 2], 0.0)
    axis = _unit(d0); ref = np.tile(np.array([0, 0, 1.0], np.float32), (N, 1))
    ref[np.abs((axis * ref).sum(1)) > 0.95] = np.array([1, 0, 0.0], np.float32)
    e1 = _unit(np.cross(axis, ref)); e2 = _unit(np.cross(axis, e1)); tt = np.linspace(0, 1, L)[None]
    ang = 2 * np.pi * curl_freq * tt + curl_phase[:, None]; amp = curl_amp[:, None] * tt
    pts = pts + (amp[..., None] * (np.cos(ang)[..., None] * e1[:, None] + np.sin(ang)[..., None] * e2[:, None])).astype(np.float32)
    pts[..., 2] = np.maximum(pts[..., 2], 0.0); pts[:, 0] = roots

    # colour: undercoat = pale (cream/off_white/pale_gray); guard = tan/orange/brown root -> darker tip (tipping)
    uc_pal = np.array([NATURAL[c] for c in rng.choice(["cream", "off_white", "pale_gray"], 2, replace=False)], np.float32)
    gd_pal = np.array([NATURAL[c] for c in rng.choice(["tan", "orange_tabby", "brown"], 2, replace=False)], np.float32)
    fc = f_val(uv); uc = uc_pal[0] * (1 - fc[:, None]) + uc_pal[1] * fc[:, None]
    gd = gd_pal[0] * (1 - fc[:, None]) + gd_pal[1] * fc[:, None]
    base = np.where(is_uc[:, None], uc, gd)
    base = (base * (0.85 + 0.3 * f_val(uv))[:, None] + rng.normal(0, 0.02, (N, 3))).clip(0, 1).astype(np.float32)
    shiba_type = str(rng.choice(["red", "sesame"], p=[0.65, 0.35]))      # sesame = stronger dark tipping
    gtip = 0.60 if shiba_type == "red" else 0.42                          # guard tip darkness (medium vs strong)
    tip = np.where(is_uc[:, None], np.clip(base * 1.05 + 0.02, 0, 1), (base * gtip).clip(0, 1)).astype(np.float32)
    params = dict(bucket="shiba_short_doublecoat", seed=int(seed), n_strands=int(N), version=3,
                  doublecoat=True, shiba_type=shiba_type, uc_frac=float(is_uc.mean()), radius=float(np.median(radius)))
    return dict(strands=pts, root_uv=uv.astype(np.float32), color=base, color_tip=tip,
                radius=radius, size=np.float32(size), params=params)


def _palette(rng, mode):
    if mode == "solid":
        return [rng.choice(NAT_NAMES)]
    if mode == "two_tone":
        return list(rng.choice(NAT_NAMES, 2, replace=False))
    return list(rng.choice(NAT_NAMES, 3, replace=False))           # tabby/mixed


def generate_patch(seed=0, bucket="medium_normal", size=1.0):
    b = BUCKETS[bucket]
    if b.get("doublecoat"):
        return _doublecoat(seed, b, size)
    rng = np.random.default_rng(seed)
    N_t = int(rng.integers(*b["density"])); L = 32
    f_density, f_length = scalar_field(rng), scalar_field(rng)
    f_az, f_elev, f_curl = scalar_field(rng), scalar_field(rng), scalar_field(rng)
    f_col = scalar_field(rng, n_freq=3)                            # which palette colour, spatially
    f_val = scalar_field(rng, n_freq=3)                           # local lightness
    base_az = rng.uniform(0, 2 * np.pi); az_spread = rng.uniform(0.25, 1.1)
    swirl_c = rng.uniform(0.2, 0.8, 2)
    mode = rng.choice(["solid", "solid", "two_tone", "tabby"]); pal = _palette(rng, mode)
    pal_rgb = np.array([NATURAL[c] for c in pal], np.float32)

    cand = rng.uniform(0, 1, (N_t * 3, 2)).astype(np.float32)
    keep = rng.uniform(0, 1, len(cand)) < (0.3 + 0.7 * f_density(cand))
    uv = cand[keep][:N_t]; N = len(uv)
    roots = np.stack([(uv[:, 0] - 0.5) * size, (uv[:, 1] - 0.5) * size, np.zeros(N)], 1).astype(np.float32)

    lo, hi = b["length"]; length = (lo + (hi - lo) * f_length(uv)).astype(np.float32)
    e_lo, e_hi = b["elev"]; elev = np.deg2rad(e_lo + (e_hi - e_lo) * f_elev(uv))
    if b["swirl"]:
        d = uv - swirl_c; az = np.arctan2(d[:, 1], d[:, 0]) + np.pi / 2 + (f_az(uv) - 0.5) * 0.5
    else:
        az = base_az + (f_az(uv) - 0.5) * 2 * az_spread
    d0 = _unit(np.stack([np.cos(az) * np.cos(elev), np.sin(az) * np.cos(elev), np.sin(elev)], 1).astype(np.float32))
    c_lo, c_hi = b["curl"]; curl_amp = (c_lo + (c_hi - c_lo) * f_curl(uv)).astype(np.float32)
    curl_freq = float(rng.uniform(0.8, 2.0))                       # fewer turns -> waves not coils
    curl_phase = rng.uniform(0, 2 * np.pi, N).astype(np.float32)
    clump_strength = float(rng.uniform(*b["clump"])); gravity = float(rng.uniform(0.05, 0.55))
    noise_amp = float(rng.uniform(0.01, 0.05))                     # micro-kink instead of regular helix

    n_clump = int(rng.integers(4, 16)); cuv = rng.uniform(0.05, 0.95, (n_clump, 2))
    near = np.argmin(((uv[:, None] - cuv[None]) ** 2).sum(-1), 1)
    clump_xyz = np.concatenate([(cuv[near] - 0.5) * size, np.zeros((N, 1))], 1).astype(np.float32)

    seg = (length / (L - 1)).astype(np.float32)
    pts = np.zeros((N, L, 3), np.float32); pts[:, 0] = roots
    dirs = _unit(d0 + rng.normal(0, 0.06, (N, 3)).astype(np.float32)); gz = np.array([0, 0, -1.0], np.float32)
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

    # colour: blend CONTINUOUSLY between palette colours along a spatial field (no hard blocks)
    fc = f_col(uv) * (len(pal) - 1); lo = np.floor(fc).astype(int)
    hi = np.minimum(lo + 1, len(pal) - 1); fr = (fc - lo)[:, None]
    base = pal_rgb[lo] * (1 - fr) + pal_rgb[hi] * fr               # (N,3) smooth palette mix
    base = (base * (0.78 + 0.42 * f_val(uv))[:, None] + rng.normal(0, 0.02, (N, 3))).clip(0, 1).astype(np.float32)
    tip = np.clip(base * 1.08 + 0.02, 0, 1).astype(np.float32)     # gentle root->tip lightening (avoid over-expose)
    dense = b["density"][1] > 1500
    radius = np.full(N, float(rng.uniform(0.0012, 0.0028) if dense else rng.uniform(0.0020, 0.0035)), np.float32)

    params = dict(bucket=bucket, seed=int(seed), n_strands=int(N), color_mode=mode, palette=pal,
                  length_range=b["length"], curl_range=b["curl"], clump_strength=clump_strength,
                  elev_range=b["elev"], swirl=b["swirl"], radius=float(radius[0]), version=3)
    from fur_fields import v3_fields
    _fc = dict(f_density=f_density, f_length=f_length, f_az=f_az, f_elev=f_elev, f_curl=f_curl, f_col=f_col, f_val=f_val)
    fields = {r: v3_fields(r, b, _fc, base_az=base_az, az_spread=az_spread, swirl_c=swirl_c,
                           pal_rgb=pal_rgb, radius_val=float(radius[0])) for r in (16, 32)}
    return dict(strands=pts, root_uv=uv.astype(np.float32), color=base, color_tip=tip,
                radius=radius, size=np.float32(size), params=params, fields=fields)


def save_patch(d, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    extra = {"color_tip": d["color_tip"]} if "color_tip" in d else {}
    np.savez(out_path, strands=d["strands"], root_uv=d["root_uv"], color=d["color"], radius=d["radius"],
             size=d["size"], params=json.dumps(d["params"]), **extra)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bucket", default="medium_dense")
    ap.add_argument("--out", default="patchlocks/datasets/debug/v3_patch.npz")
    a = ap.parse_args(); d = generate_patch(a.seed, a.bucket); save_patch(d, a.out)
    pp = d['params']
    print(f"v3 bucket={a.bucket} N={len(d['strands'])} radius~{np.median(d['radius']):.4f} "
          f"{'doublecoat uc_frac=%.2f' % pp['uc_frac'] if pp.get('doublecoat') else 'palette=%s' % pp.get('palette')}")
    print("saved:", a.out)
