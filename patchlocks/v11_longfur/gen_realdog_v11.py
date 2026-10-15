#!/usr/bin/env python3
"""Realistic dog-fur GEOMETRY target generator (v8). Multiple coat archetypes so the dataset spans real dog
variety (short-dense, curly, wavy), with a dog color palette weighted toward solids.

Motivation: the v7 pilot proved photoreal *input* closes much of the conditioning OOD, but the prediction
still does not read as real dog fur — the remaining gap is the GEOMETRY TARGET (density + lie statistics).
This generator targets those directly, calibrated to NeuralFur real captured fur (result_strands_scaled_back.ply,
50k strands x 100 pts): lie elevation median 7.4 deg (short coats lie nearly flat/贴伏), near-straight
(curl med 0.05) for short coats, coherent flow (顺流). Density set HIGH (a real coat is a dense continuous
mat; NeuralFur's 50k are sparse guide strands). Curly/wavy coats stand more + curl, per breed reality.

Output contract identical to v5 (strands, root_uv, color, color_tip, radius, size, params) so bake/field/
render/train are unchanged.
Run: python patchlocks/data_gen/gen_realdog_patch.py --seed 0 --out /tmp/rd.npz [--coat curly]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_gen"))
import numpy as np
from gen_fur_patch_v2 import scalar_field, _unit
from gen_fur_patch_v4 import _keep_prob, DENS_MODES
from gen_fur_patch_v5 import _grow, _color_scheme, NATURAL

DOG_SCHEMES = ["solid", "two_tone", "bicolor", "tabby", "tipped"]   # no tortie (cat-specific calico)
DOG_SCHEME_P = [0.56, 0.14, 0.12, 0.08, 0.10]                       # 单色 dogs emphasized

# coat archetypes. elev_deg = elev_floor + (elev_max-elev_floor) * u^elev_skew  (u = smooth field in [0,1]);
# higher skew => more mass at low angle (flatter/贴伏). short coats lie flat & near-straight; curly/wavy/golden
# stand & curl/arc more. density HIGH (continuous mat) + FINE radius => reads as fur not coarse tufts. each
# coat carries a breed-realistic preferred solid palette ("solid"). long_golden uses high gravity so its long
# strands arc/drape (金毛 弧度). ultra_short = dachshund/min-pin sleek black very-short.
COATS = {
    "ultra_short":           dict(density=(18000, 38000), length=(0.025, 0.060), elev_floor=1.0, elev_max=16.0, elev_skew=2.6,
                        curl=(0.0, 0.004), curl_freq=(0.3, 0.6), clump=(0.0, 0.05), gravity=(0.02, 0.05),
                        noise=(0.002, 0.005), az_spread=(0.05, 0.16), radius=(0.0006, 0.0012),
                        solid=["black", "black", "black", "dark_brown", "charcoal", "chocolate", "tan", "brown", "ginger"]),  # dachshund/min-pin/boxer — sleek ultra-short
    "short_dense":           dict(density=(14000, 30000), length=(0.05, 0.13), elev_floor=1.0, elev_max=30.0, elev_skew=2.0,
                        curl=(0.0, 0.010), curl_freq=(0.4, 0.9), clump=(0.0, 0.12), gravity=(0.02, 0.07),
                        noise=(0.003, 0.007), az_spread=(0.06, 0.20), radius=(0.0007, 0.0016),
                        solid=["black", "cream", "chocolate", "tan", "fawn", "brown", "off_white", "dark_brown"]),   # lab/beagle/pug — short flat dense
    "short_double_standoff": dict(density=(14000, 28000), length=(0.08, 0.20), elev_floor=28.0, elev_max=80.0, elev_skew=0.8,
                        curl=(0.0, 0.008), curl_freq=(0.3, 0.7), clump=(0.0, 0.10), gravity=(0.03, 0.10),
                        noise=(0.004, 0.008), az_spread=(0.10, 0.30), radius=(0.0009, 0.0020),
                        solid=["gray", "black", "charcoal", "ginger", "cream", "tan", "off_white", "blue_gray"]),    # husky/shiba/corgi/pom — STANDING double coat
    "wiry":                  dict(density=(7000, 14000), length=(0.08, 0.20), elev_floor=25.0, elev_max=60.0, elev_skew=1.0,
                        curl=(0.010, 0.028), curl_freq=(1.5, 2.6), clump=(0.05, 0.20), gravity=(0.06, 0.18),
                        noise=(0.008, 0.014), az_spread=(0.15, 0.40), radius=(0.0018, 0.0034),
                        solid=["charcoal", "gray", "silver", "black", "brown", "dark_brown", "fawn", "tan"]),        # schnauzer/terrier — wiry crinkle coarse
    "wavy":                  dict(density=(8000, 16000), length=(0.14, 0.30), elev_floor=3.0, elev_max=45.0, elev_skew=1.3,
                        curl=(0.012, 0.030), curl_freq=(0.6, 1.1), clump=(0.10, 0.30), gravity=(0.10, 0.30),
                        noise=(0.005, 0.010), az_spread=(0.10, 0.30), radius=(0.0010, 0.0020),
                        solid=["brown", "dark_brown", "black", "ginger", "tan", "chocolate", "cream"]),              # cocker/spaniel — wavy medium
    "curly":                 dict(density=(9000, 18000), length=(0.12, 0.26), elev_floor=8.0, elev_max=65.0, elev_skew=1.0,
                        curl=(0.030, 0.070), curl_freq=(1.2, 2.2), clump=(0.20, 0.50), gravity=(0.03, 0.10),
                        noise=(0.005, 0.010), az_spread=(0.15, 0.45), radius=(0.0009, 0.0018),
                        solid=["white", "off_white", "black", "ginger", "brown", "cream", "silver", "fawn"]),        # poodle/bichon/doodle — tight curls
    "long_golden":           dict(density=(6000, 13000), length=(0.30, 0.55), elev_floor=5.0, elev_max=42.0, elev_skew=1.4,
                        curl=(0.008, 0.022), curl_freq=(0.4, 0.9), clump=(0.10, 0.35), gravity=(0.28, 0.52),
                        noise=(0.005, 0.010), az_spread=(0.10, 0.30), radius=(0.0013, 0.0026),
                        solid=["ginger", "fawn", "tan", "cream", "ginger", "fawn", "tan", "off_white"]),             # golden/setter — long arced golden + feathering
    "long_silky":            dict(density=(5000, 11000), length=(0.40, 0.85), elev_floor=2.0, elev_max=22.0, elev_skew=1.8,
                        curl=(0.0, 0.012), curl_freq=(0.4, 0.8), clump=(0.10, 0.35), gravity=(0.45, 0.72),
                        noise=(0.004, 0.009), az_spread=(0.08, 0.25), radius=(0.0010, 0.0020),
                        solid=["white", "cream", "off_white", "silver", "gray", "black", "tan", "blue_gray"]),       # afghan/maltese/yorkie — long straight draping silky
    "fluffy_double":         dict(density=(16000, 32000), length=(0.18, 0.40), elev_floor=35.0, elev_max=78.0, elev_skew=0.9,
                        curl=(0.005, 0.020), curl_freq=(0.5, 1.0), clump=(0.10, 0.30), gravity=(0.08, 0.22),
                        noise=(0.006, 0.012), az_spread=(0.15, 0.40), radius=(0.0010, 0.0022),
                        solid=["cream", "white", "off_white", "ginger", "black", "gray", "fawn", "charcoal"]),       # chow/samoyed/keeshond — dense standing fluffy
}
COAT_NAMES = ["ultra_short", "short_dense", "short_double_standoff", "wiry", "wavy", "curly", "long_golden", "long_silky", "fluffy_double"]
# V11: emphasize LONG / CURLY / FLUFFY (the OOD coats V10 averaged into felt). short coats kept (0.24) so
# the model still covers its old domain. wavy+curly+long_golden+long_silky+fluffy = 0.76.
COAT_P =     [0.05,          0.07,          0.06,                    0.06,   0.13,   0.16,    0.17,          0.13,         0.17]


def generate_realdog_patch(seed=0, size=1.0, color_scheme=None, coat=None, density_scale=1.0):
    rng = np.random.default_rng(seed); L = 32
    coat = coat or str(rng.choice(COAT_NAMES, p=COAT_P)); RD = COATS[coat]
    f_density = scalar_field(rng); f_length = scalar_field(rng); f_az = scalar_field(rng); f_az2 = scalar_field(rng)
    f_elev = scalar_field(rng); f_curl = scalar_field(rng); f_phase = scalar_field(rng)
    f_col = scalar_field(rng, n_freq=3); f_val = scalar_field(rng, n_freq=3)

    # ---- dense roots (uniform-ish with mild density field) ----
    dens_mode = str(rng.choice(DENS_MODES, p=[0.7, 0.2, 0.1]))
    N_t = int(rng.integers(*RD["density"]) * density_scale)
    cand = rng.uniform(0, 1, (N_t * 3, 2)).astype(np.float32)
    keep = rng.uniform(0, 1, len(cand)) < _keep_prob(dens_mode, cand, rng, f_density)
    uv = cand[keep][:N_t]; N = len(uv)
    roots = np.stack([(uv[:, 0] - 0.5) * size, (uv[:, 1] - 0.5) * size, np.zeros(N)], 1).astype(np.float32)

    # ---- length (mild spatial ramp) ----
    lo, hi = RD["length"]; base_len = float(rng.uniform(lo, hi))
    length = (base_len * (0.7 + 0.6 * f_length(uv))).clip(0.03, None).astype(np.float32)

    # ---- elevation (lie): low-skew for short coats (贴伏), more upright for curly/wavy ----
    u = f_elev(uv)
    elev_deg = RD["elev_floor"] + (RD["elev_max"] - RD["elev_floor"]) * (u ** RD["elev_skew"])
    elev = np.deg2rad(elev_deg.astype(np.float32))

    # ---- COHERENT flow azimuth (顺流): base dir + small smooth spread + 2nd field + jitter ----
    base_az = float(rng.uniform(0, 2 * np.pi)); az_spread = float(rng.uniform(*RD["az_spread"]))
    az = (base_az + az_spread * (2 * f_az(uv) - 1) + 0.35 * az_spread * (2 * f_az2(uv) - 1)
          + rng.normal(0, 0.07, N)).astype(np.float32)
    d0 = _unit(np.stack([np.cos(az) * np.cos(elev), np.sin(az) * np.cos(elev), np.sin(elev)], 1).astype(np.float32))

    # ---- curl + fine radius ----
    c_lo, c_hi = RD["curl"]; base_curl = float(rng.uniform(c_lo, c_hi))
    curl_amp = (base_curl * (0.3 + 1.4 * f_curl(uv))).astype(np.float32)
    rlo, rhi = RD["radius"]; radius = np.full(N, float(rng.uniform(rlo, rhi)), np.float32)
    clump_strength = float(rng.uniform(*RD["clump"]))
    gravity = float(rng.uniform(*RD["gravity"])); noise_amp = float(rng.uniform(*RD["noise"]))
    curl_freq = float(rng.uniform(*RD["curl_freq"])); curl_phase = (2 * np.pi * f_phase(uv) + rng.normal(0, 0.25, N)).astype(np.float32)

    pts = _grow(roots, d0, length, clump_strength, gravity, noise_amp, curl_amp, curl_freq, curl_phase, rng,
                L=L, n_clump=(2, 6), size=size, uv=uv)

    # ---- dog color (palette weighted to solids) ----
    color_scheme = color_scheme or str(rng.choice(DOG_SCHEMES, p=DOG_SCHEME_P))
    import gen_fur_patch_v5 as v5
    _save = v5.SOLID_COLORS; v5.SOLID_COLORS = RD["solid"]      # breed-realistic per-coat palette
    try:
        base = _color_scheme(color_scheme, uv, rng, f_col, f_val)
    finally:
        v5.SOLID_COLORS = _save
    if color_scheme == "tipped":
        dark = np.array(NATURAL[str(rng.choice(["dark_brown", "black", "charcoal", "brown"]))], np.float32)
        tipf = float(rng.uniform(0.40, 0.70)); tip = (base * (1 - tipf) + dark[None] * tipf).clip(0, 1).astype(np.float32)
    else:
        tip = np.clip(base * float(rng.uniform(1.02, 1.10)) + 0.015, 0, 1).astype(np.float32)
    # per-strand luminance jitter: adjacent strands catch light differently -> reads as fur, not a flat sheet
    vj = rng.normal(1.0, 0.14, (N, 1)).astype(np.float32).clip(0.6, 1.4)
    base = (base * vj).clip(0, 1).astype(np.float32); tip = (tip * vj).clip(0, 1).astype(np.float32)

    rough = float(rng.uniform(0.45, 0.70)); material = dict(finish="normal", roughness=rough,
            specular=float(rng.uniform(0.10, 0.30)), transmission=float(rng.uniform(0.03, 0.12)), coat=float(rng.uniform(0.0, 0.08)))
    params = dict(version=8, archetype=f"realdog_{coat}", coat=coat, seed=int(seed), n_strands=int(N),
                  color_scheme=color_scheme, finish="normal", material=material,
                  structure=dict(az="flow_coherent", density=dens_mode),
                  length_med=float(np.median(length)), curl_med=float(np.median(curl_amp)),
                  elev_med_deg=float(np.median(elev_deg)), radius=float(np.median(radius)), base_az=base_az)
    return dict(strands=pts, root_uv=uv.astype(np.float32), color=base, color_tip=tip,
                radius=radius, size=np.float32(size), params=params)


def save_patch(d, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, strands=d["strands"], root_uv=d["root_uv"], color=d["color"], color_tip=d["color_tip"],
             radius=d["radius"], size=d["size"], params=json.dumps(d["params"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="/tmp/realdog.npz")
    ap.add_argument("--size", type=float, default=1.0); ap.add_argument("--color_scheme", default="")
    ap.add_argument("--coat", default=""); ap.add_argument("--density_scale", type=float, default=1.0)
    a = ap.parse_args()
    d = generate_realdog_patch(a.seed, a.size, color_scheme=a.color_scheme or None, coat=a.coat or None, density_scale=a.density_scale)
    save_patch(d, a.out); p = d["params"]
    S = d["strands"]; arc = np.linalg.norm(np.diff(S, axis=1), axis=-1).sum(1)
    print(f"realdog coat={p['coat']:11s} N={len(S):6d} scheme={p['color_scheme']:8s} len_med={p['length_med']:.3f} "
          f"({np.median(arc)/float(d['size']):.2f}x) elev_med={p['elev_med_deg']:.0f}deg curl_med={p['curl_med']:.4f} rad={p['radius']:.4f}")
    print("saved:", a.out)
