#!/usr/bin/env python3
"""Fur generator v5 — REALISTIC PET FUR. Goal: each patch should read as real cat/dog fur, with the
defining structure of real coats (double coat = dense soft undercoat + sparser stiff guard hairs), realistic
breed-like archetypes, orthogonal color schemes (solid/tabby/bicolor/tortie/tipped/two_tone) and a
finish/material axis (matte/normal/glossy/wet) consumed by the renderer (roughness/specular/translucency).

Design:
  ARCHETYPES = structural coat presets (cat/dog, single/double), each with density/length/curl/clump/elev
               and (for double coat) undercoat ratio + bimodal undercoat/guard length & radius.
  COLOR_SCHEMES = orthogonal, deterministic spatial color (so appearance is recoverable from views).
  FINISHES = per-patch material (roughness/specular/transmission); 'wet' also clumps + darkens the coat.

Keeps the v1..v4 return contract (strands, root_uv, color, color_tip, radius, size, params) so the rest of
the pipeline is unchanged; params gains: archetype, coat, color_scheme, finish, material{...}, uc_frac.
Run:  python patchlocks/data_gen/gen_fur_patch_v5.py --archetype cat_short_plush --seed 0 --out /tmp/p.npz
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gen_fur_patch_v2 import scalar_field, _unit
from gen_fur_patch_v3 import NATURAL as _NAT
from gen_fur_patch_v4 import _az_field, _keep_prob, AZ_MODES, DENS_MODES

# extend natural palette with a few more real coat colors
NATURAL = dict(_NAT)
NATURAL.update({"chocolate": (0.12, 0.07, 0.045), "fawn": (0.55, 0.43, 0.30), "blue_gray": (0.40, 0.43, 0.46),
                "ginger": (0.62, 0.32, 0.12), "silver": (0.66, 0.66, 0.67), "charcoal": (0.10, 0.10, 0.11)})

# ---- structural archetypes (real-coat presets). coat: single | double ----
ARCHETYPES = {
    "cat_short_sleek":  dict(coat="single", length=(0.10, 0.20), density=(1400, 2400), curl=(0.00, 0.012), clump=(0.0, 0.12), elev=(35, 70), rad=(0.0014, 0.0026)),  # siamese / short sleek
    "cat_short_plush":  dict(coat="double", density=(2600, 5000), uc_ratio=(0.65, 0.85), uc_len=(0.06, 0.14), guard_len=(0.14, 0.26), curl=(0.0, 0.012), clump=(0.0, 0.15), elev=(35, 70), uc_rad=(0.0008, 0.0016), guard_rad=(0.0014, 0.0026)),  # british/domestic shorthair
    "cat_long_fluffy":  dict(coat="double", density=(2200, 4200), uc_ratio=(0.50, 0.70), uc_len=(0.18, 0.35), guard_len=(0.45, 0.85), curl=(0.01, 0.03), clump=(0.2, 0.55), elev=(20, 55), uc_rad=(0.0010, 0.0020), guard_rad=(0.0016, 0.0030)),  # persian / maine coon
    "dog_short_double": dict(coat="double", density=(2500, 4500), uc_ratio=(0.60, 0.80), uc_len=(0.06, 0.16), guard_len=(0.18, 0.34), curl=(0.0, 0.010), clump=(0.0, 0.12), elev=(20, 45), uc_rad=(0.0008, 0.0018), guard_rad=(0.0015, 0.0030)),  # shiba / lab
    "dog_long_smooth":  dict(coat="single", length=(0.45, 0.85), density=(1400, 2400), curl=(0.006, 0.020), clump=(0.1, 0.35), elev=(15, 45), rad=(0.0016, 0.0030)),  # golden / setter (guard-dominant smooth)
    "dog_curly":        dict(coat="single", length=(0.25, 0.50), density=(1800, 3000), curl=(0.035, 0.060), clump=(0.2, 0.50), elev=(35, 70), rad=(0.0014, 0.0026)),  # poodle / bichon
    "wiry_terrier":     dict(coat="single", length=(0.12, 0.28), density=(900, 1700),  curl=(0.0, 0.015), clump=(0.0, 0.15), elev=(25, 55), rad=(0.0018, 0.0034)),
    "thin_region":      dict(coat="single", length=(0.08, 0.30), density=(600, 1300),  curl=(0.0, 0.020), clump=(0.0, 0.12), elev=(25, 65), rad=(0.0016, 0.0030), gravity=(0.10, 0.30), noise=(0.006, 0.012), curl_freq=(0.5, 1.2), az_p=[0.60, 0.25, 0.05, 0.10]),  # belly / legs / ears (region filler)
    "generic_medium":   dict(coat="single", length=(0.25, 0.55), density=(1200, 2200), curl=(0.01, 0.030), clump=(0.1, 0.35), elev=(30, 70), rad=(0.0014, 0.0028)),
    "clumped_wet":      dict(coat="single", length=(0.30, 0.60), density=(1400, 2600), curl=(0.0, 0.020), clump=(0.45, 0.80), elev=(25, 60), rad=(0.0014, 0.0028)),  # matted / wet look
    # ---- v6 redesign: census-grounded, breed-correct geometry. per-archetype gravity/noise/curl_freq/az_p override the old globals. az_p over AZ_MODES=[flow,parting,swirl,crown]. ----
    "stiff_double":     dict(coat="double", density=(3000, 5200), uc_ratio=(0.62, 0.80), uc_len=(0.06, 0.13), guard_len=(0.13, 0.22), curl=(0.0, 0.006), clump=(0.0, 0.10), elev=(65, 85), uc_rad=(0.0008, 0.0016), guard_rad=(0.0018, 0.0034), gravity=(0.03, 0.07), noise=(0.004, 0.007), curl_freq=(0.3, 0.6), az_p=[0.55, 0.30, 0.05, 0.10], stand_off=True),  # shiba/husky/akita/corgi/pom — STAND-OFF (the data-gap fix)
    "smooth_short":     dict(coat="single", length=(0.06, 0.16), density=(2500, 4200), curl=(0.0, 0.010), clump=(0.0, 0.08), elev=(15, 40), rad=(0.0014, 0.0026), gravity=(0.10, 0.25), noise=(0.004, 0.008), curl_freq=(0.4, 0.7), az_p=[0.70, 0.22, 0.02, 0.06]),  # lab/beagle/pug — sleek flat (short, lies close)
    "long_straight":    dict(coat="single", length=(0.55, 1.00), density=(1600, 2600), curl=(0.0, 0.012), clump=(0.10, 0.35), elev=(10, 30), rad=(0.0014, 0.0028), gravity=(0.40, 0.60), noise=(0.004, 0.010), curl_freq=(0.5, 0.9), az_p=[0.65, 0.32, 0.0, 0.03]),  # maltese/yorkie/silky — long hanging curtain
    "curly_teddy":      dict(coat="single", length=(0.22, 0.45), density=(2200, 3400), curl=(0.035, 0.065), clump=(0.20, 0.50), elev=(40, 72), rad=(0.0014, 0.0026), gravity=(0.05, 0.18), noise=(0.006, 0.012), curl_freq=(1.0, 1.4), az_p=[0.45, 0.20, 0.15, 0.20]),  # poodle/bichon/doodle — tight curls
    "wavy_long":        dict(coat="double", density=(1800, 3000), uc_ratio=(0.45, 0.65), uc_len=(0.18, 0.34), guard_len=(0.45, 0.85), curl=(0.010, 0.030), clump=(0.15, 0.45), elev=(15, 50), uc_rad=(0.0010, 0.0020), guard_rad=(0.0016, 0.0030), gravity=(0.30, 0.55), noise=(0.006, 0.012), curl_freq=(0.5, 1.0), az_p=[0.60, 0.30, 0.03, 0.07]),  # cocker/golden — long wavy + feathering
    "wiry":             dict(coat="single", length=(0.12, 0.28), density=(1200, 2200), curl=(0.008, 0.025), clump=(0.05, 0.20), elev=(30, 60), rad=(0.0020, 0.0036), gravity=(0.08, 0.20), noise=(0.006, 0.012), curl_freq=(1.2, 2.0), az_p=[0.55, 0.25, 0.08, 0.12]),  # schnauzer/scottie — harsh crinkle
}
# v6 coat-type taxonomy + census-grounded sampling weights (sum=1). Legacy archetypes stay defined but get 0 weight.
V6_COATS = ["curly_teddy", "wavy_long", "stiff_double", "smooth_short", "thin_region", "long_straight", "wiry"]
V6_SHARES = [0.26, 0.23, 0.20, 0.10, 0.09, 0.07, 0.05]
ARCH_NAMES = list(ARCHETYPES)

COLOR_SCHEMES = ["solid", "two_tone", "tabby", "bicolor", "tortie", "tipped"]
# realistic solid coat colors
SOLID_COLORS = ["black", "charcoal", "white", "off_white", "gray", "blue_gray", "silver",
                "brown", "dark_brown", "chocolate", "tan", "fawn", "cream", "orange_tabby", "ginger"]
FINISHES = {
    "matte":  dict(roughness=(0.55, 0.75), specular=(0.05, 0.25), transmission=(0.00, 0.08), coat=(0.0, 0.05), darken=1.00, sat=1.00),
    "normal": dict(roughness=(0.40, 0.55), specular=(0.25, 0.45), transmission=(0.05, 0.18), coat=(0.0, 0.10), darken=1.00, sat=1.00),
    "glossy": dict(roughness=(0.25, 0.40), specular=(0.45, 0.70), transmission=(0.05, 0.20), coat=(0.10, 0.30), darken=0.96, sat=1.05),
    "wet":    dict(roughness=(0.10, 0.26), specular=(0.60, 0.90), transmission=(0.00, 0.10), coat=(0.20, 0.45), darken=0.80, sat=1.12),
}
FINISH_NAMES = list(FINISHES)


def _grow(roots, d0, length, clump_strength, gravity, noise_amp, curl_amp, curl_freq, curl_phase, rng, L=32, n_clump=(4, 12), size=1.0, uv=None):
    """Shared strand growth: clump attractors + gravity + per-strand noise, then a helical curl offset."""
    N = len(roots); seg = (length / (L - 1)).astype(np.float32)
    nc = int(rng.integers(*n_clump)); cuv = rng.uniform(0.05, 0.95, (nc, 2))
    near = np.argmin(((uv[:, None] - cuv[None]) ** 2).sum(-1), 1)
    clump_xyz = np.concatenate([(cuv[near] - 0.5) * size, np.zeros((N, 1))], 1).astype(np.float32)
    pts = np.zeros((N, L, 3), np.float32); pts[:, 0] = roots
    dirs = _unit(d0 + rng.normal(0, 0.015, (N, 3)).astype(np.float32)); gz = np.array([0, 0, -1.0], np.float32)
    for k in range(1, L):
        t = k / (L - 1); to_c = _unit(clump_xyz - pts[:, k - 1])
        dirs = _unit(dirs + clump_strength * t * to_c + gravity * t * gz + noise_amp * rng.normal(0, 1, (N, 3)).astype(np.float32))
        pts[:, k] = pts[:, k - 1] + seg[:, None] * dirs; pts[:, k, 2] = np.maximum(pts[:, k, 2], 0.0)
    axis = _unit(d0); ref = np.tile(np.array([0, 0, 1.0], np.float32), (N, 1))
    ref[np.abs((axis * ref).sum(1)) > 0.95] = np.array([1, 0, 0.0], np.float32)
    e1 = _unit(np.cross(axis, ref)); e2 = _unit(np.cross(axis, e1)); tt = np.linspace(0, 1, L)[None]
    ang = 2 * np.pi * curl_freq * tt + curl_phase[:, None]; amp = curl_amp[:, None] * tt
    # planar-dominant wave (sin on e1, small e2) reads as natural fur waviness; a full circular helix
    # self-loops into wire-loop artifacts at high freq/amp. 0.25*e2 keeps a little 3D.
    pts = pts + (amp[..., None] * (np.sin(ang)[..., None] * e1[:, None] + 0.25 * np.cos(ang)[..., None] * e2[:, None])).astype(np.float32)
    pts[..., 2] = np.maximum(pts[..., 2], 0.0); pts[:, 0] = roots
    return pts


def _color_scheme(scheme, uv, rng, f_col, f_val):
    """Deterministic spatial color -> (base_rgb (N,3), is_dark_region for tip handling). Realistic pet palettes."""
    N = len(uv)
    def C(name): return np.array(NATURAL[name], np.float32)
    if scheme == "solid":
        c = C(str(rng.choice(SOLID_COLORS))); base = np.tile(c, (N, 1))
    elif scheme == "two_tone":
        a, b = rng.choice(SOLID_COLORS, 2, replace=False); fc = f_col(uv)
        base = C(a)[None] * (1 - fc[:, None]) + C(b)[None] * fc[:, None]
    elif scheme == "tabby":                                            # stripes (mackerel/classic)
        light = str(rng.choice(["cream", "tan", "fawn", "off_white", "silver", "ginger"]))
        dark = str(rng.choice(["dark_brown", "brown", "charcoal", "black", "chocolate"]))
        th = rng.uniform(0, np.pi); dv = np.array([np.cos(th), np.sin(th)])
        band = 0.5 * (1 + np.sin(2 * np.pi * rng.uniform(3.5, 7) * (uv @ dv) + 1.5 * (f_col(uv) - 0.5)))
        band = np.clip((band - 0.45) * 3 + 0.5, 0, 1)                  # sharpen stripes
        base = C(light)[None] * (1 - band[:, None]) + C(dark)[None] * band[:, None]
    elif scheme == "bicolor":                                          # two spatial regions (e.g. white + color)
        col = str(rng.choice(["black", "brown", "orange_tabby", "gray", "tan", "ginger", "chocolate"]))
        pc = rng.uniform(0.3, 0.7, 2); th = rng.uniform(0, np.pi); n = np.array([-np.sin(th), np.cos(th)])
        s = (uv - pc) @ n + 0.10 * (f_val(uv) - 0.5); reg = (s > 0).astype(np.float32)
        base = C("white")[None] * (1 - reg[:, None]) + C(col)[None] * reg[:, None]
    elif scheme == "tortie":                                           # calico/tortoiseshell multi-patch
        cols = [C(c) for c in rng.choice(["black", "orange_tabby", "white", "brown", "cream", "ginger"], 3, replace=False)]
        sel = np.zeros((N, 3)); owner = np.full(N, 0); score = np.full(N, -1.0)
        for ci, c in enumerate(cols):
            np_ = int(rng.integers(2, 5)); sc = np.zeros(N)
            for _ in range(np_):
                cc = rng.uniform(0.1, 0.9, 2); ss = rng.uniform(0.07, 0.16)
                sc = np.maximum(sc, np.exp(-((uv - cc) ** 2).sum(1) / (2 * ss * ss)))
            take = sc > score; owner[take] = ci; score[take] = sc[take]
        base = np.stack([cols[o] for o in owner])
    else:                                                              # tipped / agouti (banded): handled in tip
        c = C(str(rng.choice(["tan", "fawn", "gray", "silver", "ginger", "cream"]))); base = np.tile(c, (N, 1))
    # micro spatial value modulation + tiny per-strand noise
    base = (base * (0.82 + 0.34 * f_val(uv))[:, None] + rng.normal(0, 0.012, base.shape)).clip(0, 1).astype(np.float32)
    return base


def generate_patch(seed=0, archetype="generic_medium", size=1.0, color_scheme=None, finish=None):
    a = ARCHETYPES[archetype]; rng = np.random.default_rng(seed); L = 32
    double = a["coat"] == "double"
    f_density = scalar_field(rng); f_length = scalar_field(rng); f_az = scalar_field(rng)
    f_elev = scalar_field(rng); f_curl = scalar_field(rng); f_phase = scalar_field(rng)
    f_col = scalar_field(rng, n_freq=3); f_val = scalar_field(rng, n_freq=3)

    az_mode = str(rng.choice(AZ_MODES, p=a.get("az_p")))  # breed-weighted flow; az_p=None -> uniform (legacy archetypes). kills the old uniform 25% vortex.
    dens_mode = str(rng.choice(DENS_MODES, p=[0.5, 0.32, 0.18]))
    color_scheme = color_scheme or str(rng.choice(COLOR_SCHEMES, p=[0.30, 0.18, 0.18, 0.14, 0.10, 0.10]))
    finish = finish or str(rng.choice(FINISH_NAMES, p=[0.30, 0.45, 0.18, 0.07]))
    base_az = rng.uniform(0, 2 * np.pi); az_spread = float(rng.uniform(0.10, 0.40))

    N_t = int(rng.integers(*a["density"]))
    cand = rng.uniform(0, 1, (N_t * 3, 2)).astype(np.float32)
    keep = rng.uniform(0, 1, len(cand)) < _keep_prob(dens_mode, cand, rng, f_density)
    uv = cand[keep][:N_t]; N = len(uv)
    roots = np.stack([(uv[:, 0] - 0.5) * size, (uv[:, 1] - 0.5) * size, np.zeros(N)], 1).astype(np.float32)

    # ---- per-strand length / radius / curl (bimodal for double coat) ----
    if double:
        uc_ratio = float(rng.uniform(*a["uc_ratio"])); is_uc = rng.uniform(0, 1, N) < uc_ratio
        length = np.where(is_uc, rng.uniform(*a["uc_len"], N), rng.uniform(*a["guard_len"], N)).astype(np.float32)
        length *= (0.85 + 0.30 * f_length(uv))
        radius = np.where(is_uc, rng.uniform(*a["uc_rad"], N), rng.uniform(*a["guard_rad"], N)).astype(np.float32)
        c_lo, c_hi = a["curl"]; curl_amp = np.where(is_uc, rng.uniform(c_lo, c_hi, N), rng.uniform(0.0, 0.004, N)).astype(np.float32)  # guard straighter
    else:
        is_uc = np.zeros(N, bool); lo, hi = a["length"]; base_len = float(rng.uniform(lo, hi))
        grad_dir = _unit(rng.normal(0, 1, 2).astype(np.float64)); grad_mag = float(rng.uniform(0.5, 1.4))
        ramp = np.clip(0.5 + grad_mag * ((uv - 0.5) @ grad_dir), 0.0, 1.0)
        length = (base_len * (0.5 + 1.0 * (0.6 * ramp + 0.4 * f_length(uv)))).clip(0.04, None).astype(np.float32)
        c_lo, c_hi = a["curl"]; base_curl = float(rng.uniform(c_lo, c_hi)); curl_amp = (base_curl * (0.3 + 1.4 * f_curl(uv))).astype(np.float32)
        rlo, rhi = a["rad"]; radius = np.full(N, float(rng.uniform(rlo, rhi)), np.float32)

    e_lo, e_hi = a["elev"]; elev = np.deg2rad(e_lo + (e_hi - e_lo) * f_elev(uv))
    az = _az_field(az_mode, uv, rng, base_az, az_spread, f_az)
    d0 = _unit(np.stack([np.cos(az) * np.cos(elev), np.sin(az) * np.cos(elev), np.sin(elev)], 1).astype(np.float32))

    clump_strength = float(rng.uniform(*a["clump"]))                                                   # per-archetype gravity/noise/curl_freq (v6); a.get falls back to legacy globals
    gravity = float(rng.uniform(*a.get("gravity", (0.05, 0.40)))); noise_amp = float(rng.uniform(*a.get("noise", (0.004, 0.014))))
    curl_freq = float(rng.uniform(*a.get("curl_freq", (0.5, 1.2)))); curl_phase = (2 * np.pi * f_phase(uv) + rng.normal(0, 0.25, N)).astype(np.float32)

    fin = FINISHES[finish]
    if finish == "wet":                                                # wet = matted clumps + flatter + darker
        clump_strength = min(0.85, clump_strength + 0.25); curl_amp = curl_amp * 0.6
        if not a.get("stand_off"): gravity = min(0.5, gravity + 0.1)   # stand-off coats (stiff_double) must NOT be flattened by the wet bump
    pts = _grow(roots, d0, length, clump_strength, gravity, noise_amp, curl_amp, curl_freq, curl_phase, rng,
                L=L, n_clump=(3, 9) if double else (4, 12), size=size, uv=uv)

    # ---- color ----
    base = _color_scheme(color_scheme, uv, rng, f_col, f_val)
    uc_tip = lambda: np.clip(base * 1.06 + 0.02, 0, 1)                  # undercoat tip slightly lighter
    if color_scheme == "tipped":                                       # agouti banding: tip darker than root
        dark = np.array(NATURAL[str(rng.choice(["dark_brown", "black", "charcoal", "brown"]))], np.float32)
        tipf = float(rng.uniform(0.45, 0.75)); agouti = (base * (1 - tipf) + dark[None] * tipf).clip(0, 1)
        # tipped+double: undercoat stays lighter, guard gets the agouti dark tip (mutually-exclusive branches)
        tip = (np.where(is_uc[:, None], uc_tip(), agouti) if double else agouti).astype(np.float32)
    elif double:                                                       # undercoat tip lighter; guard tip darker (tipping)
        gtip = float(rng.uniform(0.45, 0.65)); tip = np.where(is_uc[:, None], uc_tip(), (base * gtip).clip(0, 1)).astype(np.float32)
    else:
        tip = np.clip(base * float(rng.uniform(1.02, 1.12)) + 0.02, 0, 1).astype(np.float32)
    base = (base * fin["darken"]).clip(0, 1).astype(np.float32); tip = (tip * fin["darken"]).clip(0, 1).astype(np.float32)
    # saturation tweak for wet/glossy
    if fin["sat"] != 1.0:
        for arr in (base, tip):
            g = arr.mean(1, keepdims=True); arr[:] = np.clip(g + (arr - g) * fin["sat"], 0, 1)

    material = dict(finish=finish, roughness=float(rng.uniform(*fin["roughness"])), specular=float(rng.uniform(*fin["specular"])),
                    transmission=float(rng.uniform(*fin["transmission"])), coat=float(rng.uniform(*fin["coat"])))
    params = dict(version=5, archetype=archetype, coat=a["coat"], seed=int(seed), n_strands=int(N),
                  color_scheme=color_scheme, finish=finish, material=material,
                  structure=dict(az=az_mode, density=dens_mode), uc_frac=float(is_uc.mean()) if double else 0.0,
                  clump_strength=clump_strength, length_med=float(np.median(length)), curl_med=float(np.median(curl_amp)),
                  radius=float(np.median(radius)))
    return dict(strands=pts, root_uv=uv.astype(np.float32), color=base, color_tip=tip,
                radius=radius, size=np.float32(size), params=params)


def save_patch(d, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, strands=d["strands"], root_uv=d["root_uv"], color=d["color"], color_tip=d["color_tip"],
             radius=d["radius"], size=d["size"], params=json.dumps(d["params"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--archetype", default="cat_short_plush")
    ap.add_argument("--color_scheme", default=""); ap.add_argument("--finish", default="")
    ap.add_argument("--out", default="/tmp/v5_patch.npz")
    a = ap.parse_args()
    d = generate_patch(a.seed, a.archetype, color_scheme=a.color_scheme or None, finish=a.finish or None)
    save_patch(d, a.out); p = d["params"]
    print(f"v5 {a.archetype} N={len(d['strands'])} coat={p['coat']} color={p['color_scheme']} finish={p['finish']} "
          f"uc_frac={p['uc_frac']:.2f} len_med={p['length_med']:.3f} curl_med={p['curl_med']:.4f}")
    print("saved:", a.out)
