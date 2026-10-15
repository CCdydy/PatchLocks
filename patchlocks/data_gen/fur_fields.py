"""Deterministic fur-FIELD export (Stage-1 field-first route). The generators draw smooth spatial
fields + structure params; here we sample those exact continuous fields on a cell-center grid so the
GT is the underlying field, NOT a noisy estimate binned from stochastic strand landings.
Shared by gen v3 (pure scalar-field closures) and gen v4 (structure params). Fields saved at 16 & 32.
"""
import numpy as np


def cell_grid(res):
    c = (np.arange(res) + 0.5) / res
    gu, gv = np.meshgrid(c, c)                                  # gu=u (col), gv=v (row); index [v,u]
    return np.stack([gu.ravel(), gv.ravel()], 1).astype(np.float32)   # (res*res,2)


def dir_from_az_elev(az, elev_rad):
    return np.stack([np.cos(az) * np.cos(elev_rad), np.sin(az) * np.cos(elev_rad), np.sin(elev_rad)], 1).astype(np.float32)


def _pack(res, maps):
    out = {}
    for k, v in maps.items():
        v = np.asarray(v, np.float32)
        out[k] = v.reshape(res, res, -1) if v.ndim > 1 else v.reshape(res, res)
    return out


def v3_fields(res, b, fc, *, base_az, az_spread, swirl_c, pal_rgb, radius_val):
    """fc = dict of the live scalar-field closures from gen_fur_patch_v3.generate_patch."""
    uv = cell_grid(res)
    lam = (0.3 + 0.7 * fc["f_density"](uv)).astype(np.float32)
    lo, hi = b["length"]; length = (lo + (hi - lo) * fc["f_length"](uv)).astype(np.float32)
    e_lo, e_hi = b["elev"]; elev = np.deg2rad(e_lo + (e_hi - e_lo) * fc["f_elev"](uv)).astype(np.float32)
    if b["swirl"]:
        d = uv - swirl_c; az = np.arctan2(d[:, 1], d[:, 0]) + np.pi / 2 + (fc["f_az"](uv) - 0.5) * 0.5
    else:
        az = base_az + (fc["f_az"](uv) - 0.5) * 2 * az_spread
    direction = dir_from_az_elev(az, elev)
    c_lo, c_hi = b["curl"]; curl = (c_lo + (c_hi - c_lo) * fc["f_curl"](uv)).astype(np.float32)
    n = len(pal_rgb); cc = fc["f_col"](uv) * (n - 1); clo = np.floor(cc).astype(int)
    chi = np.minimum(clo + 1, n - 1); cfr = (cc - clo)[:, None]
    color = (pal_rgb[clo] * (1 - cfr) + pal_rgb[chi] * cfr) * (0.78 + 0.42 * fc["f_val"](uv))[:, None]
    radius = np.full(len(uv), radius_val, np.float32)
    type_mix = np.ones(len(uv), np.float32)                    # single coat = all guard
    valid = (lam > 0.2).astype(np.float32)
    return _pack(res, dict(lambda_map=lam, direction_map=direction, length_mu=length, curl_mu=curl,
                           elev_mu=elev, color_map=color.astype(np.float32), radius_mu=radius,
                           type_mix=type_mix, valid_mask=valid))


def v4_fields(res, b, fc, st, *, pal_rgb, radius_val):
    """v4: structure params `st` (centers/lines/grad/palette modes) hoisted out so fields match strands.
    fc = scalar-field closures; st = dict from gen_fur_patch_v4 with deterministic structure params."""
    uv = cell_grid(res)
    lam = _v4_keep(st["dens_mode"], uv, st, fc["f_density"]).astype(np.float32)
    lo, hi = b["length"]
    ramp = np.clip(0.5 + st["grad_mag"] * ((uv - 0.5) @ st["grad_dir"]), 0.05, 1.0)
    length = (lo + (hi - lo) * (0.65 * ramp + 0.35 * fc["f_length"](uv))).astype(np.float32)
    e_lo, e_hi = b["elev"]; elev = np.deg2rad(e_lo + (e_hi - e_lo) * fc["f_elev"](uv)).astype(np.float32)
    az = _v4_az(st["az_mode"], uv, st, fc["f_az"])
    direction = dir_from_az_elev(az, elev)
    c_lo, c_hi = b["curl"]; curl = (c_lo + (c_hi - c_lo) * fc["f_curl"](uv)).astype(np.float32)
    color = _v4_colour(st["col_mode"], uv, pal_rgb, fc["f_col"], fc["f_val"])
    radius = np.full(len(uv), radius_val, np.float32)
    type_mix = np.ones(len(uv), np.float32)
    valid = (lam > 0.2).astype(np.float32)
    return _pack(res, dict(lambda_map=lam, direction_map=direction, length_mu=length, curl_mu=curl,
                           elev_mu=elev, color_map=color.astype(np.float32), radius_mu=radius,
                           type_mix=type_mix, valid_mask=valid))


# ---- pure (rng-free) v4 structure functions; identical math to gen_fur_patch_v4, params hoisted ----
def _v4_az(mode, uv, st, f_az):
    if mode == "flow":
        return st["base_az"] + (f_az(uv) - 0.5) * 2 * st["az_spread"]
    if mode == "parting":
        side = np.sign((uv - st["pc"]) @ st["pn"]); side[side == 0] = 1
        v = side[:, None] * st["pn"]
        return np.arctan2(v[:, 1], v[:, 0]) + (f_az(uv) - 0.5) * 0.4
    if mode == "swirl":
        d = uv - st["sc"]; return np.arctan2(d[:, 1], d[:, 0]) + np.pi / 2 + (f_az(uv) - 0.5) * 0.4
    d = uv - st["sc"]; return np.arctan2(d[:, 1], d[:, 0]) + (f_az(uv) - 0.5) * 0.4


def _v4_keep(mode, uv, st, f_density):
    base = 0.30 + 0.70 * f_density(uv)
    if mode == "tuft":
        p = np.zeros(len(uv))
        for c, s in st["tufts"]:
            p += np.exp(-((uv - c) ** 2).sum(1) / (2 * s * s))
        return np.clip(0.20 + 0.95 * np.clip(p, 0, 1) + 0.3 * f_density(uv), 0, 1)
    if mode == "bald":
        c, s = st["bald"]; hole = np.exp(-((uv - c) ** 2).sum(1) / (2 * s * s))
        return np.clip(base * (1 - 0.92 * hole), 0, 1)
    return base


def _v4_colour(mode, uv, pal_rgb, f_col, f_val):
    n = len(pal_rgb)
    if mode == "spots" and n >= 2:
        sel = np.zeros(len(uv))
        # spots params folded into f_col fallback handled by caller passing precomputed; use gradient if absent
        fc = f_col(uv) * (n - 1); clo = np.floor(fc).astype(int); chi = np.minimum(clo + 1, n - 1); cfr = (fc - clo)[:, None]
        base = pal_rgb[clo] * (1 - cfr) + pal_rgb[chi] * cfr
    else:
        fc = f_col(uv) * (n - 1); clo = np.floor(fc).astype(int); chi = np.minimum(clo + 1, n - 1); cfr = (fc - clo)[:, None]
        base = pal_rgb[clo] * (1 - cfr) + pal_rgb[chi] * cfr
    return (base * (0.78 + 0.42 * f_val(uv))[:, None]).astype(np.float32)
