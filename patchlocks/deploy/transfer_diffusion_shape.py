#!/usr/bin/env python3
"""Shape the grown strands like the diffusion predicted them, patch by patch (Step 3c).

The grow stamps one canonical primitive everywhere, built from the statistics of a few of the
straightest reliable patches, so every strand comes out nearly straight, while the diffusion
predicts for every patch strands that bend and wave, differently from patch to patch. Here each
patch's predicted strands are decoded from Step 1's latent map with the strand codec, smoothed a
little along their length, and expressed relative to their own chord: along it, sideways, and
towards the surface normal, in units of the chord length. A predicted strand serves as a template
only if it is plausible (straightness >= --min_straightness: no loops or hooks); patches with fewer
than --min_templates plausible strands take templates from the nearest patch that has enough.

Every grown strand keeps its chord (root, direction and reach come from the multi-view fit and the
silhouette calibration) and takes the bend of a template drawn from its patch; --strength blends
between straight (0) and the template (1). The predicted within-patch length spread sets each
strand's length variation (--length_spread, as an exponent on length / patch median), and
--radius_scale thins the strands (the grow's radius covers the skin and reads as straw).

  python patchlocks/deploy/transfer_diffusion_shape.py --groom <run>/grow/surfpack_v2fur.npz \\
      --proposals <run>/proposals_maps/v14_surfpack_proposals.npz \\
      --surfpack <in>/surfpack/surface_patch_dataset.npz \\
      --region_prior <run>/v2_region_prior.annotated.npz --regions body,legs \\
      --out <run>/grow/surfpack_v2fur_dshape.npz
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bake_mesh_albedo_robust import load_obj_mesh  # noqa: E402
from extract_fur_character import load_codec  # noqa: E402


def resample_by_arc(S, n):
    """(N, P, 3) torch -> (N, n, 3), points evenly spaced along each strand; also the arc lengths."""
    seg = torch.linalg.norm(S[:, 1:] - S[:, :-1], dim=2)
    cum = torch.cat([torch.zeros_like(seg[:, :1]), torch.cumsum(seg, 1)], 1)
    arc = cum[:, -1]
    t = torch.linspace(0, 1, n, device=S.device)[None] * arc[:, None]
    i = torch.searchsorted(cum.contiguous(), t.contiguous()).clamp(1, S.shape[1] - 1)
    c0, c1 = torch.gather(cum, 1, i - 1), torch.gather(cum, 1, i)
    w = ((t - c0) / (c1 - c0).clamp(min=1e-12))[..., None]
    p0 = torch.gather(S, 1, (i - 1)[..., None].expand(-1, -1, 3))
    p1 = torch.gather(S, 1, i[..., None].expand(-1, -1, 3))
    return p0 + w * (p1 - p0), arc


def chord_coords(P, normal):
    """Coordinates of root-centred strands P (N, n, 3) along their chord, sideways and towards
    the normal side, in units of the chord length; also the chord length."""
    C = P[:, -1]
    L = np.linalg.norm(C, axis=1)
    e1 = C / np.maximum(L, 1e-12)[:, None]
    e3 = normal - (normal * e1).sum(1, keepdims=True) * e1
    bad = np.linalg.norm(e3, axis=1) < 1e-3
    if bad.any():                                            # chord along the normal: any perpendicular
        alt = np.cross(e1[bad], [1.0, 0.0, 0.0])
        small = np.linalg.norm(alt, axis=1) < 1e-3
        alt[small] = np.cross(e1[bad][small], [0.0, 1.0, 0.0])
        e3[bad] = alt
    e3 /= np.linalg.norm(e3, axis=1, keepdims=True)
    e2 = np.cross(e3, e1)
    coords = np.stack([(P * e1[:, None]).sum(2), (P * e2[:, None]).sum(2), (P * e3[:, None]).sum(2)], 2)
    return coords / np.maximum(L, 1e-12)[:, None, None], L, (e1, e2, e3)


def build_bank(a, zmap, valid, centers, n_pts, dev):
    codec, cnorm = load_codec(a.codec, dev)
    P = zmap.shape[0]
    bank = np.zeros((P, a.templates, n_pts, 3), np.float32)
    lrel = np.ones((P, a.templates), np.float32)
    count = np.zeros(P, np.int32)
    straight_all = []
    rng = np.random.default_rng(a.seed)
    zhat = np.array([0.0, 0.0, 1.0])
    for p in range(P):
        ys, xs = np.where(valid[p])
        if len(ys) < 8:
            continue
        sel = rng.choice(len(ys), min(3 * a.templates, len(ys)), replace=False)
        with torch.no_grad():
            pr = codec.decoder(torch.tensor(zmap[p][ys[sel], xs[sel]].astype(np.float32), device=dev), None, cnorm)
            St = pr["strand_positions"].float()
            St = St - St[:, :1]
            if a.smooth > 0:
                St = torch.as_tensor(gaussian_filter1d(St.cpu().numpy(), a.smooth, axis=1, mode="nearest"), device=dev)
                St = St - St[:, :1]
            Sr, arc = resample_by_arc(St, n_pts)
        Sr, arc = Sr.cpu().numpy().astype(np.float64), arc.cpu().numpy()
        coords, L, _ = chord_coords(Sr, np.repeat(zhat[None], len(Sr), 0))
        s = L / np.maximum(arc, 1e-12)
        straight_all.append(s)
        ok = (s >= a.min_straightness) & (L > 1e-6)
        k = int(min(ok.sum(), a.templates))
        if k == 0:
            continue
        keep = np.flatnonzero(ok)[:k]
        bank[p, :k] = coords[keep]
        med = np.median(arc[keep])
        lrel[p, :k] = arc[keep] / max(med, 1e-12)
        if k < a.templates:                                   # repeat to fill the bank
            rep = np.arange(a.templates - k) % k
            bank[p, k:] = bank[p, rep]
            lrel[p, k:] = lrel[p, rep]
        count[p] = k
    enough = count >= a.min_templates
    src = np.arange(P)
    if (~enough).any() and enough.any():
        tree = cKDTree(centers[enough])
        src[~enough] = np.flatnonzero(enough)[tree.query(centers[~enough])[1]]
    s_all = np.concatenate(straight_all) if straight_all else np.zeros(1)
    info = {"patches": int(P), "patches_with_enough_templates": int(enough.sum()),
            "decoded_straightness_p10_50_90": np.round(np.percentile(s_all, [10, 50, 90]), 3).tolist(),
            "template_pass_frac": round(float((s_all >= a.min_straightness).mean()), 3)}
    return bank, lrel, src, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groom", required=True)
    ap.add_argument("--proposals", required=True, help="Step 1 proposals with patch_z64_map")
    ap.add_argument("--surfpack", required=True, help="surface_patch_dataset.npz (valid texels, patch centres)")
    ap.add_argument("--region_prior", required=True)
    ap.add_argument("--regions", default="body,legs")
    ap.add_argument("--mesh", default="", help="furless mesh of the groom (default: its mesh_path)")
    ap.add_argument("--codec", default="patchlocks/models/fur_strand_codec_v6upright.pt")
    ap.add_argument("--templates", type=int, default=48, help="templates kept per patch")
    ap.add_argument("--min_templates", type=int, default=8)
    ap.add_argument("--min_straightness", type=float, default=0.7, help="plausible templates: no loops or hooks")
    ap.add_argument("--smooth", type=float, default=4.0, help="smoothing along the decoded strand (samples)")
    ap.add_argument("--strength", type=float, default=1.0, help="0 = keep straight, 1 = the template's bend")
    ap.add_argument("--length_spread", type=float, default=0.5, help="exponent on the template's length / patch median")
    ap.add_argument("--radius_scale", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    g = dict(np.load(a.groom, allow_pickle=True))
    S = g["strands"].astype(np.float32)
    n_pts = S.shape[1]
    d = np.load(a.proposals, allow_pickle=True)
    sp = np.load(a.surfpack, allow_pickle=True)
    bank, lrel, src, info = build_bank(a, d["patch_z64_map"], sp["valid"] > 0.5,
                                       np.asarray(sp["centers"], np.float64), n_pts, dev)
    print(json.dumps(info), flush=True)

    V, F, _ = load_obj_mesh(a.mesh or str(g["mesh_path"]))
    V = V.astype(np.float64)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-15)
    reg = np.load(a.region_prior, allow_pickle=True)["patch_region"].astype(str)
    row = g["root_prior_row"] if "root_prior_row" in g else g["root_patch"]
    idx = np.flatnonzero(np.isin(reg[np.clip(row, 0, len(reg) - 1)], a.regions.split(",")))
    rp = np.clip(g["root_patch"][idx], 0, len(src) - 1)
    rng = np.random.default_rng(a.seed + 1)
    lin = np.zeros((n_pts, 3))
    lin[:, 0] = np.linspace(0, 1, n_pts)
    straight_after = []
    for c0 in range(0, len(idx), 400_000):
        ii = idx[c0:c0 + 400_000]
        P = S[ii].astype(np.float64)
        root = P[:, 0]
        _, L, (e1, e2, e3) = chord_coords(P - root[:, None], fn[g["root_face"][ii]])
        t = rng.integers(0, a.templates, len(ii))
        s = src[rp[c0:c0 + 400_000]]
        T = lin[None] + a.strength * (bank[s, t].astype(np.float64) - lin[None])
        lf = np.clip(lrel[s, t] ** a.length_spread, 0.6, 1.5)
        Ls = (L * lf)[:, None, None]
        Q = root[:, None] + Ls * (T[..., :1] * e1[:, None] + T[..., 1:2] * e2[:, None] + T[..., 2:] * e3[:, None])
        S[ii] = Q.astype(np.float32)
        arc = np.linalg.norm(np.diff(Q, axis=1), axis=2).sum(1)
        straight_after.append(np.linalg.norm(Q[:, -1] - Q[:, 0], axis=1) / np.maximum(arc, 1e-12))
    g["strands"] = S
    g["radius"] = g["radius"].astype(np.float32).copy()
    g["radius"][idx] *= a.radius_scale
    g["shape_source"] = np.asarray(f"diffusion templates (transfer_diffusion_shape.py) on {a.regions}: "
                                   f"strength={a.strength} min_straightness={a.min_straightness} "
                                   f"length_spread={a.length_spread} radius_scale={a.radius_scale}")
    np.savez(a.out, **g)
    st = np.concatenate(straight_after) if straight_after else np.zeros(1)
    print(json.dumps({"strands": int(len(idx)),
                      "straightness_after_p10_50_90": np.round(np.percentile(st, [10, 50, 90]), 3).tolist()}))


if __name__ == "__main__":
    main()
