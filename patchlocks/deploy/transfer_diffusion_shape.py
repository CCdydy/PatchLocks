#!/usr/bin/env python3
"""Shape the grown strands like the diffusion predicted them, patch by patch (Step 3c).

The grow stamps one canonical primitive everywhere, built from the statistics of a few of the
straightest reliable patches, so every strand comes out nearly straight, while the diffusion
predicts for every patch strands that bend and wave, differently from patch to patch. Here each
patch's predicted strands are decoded from Step 1's latent map with the strand codec, smoothed a
little along their length, and expressed relative to their own chord: along it, sideways, and
towards the surface normal, in units of the chord length. A predicted strand serves as a template
only if it is plausible (straightness >= --min_straightness: no loops or hooks).

Which template a grown strand takes (--assign):
  texel   the prediction at the strand's root: one of the patches whose texels cover the root,
          drawn with weight exp(-(d / sigma)^2) on the distance d to the patch centre (sigma =
          --seam_sigma * patch size) so patch borders dither instead of seaming, and in it the four
          texels nearest the root, blended with Gaussian weights (at showcase densities a texel holds
          several roots, which would otherwise all copy one shape and read as a lock). The latent
          map is spatially coherent (neighbouring texels predict similar bends: waves and clumps),
          and this keeps that structure. An implausible texel gives way to the nearest plausible
          texel of its patch. Given several --proposals (Step 1 rerun with other --seed values),
          each texel's template is the average of its plausible predictions over the runs: bends
          the runs agree on survive and those they contradict cancel, so the diffusion itself
          decides how much of its bend to trust.
  random  --templates plausible predictions drawn anywhere in the root's patch, assigned at random.
Patches with fewer than --min_templates plausible predictions take them from the nearest patch that
has enough.

Every grown strand keeps its chord (root, direction and reach come from the multi-view fit and the
silhouette calibration) and takes the bend of its template; --strength blends between straight (0)
and the template (1). With --adaptive the blend is set per patch by the agreement of neighbouring
predictions: the mean cosine between the bends of texels --coherence_step apart. Neighbours share
the coat's real waves and clumps while the per-texel scatter is independent, so this agreement
estimates the share of the predicted bend that is signal, and scaling by it shrinks the bend where
the predictions contradict each other (a smooth coat the model saw little of) and keeps it where
they agree; --strength is then the upper bound. The predicted length spread sets each strand's
length variation (--length_spread, as an exponent on length / patch median), and --radius_scale
thins the strands (the grow's radius covers the skin and reads as straw).

  python patchlocks/deploy/transfer_diffusion_shape.py --groom <run>/grow/surfpack_v2fur.npz \\
      --proposals <run>/proposals_maps/v14_surfpack_proposals.npz \\
      --surfpack <in>/surfpack/surface_patch_dataset.npz \\
      --region_prior <run>/v2_region_prior.annotated.npz --regions body,legs \\
      --assign texel --adaptive --strength 1.0 --out <run>/grow/surfpack_v2fur_dshape.npz
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bake_mesh_albedo_robust import load_obj_mesh  # noqa: E402
from extract_fur_character import load_codec  # noqa: E402

ZHAT = np.array([0.0, 0.0, 1.0])                             # decoded strands: patch frame, normal along z


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


def smooth_along(S, sigma):
    """Gaussian smoothing of strands (N, P, 3) torch along their length, end samples repeated."""
    r = int(4.0 * sigma + 0.5)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=S.device, dtype=S.dtype) / sigma) ** 2)
    X = F.pad(S.permute(0, 2, 1).reshape(-1, 1, S.shape[1]), (r, r), mode="replicate")
    return F.conv1d(X, (k / k.sum())[None, None]).reshape(S.shape[0], 3, -1).permute(0, 2, 1)


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


def decode(codec, cnorm, z, n_pts, smooth, dev, batch=8192):
    """Latents (n, 64) -> templates: chord coordinates (n, n_pts, 3), arc lengths, straightness."""
    out = []
    for b0 in range(0, len(z), batch):
        with torch.no_grad():
            pr = codec.decoder(torch.tensor(np.asarray(z[b0:b0 + batch], np.float32), device=dev), None, cnorm)
            St = pr["strand_positions"].float()
            St = St - St[:, :1]
            if smooth > 0:
                St = smooth_along(St, smooth)
                St = St - St[:, :1]
            Sr, arc = resample_by_arc(St, n_pts)
        Sr, arc = Sr.cpu().numpy().astype(np.float64), arc.cpu().numpy()
        coords, L, _ = chord_coords(Sr, np.repeat(ZHAT[None], len(Sr), 0))
        out.append((coords.astype(np.float32), arc, L / np.maximum(arc, 1e-12)))
    return tuple(np.concatenate(x) for x in zip(*out))


def build_bank(a, zmap, valid, centers, n_pts, codec, cnorm, dev):
    """--assign random: per patch, --templates plausible predictions drawn anywhere in the patch."""
    P = zmap.shape[0]
    bank = np.zeros((P, a.templates, n_pts, 3), np.float32)
    lrel = np.ones((P, a.templates), np.float32)
    count = np.zeros(P, np.int32)
    straight_all = []
    rng = np.random.default_rng(a.seed)
    for p in range(P):
        ys, xs = np.where(valid[p])
        if len(ys) < 8:
            continue
        sel = rng.choice(len(ys), min(3 * a.templates, len(ys)), replace=False)
        coords, arc, s = decode(codec, cnorm, zmap[p][ys[sel], xs[sel]], n_pts, a.smooth, dev)
        straight_all.append(s)
        ok = s >= a.min_straightness
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


def assign_texels(roots, centers, surf, valid, psize, sigma, rng, k=8, blend=4):
    """For every root, one of the patches whose texels cover it (nearest valid texel within 1.5 texel
    spacings), drawn with weight exp(-(d / sigma)^2) on the distance d to the patch centre; then the
    `blend` texels of that patch nearest the root (flat index row * H + col) with Gaussian weights
    (0.6 texel spacings), so that the several strands rooted within one texel differ smoothly
    instead of copying one prediction."""
    H = valid.shape[1]
    spacing = psize / H
    trees = {}

    def tree(p):
        if p not in trees:
            ys, xs = np.nonzero(valid[p])
            trees[p] = (cKDTree(surf[p][ys, xs]), ys * H + xs) if len(ys) else None
        return trees[p]

    def by_patch(ids):
        order = np.argsort(ids, kind="stable")
        return np.split(order, np.flatnonzero(np.diff(ids[order])) + 1)

    cand = cKDTree(centers).query(roots, k=k)[1]
    dist = np.full(cand.shape, np.inf)
    for grp in by_patch(cand.ravel()):
        t = tree(cand.ravel()[grp[0]])
        if t is not None:
            dist.reshape(-1)[grp] = t[0].query(roots[grp // k])[0]
    dc = np.linalg.norm(roots[:, None] - centers[cand], axis=2)
    w = np.where(dist <= 1.5 * spacing, np.exp(-(dc / sigma) ** 2), 0.0)
    lost = w.sum(1) <= 0                                      # outside every patch: the nearest texel
    w[lost, np.argmin(dist[lost], 1)] = 1.0
    cw = np.cumsum(w, 1)
    j = np.minimum((cw < rng.random(len(w))[:, None] * cw[:, -1:]).sum(1), k - 1)
    patch = cand[np.arange(len(w)), j]
    tex = np.zeros((len(roots), blend), np.int64)
    tw = np.zeros((len(roots), blend))
    for grp in by_patch(patch):
        t = tree(patch[grp[0]])
        n = min(blend, len(t[1]))
        dd, ii = t[0].query(roots[grp], k=n)
        dd, ii = dd.reshape(len(grp), n), ii.reshape(len(grp), n)
        tex[grp, :n] = t[1][ii]
        tw[grp, :n] = np.exp(-(dd / (0.6 * spacing)) ** 2) + 1e-12
    return patch, tex, tw / tw.sum(1, keepdims=True), float(lost.mean())


def texel_templates(a, zmaps, centers, patch, tex, n_pts, codec, cnorm, dev):
    """--assign texel: decode the texels the strands use, in every Step 1 sample, and average each
    texel's plausible predictions over the samples (bends the samples agree on survive, those they
    contradict cancel); a texel without a plausible prediction gives way to the nearest plausible
    texel of its patch. Returns the templates, their length / patch median, and for every strand the
    rows of its templates (same shape as tex)."""
    H = zmaps[0].shape[1]
    uniq, inv = np.unique(patch[:, None] * (H * H) + tex, return_inverse=True)
    up, ut = uniq // (H * H), uniq % (H * H)
    acc = np.zeros((len(uniq), n_pts, 3))
    arc_acc = np.zeros(len(uniq))
    n_ok = np.zeros(len(uniq))
    energy = np.zeros(len(uniq))
    straight_all = []
    lin = np.zeros((n_pts, 3))
    lin[:, 0] = np.linspace(0, 1, n_pts)
    for zmap in zmaps:
        coords, arc, s = decode(codec, cnorm, zmap.reshape(len(zmap), H * H, -1)[up, ut], n_pts, a.smooth, dev)
        ok = s >= a.min_straightness
        acc[ok] += coords[ok]
        arc_acc[ok] += arc[ok]
        n_ok += ok
        energy[ok] += ((coords[ok] - lin) ** 2).sum((1, 2))
        straight_all.append(s)
    ok = n_ok > 0
    coords = (acc / np.maximum(n_ok, 1)[:, None, None]).astype(np.float32)
    arc = arc_acc / np.maximum(n_ok, 1)
    # share of the bend energy that survives the averaging: 1 = the samples agree, ~1/K = unrelated
    kept = ((coords - lin) ** 2).sum((1, 2)) / np.maximum(energy / np.maximum(n_ok, 1), 1e-12)
    src = np.arange(len(uniq))
    med = np.full(len(zmaps[0]), np.nan)
    yx = np.c_[ut // H, ut % H]
    starts = np.flatnonzero(np.r_[True, np.diff(up) > 0])
    lacking = []
    for s0, s1 in zip(starts, np.r_[starts[1:], len(up)]):
        good = s0 + np.flatnonzero(ok[s0:s1])
        if len(good) < min(a.min_templates, max(1, (s1 - s0) // 4)):
            lacking.append((s0, s1))
            continue
        med[up[s0]] = np.median(arc[good])
        bad = s0 + np.flatnonzero(~ok[s0:s1])
        if len(bad):
            src[bad] = good[cKDTree(yx[good]).query(yx[bad])[1]]
    have = np.flatnonzero(np.isfinite(med))
    rng = np.random.default_rng(a.seed)
    for s0, s1 in lacking:                                    # nearest patch with enough plausible ones
        donor = have[cKDTree(centers[have]).query(centers[up[s0]])[1]]
        pool = np.flatnonzero((up == donor) & ok)
        src[s0:s1] = rng.choice(pool, s1 - s0)
    lrel = (arc / np.maximum(med[up], 1e-12)).astype(np.float32)
    s_all = np.concatenate(straight_all)
    info = {"samples": len(zmaps), "texels_decoded": int(len(uniq)), "patches_used": int(len(starts)),
            "patches_lacking": len(lacking),
            "decoded_straightness_p10_50_90": np.round(np.percentile(s_all, [10, 50, 90]), 3).tolist(),
            "template_pass_frac": round(float((s_all >= a.min_straightness).mean()), 3)}
    if len(zmaps) > 1:
        info["bend_kept_p10_50_90"] = np.round(np.percentile(kept[ok], [10, 50, 90]), 3).tolist()
    return coords, lrel, src[inv.reshape(tex.shape)], info


def patch_coherence(a, zmap, valid, patches, n_pts, codec, cnorm, dev):
    """Mean cosine between the bends (sideways and towards the normal) of texels --coherence_step
    apart on a grid, per patch (NaN where too few texels are valid)."""
    H = valid.shape[1]
    g = np.arange(a.coherence_step // 2, H, a.coherence_step)
    gy, gx = np.meshgrid(g, g, indexing="ij")
    lin = np.zeros((n_pts, 3))
    lin[:, 0] = np.linspace(0, 1, n_pts)
    coh = np.full(len(valid), np.nan)
    for p in patches:
        m = valid[p][gy, gx]
        if m.sum() < 16:
            continue
        coords, _, _ = decode(codec, cnorm, zmap[p][gy[m], gx[m]], n_pts, a.smooth, dev)
        v = (coords - lin)[..., 1:].reshape(len(coords), -1)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        idx = -np.ones(m.shape, np.int64)
        idx[m] = np.arange(m.sum())
        i0 = np.r_[idx[:, :-1].ravel(), idx[:-1, :].ravel()]
        i1 = np.r_[idx[:, 1:].ravel(), idx[1:, :].ravel()]
        keep = (i0 >= 0) & (i1 >= 0)
        coh[p] = np.mean((v[i0[keep]] * v[i1[keep]]).sum(1))
    return coh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groom", required=True)
    ap.add_argument("--proposals", required=True, nargs="+",
                    help="Step 1 proposals with patch_z64_map; texel: several runs (other --seed) are averaged")
    ap.add_argument("--surfpack", required=True, help="surface_patch_dataset.npz (valid texels, patch centres)")
    ap.add_argument("--region_prior", required=True)
    ap.add_argument("--regions", default="body,legs")
    ap.add_argument("--mesh", default="", help="furless mesh of the groom (default: its mesh_path)")
    ap.add_argument("--codec", default="patchlocks/models/fur_strand_codec_v6upright.pt")
    ap.add_argument("--assign", default="random", choices=["random", "texel"],
                    help="texel: the prediction at the strand's root; random: any prediction of its patch")
    ap.add_argument("--seam_sigma", type=float, default=0.15,
                    help="texel: patch-choice falloff from the patch centre, in patch sizes")
    ap.add_argument("--templates", type=int, default=48, help="random: templates kept per patch")
    ap.add_argument("--min_templates", type=int, default=8)
    ap.add_argument("--min_straightness", type=float, default=0.7, help="plausible templates: no loops or hooks")
    ap.add_argument("--smooth", type=float, default=4.0, help="smoothing along the decoded strand (samples)")
    ap.add_argument("--strength", type=float, default=1.0, help="0 = keep straight, 1 = the template's bend")
    ap.add_argument("--adaptive", action="store_true",
                    help="scale the strength per patch by the agreement of neighbouring predictions")
    ap.add_argument("--coherence_step", type=int, default=4, help="adaptive: texel spacing of the neighbours compared")
    ap.add_argument("--length_spread", type=float, default=0.5, help="exponent on the template's length / patch median")
    ap.add_argument("--radius_scale", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    g = dict(np.load(a.groom, allow_pickle=True))
    S = g["strands"].astype(np.float32)
    n_pts = S.shape[1]
    zmaps = [np.load(p, allow_pickle=True)["patch_z64_map"] for p in a.proposals]
    zmap = zmaps[0]
    sp = np.load(a.surfpack, allow_pickle=True)
    valid = sp["valid"] > 0.5
    centers = np.asarray(sp["centers"], np.float64)
    codec, cnorm = load_codec(a.codec, dev)

    V, F_, _ = load_obj_mesh(a.mesh or str(g["mesh_path"]))
    V = V.astype(np.float64)
    fn = np.cross(V[F_[:, 1]] - V[F_[:, 0]], V[F_[:, 2]] - V[F_[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-15)
    reg = np.load(a.region_prior, allow_pickle=True)["patch_region"].astype(str)
    row = g["root_prior_row"] if "root_prior_row" in g else g["root_patch"]
    idx = np.flatnonzero(np.isin(reg[np.clip(row, 0, len(reg) - 1)], a.regions.split(",")))
    rng = np.random.default_rng(a.seed + 1)
    if a.assign == "texel":
        patch, tex, tw, lost = assign_texels(S[idx, 0].astype(np.float64), centers, sp["surf"], valid,
                                             float(sp["psize"]), a.seam_sigma * float(sp["psize"]), rng)
        table, lrel_table, tid, info = texel_templates(a, zmaps, centers, patch, tex, n_pts, codec, cnorm, dev)
        info["roots_outside_patches"] = round(lost, 4)
    else:
        bank, lrel, src, info = build_bank(a, zmap, valid, centers, n_pts, codec, cnorm, dev)
        patch = np.clip(g["root_patch"][idx], 0, len(src) - 1)
        tid = (src[patch] * a.templates + rng.integers(0, a.templates, len(idx)))[:, None]
        tw = np.ones((len(idx), 1))
        table, lrel_table = bank.reshape(-1, n_pts, 3), lrel.reshape(-1)
    del zmaps
    strength = np.full(len(zmap), a.strength)
    if a.adaptive:
        coh = patch_coherence(a, zmap, valid, np.unique(patch), n_pts, codec, cnorm, dev)
        used = np.isfinite(coh)
        strength = a.strength * np.clip(np.where(used, coh, np.nanmedian(coh)), 0.0, 1.0)
        info["coherence_p10_50_90"] = np.round(np.nanpercentile(coh, [10, 50, 90]), 3).tolist()
    info["strength_mean"] = round(float(strength[patch].mean()), 3)
    print(json.dumps(info), flush=True)

    lin = np.zeros((n_pts, 3))
    lin[:, 0] = np.linspace(0, 1, n_pts)
    straight_after = []
    for c0 in range(0, len(idx), 400_000):
        ii = idx[c0:c0 + 400_000]
        P = S[ii].astype(np.float64)
        root = P[:, 0]
        _, L, (e1, e2, e3) = chord_coords(P - root[:, None], fn[g["root_face"][ii]])
        t, tk = tid[c0:c0 + 400_000], tw[c0:c0 + 400_000]
        B = sum(tk[:, j, None, None] * table[t[:, j]] for j in range(t.shape[1]))
        w = strength[patch[c0:c0 + 400_000]][:, None, None]
        T = lin[None] + w * (B - lin[None])
        lf = np.clip((tk * lrel_table[t]).sum(1) ** a.length_spread, 0.6, 1.5)
        Ls = (L * lf)[:, None, None]
        Q = root[:, None] + Ls * (T[..., :1] * e1[:, None] + T[..., 1:2] * e2[:, None] + T[..., 2:] * e3[:, None])
        S[ii] = Q.astype(np.float32)
        arc = np.linalg.norm(np.diff(Q, axis=1), axis=2).sum(1)
        straight_after.append(np.linalg.norm(Q[:, -1] - Q[:, 0], axis=1) / np.maximum(arc, 1e-12))
    g["strands"] = S
    g["radius"] = g["radius"].astype(np.float32).copy()
    g["radius"][idx] *= a.radius_scale
    g["shape_source"] = np.asarray(f"diffusion templates (transfer_diffusion_shape.py) on {a.regions}: "
                                   f"assign={a.assign} samples={len(a.proposals)} strength={a.strength} "
                                   f"adaptive={a.adaptive} "
                                   f"min_straightness={a.min_straightness} length_spread={a.length_spread} "
                                   f"radius_scale={a.radius_scale}")
    np.savez(a.out, **g)
    st = np.concatenate(straight_after) if straight_after else np.zeros(1)
    print(json.dumps({"strands": int(len(idx)),
                      "straightness_after_p10_50_90": np.round(np.percentile(st, [10, 50, 90]), 3).tolist()}))


if __name__ == "__main__":
    main()
