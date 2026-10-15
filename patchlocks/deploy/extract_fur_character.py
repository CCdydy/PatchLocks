"""Extract interpretable fur CHARACTER priors from diffusion-predicted 3D patches.

New mainline (replaces stamping raw 3D / consuming entangled z64):
  2D crop -> diffusion predicts 3D patch -> EXTRACT decoupled geometric priors -> V1 grow + Step2 consume them.

The z64 latent is NOT disentangled, so we cannot "keep shape / drop direction+color" inside it.
Instead we decode each patch to 3D strands and MEASURE direction-agnostic scalars:
  - canonicalize: drop color (ignored), drop direction (rotate every strand so its own tangent -> +x;
    keep only the RELATIVE shape), so the measured features are orientation-free.
  - character per patch:
      arc_over_psize, lift_over_arc, radius_over_arc, density_over_area,
      straightness, curl, tip_spread, clump (within-patch alignment), conformance (lie-flatness)
  - footprint_world (2D->3D scale): psize * crop_px / native_px (the 3D surface width the crop covers),
    so arc_world = arc_norm * footprint_world and radius scales by the SAME factor (no felt).
  - reliability: scale_conf * texture_conf (occlusion / oblique / dark -> low; not trusted).

Then aggregate RELIABLE patches per region (body/ear/tail/leg...) and propagate that region character
to every patch.  V1 grow synthesizes the coat parametrically from these priors; Step2 owns direction.

Output: character_prior.npz with per-patch character + per-region table + footprint_world + reliability.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, "patchlocks")
from models.strand_codec import StrandCodec


def load_codec(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cnorm = {k: torch.tensor(v, device=device, dtype=torch.float32) for k, v in ck["fur_norm"].items()}
    codec = StrandCodec(do_vae=False, decode_type="dir", scale_init=30.0, nr_verts_per_strand=256,
                        nr_values_to_decode=255, dim_per_value_decoded=3).to(device).eval()
    codec.load_state_dict(ck["state_dict"])
    for a in ("t", "start_positions"):
        if hasattr(codec.decoder, a):
            setattr(codec.decoder, a, getattr(codec.decoder, a).to(device))
    return codec, cnorm


def canonicalize(strands):
    """Root-center, then rotate EACH strand in-plane so its tip tangent points +x. Drops direction."""
    S = strands - strands[:, :1]
    ang = np.arctan2(S[:, -1, 1], S[:, -1, 0])
    c, s = np.cos(-ang), np.sin(-ang)
    x = S[..., 0].copy(); y = S[..., 1].copy()
    S[..., 0] = c[:, None] * x - s[:, None] * y
    S[..., 1] = s[:, None] * x + c[:, None] * y
    return S


def patch_character(S):
    """Direction-agnostic scalar character from canonicalized strands S (N, P, 3)."""
    seg = np.diff(S, axis=1)
    arc = np.linalg.norm(seg, axis=2).sum(1)                                   # per-strand arclen
    chord = np.linalg.norm(S[:, -1] - S[:, 0], axis=1)
    straight = chord / np.maximum(arc, 1e-9)
    liftr = np.abs(S[..., 2]).max(1) / np.maximum(arc, 1e-9)                   # standoff / length
    # curl: mean lateral (y) deviation from the +x chord, normalized by arc
    curl = np.abs(S[..., 1]).mean(1) / np.maximum(arc, 1e-9)
    # tip spread: how much canonicalized tips fan out (low => clumped/aligned)
    tip = S[:, -1, :2]
    tip_spread = float(np.sqrt(((tip - tip.mean(0)) ** 2).sum(1).mean())) / max(float(np.median(arc)), 1e-9)
    return dict(
        arc=float(np.median(arc)),
        straightness=float(np.median(straight)),
        lift_over_arc=float(np.median(liftr)),
        curl=float(np.median(curl)),
        tip_spread=tip_spread,
    )


def patch_clump_raw(strands):
    """Within-patch directional coherence before canonicalization."""
    S = strands - strands[:, :1]
    tip = S[:, -1, :2]
    n = np.linalg.norm(tip, axis=1)
    good = n > 1e-9
    if good.sum() < 4:
        return 0.0
    v = tip[good] / n[good, None]
    return float(np.linalg.norm(v.mean(axis=0)))


def smooth_scalar_field(values, centers, reliability, k=64, iters=3, floor=0.03):
    """Heavy local smoothing on the patch graph without requiring part labels."""
    values = np.asarray(values, np.float32)
    centers = np.asarray(centers, np.float32)
    reliability = np.asarray(reliability, np.float32)
    P = len(values)
    ok = np.isfinite(values)
    if P == 0:
        return values.copy()
    if ok.any():
        fill = float(np.nanmedian(values[ok]))
    else:
        fill = 0.0
    cur = np.where(ok, values, fill).astype(np.float32)
    base_w = np.where(ok, np.maximum(np.clip(reliability, 0.0, 1.0), float(floor)), 0.0).astype(np.float32)
    if not np.any(base_w > 0):
        base_w[:] = 1.0
    k = int(np.clip(k, 1, P))
    d2 = ((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    local_d2 = np.take_along_axis(d2, idx, axis=1)
    sigma2 = np.maximum(np.median(local_d2[:, -1]), 1e-10)
    spatial_w = np.exp(-local_d2 / max(float(sigma2), 1e-10)).astype(np.float32)
    for _ in range(max(int(iters), 0)):
        ww = spatial_w * base_w[idx]
        den = ww.sum(axis=1)
        sm = (ww * cur[idx]).sum(axis=1) / np.maximum(den, 1e-8)
        cur = np.where(den > 1e-8, sm, cur).astype(np.float32)
    return cur.astype(np.float32)


def robust_ref(values, reliability, fallback=1.0):
    values = np.asarray(values, np.float32)
    reliability = np.asarray(reliability, np.float32)
    ok = np.isfinite(values)
    if not ok.any():
        return float(fallback)
    thr = np.nanpercentile(reliability[ok], 60.0) if ok.sum() > 4 else -np.inf
    sel = ok & (reliability >= thr)
    if not sel.any():
        sel = ok
    ref = float(np.nanmedian(values[sel]))
    return ref if np.isfinite(ref) and abs(ref) > 1e-9 else float(fallback)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True, help="v10_surfpack_proposals.npz WITH patch_z64_map (run infer --save_maps)")
    ap.add_argument("--pack", required=True, help="surface_patch_dataset.npz")
    ap.add_argument("--geopatch", default="", help="geodesic_patchpack.npz for bbox_diag (optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--codec_path", default="patchlocks/models/fur_strand_codec_v6upright.pt")
    ap.add_argument("--reliable_pct", type=float, default=60.0, help="reliability percentile; patches at/above are trusted for region character")
    ap.add_argument("--max_strands", type=int, default=400, help="subsample texels per patch for speed")
    ap.add_argument("--field_k", type=int, default=64, help="neighbors for label-free smoothed character field")
    ap.add_argument("--field_iters", type=int, default=3, help="smoothing iterations for character field")
    ap.add_argument("--field_reliability_floor", type=float, default=0.03, help="small floor so low-confidence areas inherit nearby character")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(a.proposals, allow_pickle=True)
    if "patch_z64_map" not in d.files:
        raise KeyError("proposals must contain patch_z64_map; re-run infer with --save_maps")
    sp = np.load(a.pack, allow_pickle=True)
    zmap = d["patch_z64_map"]
    valid = sp["valid"] > 0.5
    P = zmap.shape[0]
    psize = float(sp["psize"]) if "psize" in sp.files else 1.0
    bbox = 1.0
    if a.geopatch and os.path.exists(a.geopatch):
        bbox = float(np.load(a.geopatch, allow_pickle=True)["bbox_diag"])

    region = d["patch_region"].astype(str) if "patch_region" in d.files else np.full(P, "body")
    native = d["patch_native_px"].astype(np.float32) if "patch_native_px" in d.files else np.full(P, 1.0, np.float32)
    crop = d["patch_crop_px"].astype(np.float32) if "patch_crop_px" in d.files else np.full(P, 1.0, np.float32)
    dens = d["patch_density_mean"].astype(np.float32) if "patch_density_mean" in d.files else np.full(P, 1.0, np.float32)
    radm = d["patch_radius_mean"].astype(np.float32) if "patch_radius_mean" in d.files else np.zeros(P, np.float32)
    # reliability DOMINATED by directional orderliness: empirically the diffusion predicts ORDERLY
    # (uni-directional) fur well and SWIRLING fur (face/eye whorls) badly (curly/messy). So dir_coh is
    # the primary quality signal; texture clarity gates it (dark/blurry -> unreliable too).
    tex = d["patch_texture_conf"].astype(np.float32) if "patch_texture_conf" in d.files else np.ones(P, np.float32)
    coh = d["patch_dir_coh"].astype(np.float32) if "patch_dir_coh" in d.files else np.ones(P, np.float32)
    cohn = (coh - np.nanmin(coh)) / max(float(np.nanmax(coh) - np.nanmin(coh)), 1e-6)   # normalize orderliness 0..1
    reliability = (cohn * np.clip(tex, 0.0, 1.0)).astype(np.float32)                    # orderliness * clarity

    # 2D->3D scale: footprint_world (the real surface width this crop covers).
    # Patches with a tiny 2D projection (native_px ~ 0: occluded / extreme oblique) give a degenerate
    # blow-up, so FLOOR native_px, CLAMP footprint to a sane band, and DOWNWEIGHT their reliability.
    NATIVE_FLOOR = 10.0
    footprint_world = psize * crop / np.maximum(native, NATIVE_FLOOR)
    footprint_world = np.clip(footprint_world, 0.25 * psize, 6.0 * psize).astype(np.float32)
    reliability = (reliability * np.clip(native / 30.0, 0.0, 1.0)).astype(np.float32)   # tiny projection -> untrusted

    codec, cnorm = load_codec(a.codec_path, device)
    feats = {k: np.full(P, np.nan, np.float32) for k in ["arc", "straightness", "lift_over_arc", "curl", "tip_spread", "clump"]}
    for p in range(P):
        ys, xs = np.where(valid[p])
        if len(ys) < 16:
            continue
        if len(ys) > a.max_strands:
            sub = np.linspace(0, len(ys) - 1, a.max_strands).astype(np.int64)
            ys, xs = ys[sub], xs[sub]
        with torch.no_grad():
            pr = codec.decoder(torch.tensor(zmap[p][ys, xs].astype(np.float32), device=device), None, cnorm)
            St = (pr["strand_positions"] if "strand_positions" in pr else pr[[k for k in pr if "pos" in k.lower()][0]]).cpu().numpy()
        idx = np.linspace(0, St.shape[1] - 1, 32).astype(int)
        raw = St[:, idx]
        S = canonicalize(St[:, idx])
        ch = patch_character(S)
        ch["clump"] = patch_clump_raw(raw)
        for k in feats:
            feats[k][p] = ch[k]

    # derived per-patch character (ratios are scale-free)
    arc = feats["arc"]
    arc_over_psize = arc / max(psize, 1e-9)
    radius_over_arc = np.where(arc > 1e-9, np.where(radm > 0, radm / np.maximum(arc, 1e-9), 0.0169), 0.0169).astype(np.float32)

    # per-region character from RELIABLE patches
    thr = np.nanpercentile(reliability, a.reliable_pct)
    regions = sorted(set(region.tolist()))
    region_char = {}
    print(f"P={P} psize={psize:.4f} bbox={bbox:.3f} reliable_thr(p{a.reliable_pct:.0f})={thr:.3f}", flush=True)
    print(f"{'region':10s} {'n_rel':>6s} {'arc/ps':>7s} {'straight':>8s} {'lift/arc':>8s} {'curl':>6s} {'tipspr':>7s} {'clump':>6s} {'foot_w':>7s} {'dens':>6s}")
    for rg in regions:
        m = (region == rg) & np.isfinite(feats["arc"]) & (reliability >= thr)
        if m.sum() < 1:
            m = (region == rg) & np.isfinite(feats["arc"])
        if m.sum() < 1:
            continue
        rc = dict(
            arc_over_psize=float(np.nanmedian(arc_over_psize[m])),
            straightness=float(np.nanmedian(feats["straightness"][m])),
            lift_over_arc=float(np.nanmedian(feats["lift_over_arc"][m])),
            curl=float(np.nanmedian(feats["curl"][m])),
            tip_spread=float(np.nanmedian(feats["tip_spread"][m])),
            clump=float(np.nanmedian(feats["clump"][m])),
            radius_over_arc=float(np.nanmedian(radius_over_arc[m])),
            footprint_world=float(np.nanmedian(footprint_world[m])),
            density=float(np.nanmedian(dens[m])),
            n_reliable=int(m.sum()),
        )
        region_char[rg] = rc
        print(f"{rg:10s} {rc['n_reliable']:6d} {rc['arc_over_psize']:7.3f} {rc['straightness']:8.3f} "
              f"{rc['lift_over_arc']:8.3f} {rc['curl']:6.3f} {rc['tip_spread']:7.3f} {rc['clump']:6.3f} {rc['footprint_world']:7.4f} {rc['density']:6.1f}")

    # propagate: every patch gets its region's character
    out = dict(
        patch_region=region, reliability=reliability, footprint_world=footprint_world.astype(np.float32),
        patch_arc_over_psize=arc_over_psize.astype(np.float32), patch_straightness=feats["straightness"],
        patch_lift_over_arc=feats["lift_over_arc"], patch_curl=feats["curl"], patch_tip_spread=feats["tip_spread"],
        patch_clump=feats["clump"], patch_radius_over_arc=radius_over_arc, psize=np.float32(psize), bbox_diag=np.float32(bbox),
        region_names=np.array(list(region_char.keys())),
    )
    centers = sp["centers"].astype(np.float32) if "centers" in sp.files else sp["surf"].mean(axis=(1, 2)).astype(np.float32)
    field_inputs = dict(
        arc_over_psize=arc_over_psize.astype(np.float32),
        straightness=feats["straightness"],
        lift_over_arc=feats["lift_over_arc"],
        curl=feats["curl"],
        tip_spread=feats["tip_spread"],
        clump=feats["clump"],
        radius_over_arc=radius_over_arc.astype(np.float32),
    )
    field = {}
    for k, v in field_inputs.items():
        field[k] = smooth_scalar_field(
            v, centers[:P], reliability, k=a.field_k, iters=a.field_iters,
            floor=a.field_reliability_floor)
    field_reliability = smooth_scalar_field(
        reliability, centers[:P], reliability, k=a.field_k, iters=max(1, a.field_iters),
        floor=a.field_reliability_floor)
    ref_arc = robust_ref(field["arc_over_psize"], reliability, fallback=float(np.nanmedian(arc_over_psize[np.isfinite(arc_over_psize)])) if np.isfinite(arc_over_psize).any() else 1.0)
    ref_lift = robust_ref(field["lift_over_arc"], reliability, fallback=float(np.nanmedian(feats["lift_over_arc"][np.isfinite(feats["lift_over_arc"])])) if np.isfinite(feats["lift_over_arc"]).any() else 1.0)
    patch_length_ratio_field = np.clip(field["arc_over_psize"] / max(ref_arc, 1e-8), 0.25, 4.0).astype(np.float32)
    patch_lift_scale_field = np.clip(field["lift_over_arc"] / max(ref_lift, 1e-8), 0.25, 2.5).astype(np.float32)
    print(
        f"character field: k={a.field_k} iters={a.field_iters} "
        f"len_ratio[{patch_length_ratio_field.min():.2f},{patch_length_ratio_field.max():.2f}] "
        f"lift_scale[{patch_lift_scale_field.min():.2f},{patch_lift_scale_field.max():.2f}] "
        f"conf[{field_reliability.min():.2f},{field_reliability.max():.2f}]",
        flush=True)

    for k in ["arc_over_psize", "straightness", "lift_over_arc", "curl", "tip_spread", "clump", "radius_over_arc", "footprint_world", "density"]:
        out[f"region_{k}"] = np.array([region_char[rg][k] for rg in region_char], np.float32)
    for k, v in field.items():
        out[f"character_field_{k}"] = v.astype(np.float32)
    out["character_field_reliability"] = field_reliability.astype(np.float32)
    out["patch_length_ratio_field"] = patch_length_ratio_field
    out["patch_lift_scale_field"] = patch_lift_scale_field
    out["patch_radius_over_arc_field"] = field["radius_over_arc"].astype(np.float32)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, **out)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
