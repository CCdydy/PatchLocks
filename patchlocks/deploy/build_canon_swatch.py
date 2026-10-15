"""Build a CANONICAL REPRESENTATIVE fur shape (swatch) from reliable diffusion-predicted 3D patches.

Approach A of the new mainline:
  reliable 2D patch -> diffusion 3D patch -> extract direction-free scalar character
  -> synthesize a clean V1-style representative primitive
  -> V1 grow distributes over the whole animal -> Step2 optimizes direction/length.

This builds the canonical representative shape:
  1. CANONICALIZE: for each reliable patch decode z64_map -> strands; root-center; rotate the patch's
     MAIN direction (mean tip tangent) to +x, KEEPING the within-patch relative variation (NOT per-strand
     flattening). Color dropped, absolute direction dropped. The shape now expresses only:
     straight/curl, lie-flat/standoff, length distribution, tip spread, clump.
  2. By default, measure length/radius/lift/curl from those strands and synthesize a clean V1-style
     short-hair primitive.  This avoids copying V10 decoded strand-path noise directly onto the dog.
     Pass --raw_swatch to keep the old literal decoded-strand pooling path for diagnosis.
  3. Emit in p1-swatch format (strands/root_uv/radius/color/size) so the existing V1 grow stamp path
     (`grow_v2_region_prior.py --strand_profile_npz`) consumes it -- with `--scale_contract
     patch_window_footprint` doing the 2D->3D footprint scale and Step2 owning direction.

Reliability = directional orderliness (dir_coh) * texture clarity: the diffusion predicts ORDERLY fur
well and SWIRLING fur (face/eye whorls) badly, so we only trust orderly+clear patches for the shape;
swirly regions inherit it by propagation and get their direction from Step2.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from models.strand_codec import StrandCodec


def _grid_uv(n, rng):
    """Stable jittered root layout over [0,1]^2 for a representative swatch."""
    side = int(np.ceil(np.sqrt(max(int(n), 1))))
    yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    uv = np.stack([xx.ravel(), yy.ravel()], 1).astype(np.float32)[:n]
    uv = (uv + rng.uniform(0.15, 0.85, uv.shape).astype(np.float32)) / float(side)
    return np.clip(uv, 0.0, 1.0)


def _stroke_offsets_from_stats(n, points, arc_med, curl_frac, lift_frac, radius_ratio, rng):
    """Generate clean canonical +x short-hair strokes from measured character stats.

    Units are the normalized swatch-local units consumed by grow_v2_region_prior;
    grow later multiplies by geodesic psize and the 2D->3D scale contract.
    """
    n = int(n)
    points = int(points)
    t = np.linspace(0.0, 1.0, points, dtype=np.float32)[None, :, None]

    arc_med = max(float(arc_med), 1e-6)
    curl_frac = max(float(curl_frac), 0.0)
    lift_frac = max(float(lift_frac), 0.0)
    radius_ratio = max(float(radius_ratio), 1e-6)

    # Keep the distribution, not the decoded path.  The variation is deliberately
    # low: whole-animal patch placement and Step2 provide macro diversity.
    arc_i = arc_med * np.exp(rng.normal(0.0, 0.055, size=n)).astype(np.float32)
    arc_i = np.clip(arc_i, 0.78 * arc_med, 1.22 * arc_med)
    curl_i = curl_frac * np.exp(rng.normal(0.0, 0.18, size=n)).astype(np.float32)
    lift_i = lift_frac * np.exp(rng.normal(0.0, 0.16, size=n)).astype(np.float32)

    # curl metric in extract_fur_character is mean(abs(y))/arc.  Since
    # mean(abs(sin(pi*t))) ~= 2/pi, convert measured curl to a sine amplitude.
    y_amp = (curl_i * arc_i / (2.0 / np.pi + 1e-6)).astype(np.float32)
    z_amp = (lift_i * arc_i).astype(np.float32)
    sign = rng.choice(np.array([-1.0, 1.0], np.float32), size=n).astype(np.float32)
    phase = rng.uniform(-0.18, 0.18, size=n).astype(np.float32)

    # Solve x chord with a cheap monotonic correction so median arclength stays
    # close to the measured character while lateral/normal variation remains mild.
    x_chord = (arc_i * 0.985).astype(np.float32)
    off = np.zeros((n, points, 3), np.float32)
    for _ in range(3):
        tt = t[:, :, 0]
        off[:, :, 0] = x_chord[:, None] * tt
        off[:, :, 1] = (sign * y_amp)[:, None] * np.sin(np.pi * np.clip(tt + phase[:, None] * tt * (1.0 - tt), 0.0, 1.0))
        off[:, :, 2] = z_amp[:, None] * (1.0 - np.cos(np.pi * tt)) * 0.5
        cur_arc = np.linalg.norm(np.diff(off, axis=1), axis=2).sum(1)
        x_chord *= np.clip(arc_i / np.maximum(cur_arc, 1e-9), 0.85, 1.08).astype(np.float32)

    radius = (radius_ratio * arc_i).astype(np.float32)
    return off.astype(np.float32), radius


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True, help="proposals npz WITH patch_z64_map")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--character", default="", help="character_prior.npz (for reliability); else recompute coh*tex")
    ap.add_argument("--out", required=True)
    ap.add_argument("--codec_path", default="patchlocks/models/fur_strand_codec_v6upright.pt")
    ap.add_argument("--reliable_pct", type=float, default=60.0)
    ap.add_argument("--region", default="", help="restrict source to this region (patch_region); empty = all")
    ap.add_argument("--min_straightness", type=float, default=0.0,
                    help="optional absolute character gate: keep only source patches whose extracted patch_straightness is at least this value. Prefer --min_straightness_pct for mainline runs.")
    ap.add_argument("--max_curl", type=float, default=1e9,
                    help="optional absolute character gate: keep only source patches whose extracted patch_curl is at most this value. Prefer --max_curl_pct for mainline runs.")
    ap.add_argument("--max_arc_over_psize", type=float, default=1e9,
                    help="optional absolute character gate: keep only SHORT source patches whose median arclength / geodesic patch size is at most this value. Prefer --max_arc_over_psize_pct for mainline runs.")
    ap.add_argument("--max_lift_over_arc", type=float, default=1e9,
                    help="optional absolute character gate: keep only LIE-FLAT source patches whose lift/arclength is at most this value. Prefer --max_lift_over_arc_pct for mainline runs.")
    ap.add_argument("--max_tip_spread", type=float, default=1e9,
                    help="optional absolute character gate: keep only compact source patches whose tip spread is at most this value. Prefer --max_tip_spread_pct for mainline runs.")
    ap.add_argument("--min_straightness_pct", type=float, default=-1.0,
                    help="data-driven gate over reliable source candidates: keep patch_straightness >= this percentile. <0 disables.")
    ap.add_argument("--max_curl_pct", type=float, default=-1.0,
                    help="data-driven gate over reliable source candidates: keep patch_curl <= this percentile. <0 disables.")
    ap.add_argument("--max_arc_over_psize_pct", type=float, default=-1.0,
                    help="data-driven gate over reliable source candidates: keep patch_arc_over_psize <= this percentile. <0 disables.")
    ap.add_argument("--max_lift_over_arc_pct", type=float, default=-1.0,
                    help="data-driven gate over reliable source candidates: keep patch_lift_over_arc <= this percentile. <0 disables.")
    ap.add_argument("--max_tip_spread_pct", type=float, default=-1.0,
                    help="data-driven gate over reliable source candidates: keep patch_tip_spread <= this percentile. <0 disables.")
    ap.add_argument("--per_patch", type=int, default=24, help="strands kept per reliable patch")
    ap.add_argument("--swatch_strands", type=int, default=20480)
    ap.add_argument("--points", type=int, default=32)
    ap.add_argument("--raw_swatch", action="store_true",
                    help="diagnostic legacy mode: pool literal decoded strand paths. Default is to synthesize clean V1-style strokes from measured character stats.")
    ap.add_argument("--coherent_swatch", action="store_true",
                    help="pool decoded strand paths but canonicalize EACH strand's own direction to +x, "
                         "keeping v16's real per-strand bend/length/lift variation while removing the "
                         "inter-strand direction noise (decode scatter). Direction is supplied coherently "
                         "by Step2's flow field. Bridges flat (synth) and messy (raw).")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(a.proposals, allow_pickle=True)
    if "patch_z64_map" not in d.files:
        raise KeyError("proposals need patch_z64_map (infer --save_maps)")
    sp = np.load(a.pack, allow_pickle=True)
    zmap = d["patch_z64_map"]
    valid = sp["valid"] > 0.5
    P = zmap.shape[0]
    H = zmap.shape[1]

    if a.character and os.path.exists(a.character):
        c = np.load(a.character, allow_pickle=True)
        reliability = c["reliability"].astype(np.float32)
        ch_straight = c["patch_straightness"].astype(np.float32) if "patch_straightness" in c.files else None
        ch_curl = c["patch_curl"].astype(np.float32) if "patch_curl" in c.files else None
        ch_arc = c["patch_arc_over_psize"].astype(np.float32) if "patch_arc_over_psize" in c.files else None
        ch_lift = c["patch_lift_over_arc"].astype(np.float32) if "patch_lift_over_arc" in c.files else None
        ch_spread = c["patch_tip_spread"].astype(np.float32) if "patch_tip_spread" in c.files else None
    else:
        coh = d["patch_dir_coh"].astype(np.float32) if "patch_dir_coh" in d.files else np.ones(P, np.float32)
        tex = d["patch_texture_conf"].astype(np.float32) if "patch_texture_conf" in d.files else np.ones(P, np.float32)
        cohn = (coh - coh.min()) / max(float(coh.max() - coh.min()), 1e-6)
        reliability = cohn * np.clip(tex, 0, 1)
        ch_straight = None
        ch_curl = None
        ch_arc = None
        ch_lift = None
        ch_spread = None
    region = d["patch_region"].astype(str) if "patch_region" in d.files else np.full(P, "body")
    thr = float(np.nanpercentile(reliability, a.reliable_pct))
    pick = reliability >= thr
    if a.region:
        pick &= (region == a.region)
    candidate = pick.copy()
    gate_desc = []

    def pct_gate(arr, pct, name, keep_high):
        nonlocal pick
        if arr is None or float(pct) < 0:
            return
        base = candidate & np.isfinite(arr)
        if not base.any():
            return
        q = float(np.nanpercentile(arr[base], float(pct)))
        if keep_high:
            pick &= np.isfinite(arr) & (arr >= q)
            gate_desc.append(f"{name}>=p{float(pct):.0f}({q:.3f})")
        else:
            pick &= np.isfinite(arr) & (arr <= q)
            gate_desc.append(f"{name}<=p{float(pct):.0f}({q:.3f})")

    # Mainline gates are percentile-based inside the reliable source pool.  This
    # makes the character extraction portable across short/long/wiry/curly
    # animals without hard-coding beagle-specific scalar thresholds.
    pct_gate(ch_straight, a.min_straightness_pct, "straight", keep_high=True)
    pct_gate(ch_curl, a.max_curl_pct, "curl", keep_high=False)
    pct_gate(ch_arc, a.max_arc_over_psize_pct, "arc/psize", keep_high=False)
    pct_gate(ch_lift, a.max_lift_over_arc_pct, "lift/arc", keep_high=False)
    pct_gate(ch_spread, a.max_tip_spread_pct, "tip_spread", keep_high=False)

    if ch_straight is not None and float(a.min_straightness) > 0:
        pick &= np.isfinite(ch_straight) & (ch_straight >= float(a.min_straightness))
        gate_desc.append(f"straight>={float(a.min_straightness):.3f}")
    if ch_curl is not None and float(a.max_curl) < 1e8:
        pick &= np.isfinite(ch_curl) & (ch_curl <= float(a.max_curl))
        gate_desc.append(f"curl<={float(a.max_curl):.3f}")
    if ch_arc is not None and float(a.max_arc_over_psize) < 1e8:
        pick &= np.isfinite(ch_arc) & (ch_arc <= float(a.max_arc_over_psize))
        gate_desc.append(f"arc/psize<={float(a.max_arc_over_psize):.3f}")
    if ch_lift is not None and float(a.max_lift_over_arc) < 1e8:
        pick &= np.isfinite(ch_lift) & (ch_lift <= float(a.max_lift_over_arc))
        gate_desc.append(f"lift/arc<={float(a.max_lift_over_arc):.3f}")
    if ch_spread is not None and float(a.max_tip_spread) < 1e8:
        pick &= np.isfinite(ch_spread) & (ch_spread <= float(a.max_tip_spread))
        gate_desc.append(f"tip_spread<={float(a.max_tip_spread):.3f}")
    pids = np.where(pick)[0]
    if len(pids) == 0:
        # Percentile gates are portable, but their intersection can be empty on
        # animals whose reliable patches trade off length/lift/straightness
        # differently (panda is the common case).  Fall back to a pure
        # data-driven rank inside the reliable pool instead of hard failing.
        base = np.where(candidate)[0]
        if len(base) == 0:
            raise RuntimeError("no reliable source patches before character gates; lower --reliable_pct")

        def _rank01(arr, high=True):
            if arr is None:
                return np.zeros(len(base), np.float32)
            v = arr[base].astype(np.float32)
            ok = np.isfinite(v)
            out = np.zeros(len(base), np.float32)
            if ok.sum() <= 1:
                return out
            vv = v[ok]
            order = np.argsort(vv)
            r = np.empty_like(order, dtype=np.float32)
            r[order] = np.linspace(0.0, 1.0, len(vv), dtype=np.float32)
            out[ok] = r if high else (1.0 - r)
            return out

        score = np.zeros(len(base), np.float32)
        score += _rank01(reliability, high=True)
        score += _rank01(ch_straight, high=True)
        score += _rank01(ch_curl, high=False)
        score += _rank01(ch_arc, high=False)
        score += _rank01(ch_lift, high=False)
        score += _rank01(ch_spread, high=False)
        n_fallback = int(np.clip(round(0.05 * len(base)), 6, 32))
        pids = base[np.argsort(-score)[:n_fallback]]
        gate_desc.append(f"fallback_rank_top{len(pids)}/{len(base)}")

    print(f"P={P} reliable_thr(p{a.reliable_pct:.0f})={thr:.3f} -> {len(pids)} source patches"
          + (f" (region={a.region})" if a.region else "")
          + ((" gates=" + ",".join(gate_desc)) if gate_desc else ""), flush=True)

    ck = torch.load(a.codec_path, map_location=device, weights_only=False)
    cnorm = {k: torch.tensor(v, device=device, dtype=torch.float32) for k, v in ck["fur_norm"].items()}
    codec = StrandCodec(do_vae=False, decode_type="dir", scale_init=30.0, nr_verts_per_strand=256,
                        nr_values_to_decode=255, dim_per_value_decoded=3).to(device).eval()
    codec.load_state_dict(ck["state_dict"])
    for at in ("t", "start_positions"):
        if hasattr(codec.decoder, at):
            setattr(codec.decoder, at, getattr(codec.decoder, at).to(device))

    rng = np.random.default_rng(0)
    sw_off, sw_uv = [], []
    for p in pids:
        ys, xs = np.where(valid[p])
        if len(ys) < a.per_patch:
            continue
        sub = rng.choice(len(ys), size=a.per_patch, replace=False)
        ys, xs = ys[sub], xs[sub]
        with torch.no_grad():
            pr = codec.decoder(torch.tensor(zmap[p][ys, xs].astype(np.float32), device=device), None, cnorm)
            St = (pr["strand_positions"] if "strand_positions" in pr else pr[[k for k in pr if "pos" in k.lower()][0]]).cpu().numpy()
        idx = np.linspace(0, St.shape[1] - 1, a.points).astype(int)
        S = St[:, idx] - St[:, idx][:, :1]                                  # root-center
        # patch MAIN-direction canon: rotate by -mean(tip angle), keep within-patch relative variation
        ang = np.arctan2(S[:, -1, 1], S[:, -1, 0])
        ma = float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))
        c_, s_ = np.cos(-ma), np.sin(-ma)
        x, y = S[..., 0].copy(), S[..., 1].copy()
        S[..., 0] = c_ * x - s_ * y
        S[..., 1] = s_ * x + c_ * y
        sw_off.append(S.astype(np.float32))
        sw_uv.append(np.stack([xs / (H - 1), ys / (H - 1)], 1).astype(np.float32))

    raw_off = np.concatenate(sw_off, 0)
    raw_uv = np.concatenate(sw_uv, 0)
    raw_arc = np.linalg.norm(np.diff(raw_off, axis=1), axis=2).sum(1)
    raw_chord = np.linalg.norm(raw_off[:, -1] - raw_off[:, 0], axis=1)
    raw_curl = np.abs(raw_off[..., 1]).mean(1) / np.maximum(raw_arc, 1e-9)
    raw_lift = np.abs(raw_off[..., 2]).max(1) / np.maximum(raw_arc, 1e-9)

    if a.coherent_swatch:
        # Keep v16's real per-strand bend/length/lift, but rotate EACH strand about z so
        # its own dominant direction points +x.  This strips the inter-strand direction
        # scatter (decode noise, ~47 deg even within one patch) that makes the raw pool
        # read as messy, while preserving the shape variation that synth flattens away.
        # Direction itself is owned by Step2's flow field, not the swatch.
        off = raw_off.copy()
        w = np.linspace(0.0, 1.0, off.shape[1], dtype=np.float32)[None, :]      # tip-weighted dir
        dx = (off[..., 0] * w).sum(1)
        dy = (off[..., 1] * w).sum(1)
        sa = np.arctan2(dy, dx)
        c_, s_ = np.cos(-sa), np.sin(-sa)
        x, y = off[..., 0].copy(), off[..., 1].copy()
        off[..., 0] = c_[:, None] * x - s_[:, None] * y
        off[..., 1] = s_[:, None] * x + c_[:, None] * y
        uv = raw_uv
        radius_ratio = 0.0169
        mode = "coherent_decoded_paths"
        if len(off) > a.swatch_strands:
            keep = rng.choice(len(off), size=a.swatch_strands, replace=False)
            off, uv = off[keep], uv[keep]
        arc_for_radius = np.linalg.norm(np.diff(off, axis=1), axis=2).sum(1)
        radius = (radius_ratio * np.median(arc_for_radius) * np.ones(len(off))).astype(np.float32)
    elif a.raw_swatch:
        off = raw_off
        uv = raw_uv
        radius_ratio = 0.0169
        mode = "raw_decoded_paths"
        if len(off) > a.swatch_strands:
            keep = rng.choice(len(off), size=a.swatch_strands, replace=False)
            off, uv = off[keep], uv[keep]
        arc_for_radius = np.linalg.norm(np.diff(off, axis=1), axis=2).sum(1)
        radius = (radius_ratio * np.median(arc_for_radius) * np.ones(len(off))).astype(np.float32)
    else:
        arc_med = float(np.nanmedian(raw_arc))
        curl_med = float(np.nanmedian(raw_curl))
        lift_med = float(np.nanmedian(raw_lift))
        # Prefer direction-free character scalars when available.  raw_curl is
        # measured after only patch-main-direction canonicalization, so residual
        # within-patch direction variation can be misread as curl.  The
        # character prior canonicalizes each strand before measuring curl/lift.
        if ch_curl is not None:
            cc = ch_curl[pids]
            cc = cc[np.isfinite(cc)]
            if len(cc):
                curl_med = float(np.nanmedian(cc))
        if ch_lift is not None:
            ll = ch_lift[pids]
            ll = ll[np.isfinite(ll)]
            if len(ll):
                lift_med = float(np.nanmedian(ll))
        radius_ratio = 0.0169
        if a.character and os.path.exists(a.character):
            c = np.load(a.character, allow_pickle=True)
            if "patch_radius_over_arc" in c.files:
                rr = c["patch_radius_over_arc"].astype(np.float32)
                rr = rr[pids]
                rr = rr[np.isfinite(rr) & (rr > 0)]
                if len(rr):
                    radius_ratio = float(np.nanmedian(rr))
        off, radius = _stroke_offsets_from_stats(
            a.swatch_strands,
            a.points,
            arc_med,
            curl_med,
            lift_med,
            radius_ratio,
            rng,
        )
        uv = _grid_uv(len(off), rng)
        mode = "synthesized_v1_strokes"

    # planar swatch (p1 format): root on z=0 plane at (u-0.5, v-0.5), strand = root + canonical offset
    rootp = np.stack([uv[:, 0] - 0.5, uv[:, 1] - 0.5, np.zeros(len(uv), np.float32)], 1)
    strands = (rootp[:, None] + off).astype(np.float32)
    arc = np.linalg.norm(np.diff(strands, axis=1), axis=2).sum(1)
    color = np.full((len(strands), 3), 0.55, np.float32)
    chord = np.linalg.norm(strands[:, -1] - strands[:, 0], axis=1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, strands=strands, root_uv=uv.astype(np.float32), radius=radius,
             color=color, color_tip=np.full((len(strands), 3), 0.62, np.float32), size=np.float32(1.0),
             source_mode=np.asarray(mode),
             source_patch_ids=pids.astype(np.int32),
             measured_arc_med=np.float32(np.nanmedian(raw_arc)),
             measured_straight_med=np.float32(np.nanmedian(raw_chord / np.maximum(raw_arc, 1e-9))),
             measured_curl_med=np.float32(np.nanmedian(raw_curl)),
             measured_lift_over_arc_med=np.float32(np.nanmedian(raw_lift)),
             radius_arc_ratio=np.float32(float(np.median(radius) / max(float(np.median(arc)), 1e-9))))
    print(f"canon swatch[{mode}]: {len(strands)} strands from {len(pids)} reliable patches, "
          f"measured arc {np.nanmedian(raw_arc):.4f}, measured straight {np.nanmedian(raw_chord/np.maximum(raw_arc,1e-9)):.3f}, "
          f"out arc {np.median(arc):.4f}, out straight {np.median(chord/np.maximum(arc,1e-9)):.3f}, "
          f"curl {np.median(np.abs((strands-strands[:,:1])[...,1]).mean(1)/np.maximum(arc,1e-9)):.3f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
