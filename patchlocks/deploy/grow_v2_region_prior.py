#!/usr/bin/env python3
"""Grow a drivable fur surfpack from a V2 region prior.

This is the first visual consumer for the V2 contract:

  v2_region_prior.npz
    final_density          -> root count, with PoU sharing across overlapping patches
    patch_z64_map          -> local strand texture (preferred)
    patch_z64              -> local strand mean (legacy fallback)
    patch_dir/region_dir   -> external axial groom direction, measured or diffusion-predicted
    final_color_root_tip   -> region-calibrated color
    patch_face/bary        -> surface anchors

Output is the standard surfpack_v6fur.npz:
  strands, color, color_tip, radius, root_face, root_bary
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "patchlocks/data_gen")

import numpy as np
import torch

from field_to_strands import _bilinear
from models.strand_codec import StrandCodec


def unit(v, eps=1e-9):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def axial_delta(a, b):
    """Smallest unoriented angle from b to a, returned in [-pi/2, pi/2]."""
    return 0.5 * np.arctan2(np.sin(2.0 * (a - b)), np.cos(2.0 * (a - b)))


def load_obj_tri_mesh(path):
    verts = []
    faces = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = []
                for tok in line.strip().split()[1:]:
                    # OBJ face tokens can be v, v/vt, v//vn, or v/vt/vn.
                    vi = int(tok.split("/")[0])
                    if vi < 0:
                        vi = len(verts) + vi + 1
                    idx.append(vi - 1)
                if len(idx) >= 3:
                    for k in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[k], idx[k + 1]])
    if not verts or not faces:
        raise ValueError(f"failed to load triangular mesh from {path}")
    return np.asarray(verts, np.float32), np.asarray(faces, np.int32)


def nearest_rows_for_points(points, centers, row_ids):
    try:
        from scipy.spatial import cKDTree

        nn = cKDTree(centers).query(points, k=1)[1]
        return row_ids[nn]
    except Exception:
        # Small fallback: 160k faces x ~2k patches is acceptable in chunks.
        out = np.empty(len(points), np.int64)
        c2 = (centers * centers).sum(1)
        for s in range(0, len(points), 4096):
            q = points[s:s + 4096]
            d2 = (q * q).sum(1, keepdims=True) + c2[None] - 2.0 * q @ centers.T
            out[s:s + len(q)] = row_ids[np.argmin(d2, axis=1)]
        return out


def sample_face_roots(verts, faces, face_ids, rng):
    tri = faces[face_ids]
    r1 = np.sqrt(rng.uniform(0, 1, len(face_ids))).astype(np.float32)
    r2 = rng.uniform(0, 1, len(face_ids)).astype(np.float32)
    bary = np.stack([1.0 - r1, r1 * (1.0 - r2), r1 * r2], 1).astype(np.float32)
    pts = (verts[tri] * bary[:, :, None]).sum(1).astype(np.float32)
    return pts, bary


def infer_pack(prior, pack_arg):
    if pack_arg:
        return pack_arg
    if "source_pack" in prior.files:
        p = str(prior["source_pack"])
        if p and os.path.exists(p):
            return p
    raise ValueError("--pack is required because prior has no valid source_pack")


def infer_mesh_path(pack):
    if "source_pack" in pack.files:
        sp = str(pack["source_pack"])
        if os.path.exists(sp):
            src = np.load(sp, allow_pickle=True)
            if "mesh_path" in src.files:
                return str(src["mesh_path"])
    return ""


def patch_rows(prior, pack_count, max_patches):
    pids = prior["patch_id"].astype(str) if "patch_id" in prior.files else np.arange(len(prior["final_density"])).astype(str)
    rows = []
    for i, s in enumerate(pids):
        try:
            p = int(s)
        except ValueError:
            p = i
        if 0 <= p < pack_count:
            rows.append((i, p))
    if max_patches > 0:
        rows = rows[:max_patches]
    return rows


def region_lookup(prior):
    names = [str(x) for x in prior["region_names"]] if "region_names" in prior.files else []
    return {n: i for i, n in enumerate(names)}


def direction_for_patch(prior, row, region_idx, coh_gate):
    v = np.zeros(2, np.float32)
    c = 0.0
    if "patch_dir" in prior.files and "patch_dir_coh" in prior.files and row < len(prior["patch_dir"]):
        cand = prior["patch_dir"][row].astype(np.float32)
        cn = float(np.linalg.norm(cand))
        c = float(prior["patch_dir_coh"][row])
        if c >= coh_gate and cn > 1e-6:
            v = cand / cn
    if np.linalg.norm(v) <= 1e-6 and "region_dir" in prior.files:
        if 0 <= region_idx < len(prior["region_dir"]):
            cand = prior["region_dir"][region_idx].astype(np.float32)
            cn = float(np.linalg.norm(cand))
            if cn > 1e-6:
                v = cand / cn
    if np.linalg.norm(v) <= 1e-6:
        return 0.0
    # Axial field [cos(2 theta), sin(2 theta)].
    return float(0.5 * np.arctan2(v[1], v[0]))


def decode_z64(codec, cnorm, z_np, chunk, points, device):
    outs = []
    with torch.no_grad():
        for s0 in range(0, len(z_np), chunk):
            z = torch.tensor(z_np[s0:s0 + chunk], device=device, dtype=torch.float32)
            pr = codec.decoder(z, None, cnorm)
            rec = pr["strand_positions"] if "strand_positions" in pr else pr[[k for k in pr if "pos" in k.lower()][0]]
            rec = rec.detach().cpu().numpy().astype(np.float32)
            rec = rec - rec[:, :1]
            if rec.shape[1] != points:
                idx = np.linspace(0, rec.shape[1] - 1, points).round().astype(np.int64)
                rec = rec[:, idx]
            outs.append(rec)
    return np.concatenate(outs, 0) if outs else np.zeros((0, points, 3), np.float32)


def median_by_row(values, rows, row_count, fallback):
    out = np.full(row_count, float(fallback), np.float32)
    for r in np.unique(rows):
        m = rows == r
        if m.any():
            out[int(r)] = float(np.median(values[m]))
    return out


def region_median_rows(values, patch_region, row_ids):
    out = values.copy()
    regs = patch_region[row_ids].astype(str)
    for rg in np.unique(regs):
        rr = row_ids[regs == rg]
        if len(rr):
            out[rr] = float(np.median(values[rr]))
    return out


def smooth_rows_knn(values, centers, row_ids, k, iters):
    if iters <= 0 or len(row_ids) <= 1:
        return values
    vv = values[row_ids].astype(np.float32).copy()
    kk = max(1, min(int(k), len(row_ids)))
    try:
        from scipy.spatial import cKDTree

        nn = cKDTree(centers).query(centers, k=kk)[1]
    except Exception:
        c2 = (centers * centers).sum(1)
        d2 = c2[:, None] + c2[None] - 2.0 * centers @ centers.T
        nn = np.argsort(d2, axis=1)[:, :kk]
    if kk == 1:
        nn = nn[:, None]
    for _ in range(int(iters)):
        vv = 0.5 * vv + 0.5 * np.median(vv[nn], axis=1)
    out = values.copy()
    out[row_ids] = vv
    return out


def build_length_ratios(arc, root_row, row_count, patch_region, row_ids, centers, mode, smooth_k, smooth_iters, region_blend, ratio_clip):
    global_med = float(np.median(arc)) if len(arc) else 1.0
    global_med = max(global_med, 1e-9)
    patch_len = median_by_row(arc, root_row, row_count, global_med)
    raw_ratio = np.clip(patch_len / global_med, 1.0 / max(float(ratio_clip), 1.0), max(float(ratio_clip), 1.0)).astype(np.float32)

    if mode == "raw":
        desired = raw_ratio.copy()
    elif mode == "region":
        desired = region_median_rows(raw_ratio, patch_region, row_ids)
    elif mode == "patch_smooth":
        desired = smooth_rows_knn(raw_ratio, centers, row_ids, smooth_k, smooth_iters)
    elif mode == "region_smooth":
        reg = region_median_rows(raw_ratio, patch_region, row_ids)
        rb = float(np.clip(region_blend, 0.0, 1.0))
        mixed = rb * reg + (1.0 - rb) * raw_ratio
        desired = smooth_rows_knn(mixed, centers, row_ids, smooth_k, smooth_iters)
    else:
        raise ValueError(f"unknown length mode {mode}")

    desired = np.clip(desired, 1.0 / max(float(ratio_clip), 1.0), max(float(ratio_clip), 1.0)).astype(np.float32)
    root_med = float(np.median(desired[root_row])) if len(root_row) else 1.0
    if root_med > 1e-9:
        desired = desired / root_med
    return raw_ratio.astype(np.float32), desired.astype(np.float32)


def smooth_region_flow_delta(theta, patch_region, centers, args):
    """Damp noisy Step2 flow residuals on thin/small regions before grow.

    Step2 image-flow is useful on the body, but ears and tails are small, thin,
    and often partially occluded. Their per-patch flow residual can have high
    variance and produce visible spikes when consumed by the 3D grow. This
    generic policy keeps the learned character fixed and only stabilizes how
    the optimized direction residual is consumed in those regions.
    """
    if theta is None or len(theta) == 0 or str(args.region_flow_policy) == "none":
        return theta, "none"
    if str(args.region_flow_policy) != "generic":
        raise ValueError(f"unknown region flow policy: {args.region_flow_policy}")

    out = theta.astype(np.float32).copy()
    reg = patch_region.astype(str)
    details = []

    def axial_mean(vals):
        if len(vals) == 0:
            return 0.0
        return float(0.5 * np.arctan2(np.mean(np.sin(2.0 * vals)), np.mean(np.cos(2.0 * vals))))

    def smooth_rows(mask, strength, iters, label):
        idx = np.flatnonzero(mask)
        if len(idx) == 0 or float(strength) <= 0.0:
            return
        before = float(np.std(axial_delta(out[idx], axial_mean(out[idx]))))
        kk = max(1, min(int(args.region_flow_smooth_k), len(idx)))
        c = centers[idx].astype(np.float32)
        try:
            from scipy.spatial import cKDTree
            nn = cKDTree(c).query(c, k=kk)[1]
        except Exception:
            c2 = (c * c).sum(1)
            d2 = c2[:, None] + c2[None] - 2.0 * c @ c.T
            nn = np.argsort(d2, axis=1)[:, :kk]
        if kk == 1:
            nn = nn[:, None]
        for _ in range(max(int(iters), 0)):
            cur = out.copy()
            for j, row in enumerate(idx):
                mu = axial_mean(cur[idx[nn[j]]])
                out[row] = cur[row] + float(strength) * axial_delta(mu, cur[row])
        after = float(np.std(axial_delta(out[idx], axial_mean(out[idx]))))
        details.append(f"{label}:smooth std {before:.3f}->{after:.3f}")

    def damp_to_region(mask, damp, label):
        idx = np.flatnonzero(mask)
        if len(idx) == 0 or float(damp) >= 1.0:
            return
        mu = axial_mean(out[idx])
        before = float(np.std(axial_delta(out[idx], mu)))
        out[idx] = mu + float(damp) * axial_delta(out[idx], mu)
        after = float(np.std(axial_delta(out[idx], mu)))
        details.append(f"{label}:damp={float(damp):.2f} std {before:.3f}->{after:.3f}")

    ear = np.isin(reg, ["ears", "ear"])
    paw = np.isin(reg, ["paws", "paw"])
    tail = reg == "tail"
    tail_base = reg == "tail_base"

    smooth_rows(ear, args.region_flow_ear_smooth, args.region_flow_iters, "ear")
    damp_to_region(ear, args.region_flow_ear_damp, "ear")

    # Tail base is a continuity anchor. Smooth base+tail together, then damp
    # distal tail residuals so the tail follows one groom field instead of
    # becoming an independent fuzzy ball.
    smooth_rows(tail | tail_base, args.region_flow_tail_smooth, args.region_flow_iters, "tail")
    damp_to_region(tail, args.region_flow_tail_damp, "tail")

    smooth_rows(paw, args.region_flow_paw_smooth, max(1, int(args.region_flow_iters) // 2), "paw")
    damp_to_region(paw, args.region_flow_paw_damp, "paw")

    return out.astype(np.float32), "; ".join(details) if details else "none"


def apply_region_detail_policy(
    strands,
    radius,
    root_uv,
    root_face,
    root_bary,
    root_patch,
    root_row,
    color,
    color_tip,
    patch_region,
    args,
):
    """Generic region-level cleanup after character grow.

    This is deliberately not an animal-specific tweak. The policy only uses
    semantic region names and is applied after the V1-style grow has produced
    a full coat. Body and tail_base are protected; risk regions are compacted
    with small, traceable root-level factors.
    """
    n = len(strands)
    scale = np.ones(n, np.float32)
    keep_prob = np.ones(n, np.float32)
    if str(args.region_detail_policy) == "none" or n == 0:
        summary = "none"
        return (
            strands,
            radius,
            root_uv,
            root_face,
            root_bary,
            root_patch,
            root_row,
            color,
            color_tip,
            scale,
            keep_prob,
            summary,
        )

    if str(args.region_detail_policy) != "generic":
        raise ValueError(f"unknown region detail policy: {args.region_detail_policy}")

    root_reg = np.full(n, "body", dtype=object)
    valid = (root_row >= 0) & (root_row < len(patch_region))
    root_reg[valid] = patch_region[root_row[valid]].astype(str)
    roots = strands[:, 0].astype(np.float32)
    arc0 = np.linalg.norm(np.diff(strands, axis=1), axis=2).sum(1).astype(np.float32)
    body = root_reg == "body"
    body_med = float(np.median(arc0[body])) if body.any() else float(np.median(arc0))
    body_med = max(body_med, 1e-9)

    details = []

    feather = float(getattr(args, "region_detail_feather", 0.0) or 0.0)

    def region_weight(mask, fea=None):
        """Soft membership in [0,1]: 1 deep inside `mask`, smoothstep across the
        region boundary over a band of half-width `fea` (world units; defaults to
        the global `feather`), 0 far outside. fea<=0 (or a degenerate all/none
        mask) reproduces the hard boolean mask exactly, so legacy runs are
        byte-for-byte unchanged. A per-region override lets the paw band be wider
        (kill the 'sock' length jump) without disturbing the tuned tail/ears."""
        fea = feather if fea is None else float(fea)
        if fea <= 0 or not mask.any() or not (~mask).any():
            return mask.astype(np.float32)
        from scipy.spatial import cKDTree
        d_out = cKDTree(roots[~mask]).query(roots, k=1)[0].astype(np.float32)  # dist to nearest non-region root
        d_in = cKDTree(roots[mask]).query(roots, k=1)[0].astype(np.float32)    # dist to nearest region root
        phi = np.where(mask, d_out, -d_in).astype(np.float32)                  # signed distance, + inside region
        x = np.clip((phi + fea) / (2.0 * fea), 0.0, 1.0)                       # 0 at -fea .. 1 at +fea
        return (x * x * (3.0 - 2.0 * x)).astype(np.float32)                    # smoothstep

    def cap_to_body(mask, cap, label, w=None):
        if not mask.any() or float(cap) <= 0:
            return
        med = float(np.median(arc0[mask]))
        limit = body_med * float(cap)
        if med > limit > 1e-9:
            s = limit / med
            ww = mask.astype(np.float32) if w is None else w
            scale[:] *= (1.0 - ww * (1.0 - np.float32(s))).astype(np.float32)   # lerp(1, s, ww): feathered cap
            details.append(f"{label}_body_cap:{med/body_med:.3f}->{float(cap):.3f} scale={s:.3f}")

    def compact_regions(names, s, kp, w=None):
        mask = np.isin(root_reg, list(names))
        if mask.any():
            ww = mask.astype(np.float32) if w is None else w
            scale[:] *= (1.0 - ww * (1.0 - float(s))).astype(np.float32)        # lerp(1, s, ww)
            keep_prob[:] *= (1.0 - ww * (1.0 - float(kp))).astype(np.float32)   # lerp(1, kp, ww)
            band = int((ww > 1e-3).sum()) - int(mask.sum())
            tag = f" feather={feather:.3f}(+{band} band)" if w is not None and feather > 0 else ""
            details.append(f"{'/'.join(names)}:{int(mask.sum())} scale={float(s):.3f} keep={float(kp):.3f}{tag}")

    ear_mask = np.isin(root_reg, ["ears", "ear"])
    leg_mask = np.isin(root_reg, ["legs", "leg"])
    paw_mask = np.isin(root_reg, ["paws", "paw"])
    face_mask = np.isin(root_reg, ["face"])
    leg_feather = float(getattr(args, "region_detail_leg_feather", 0.0) or 0.0)
    paw_feather = float(getattr(args, "region_detail_paw_feather", 0.0) or 0.0)
    w_ear = region_weight(ear_mask)
    w_leg = region_weight(leg_mask, leg_feather if leg_feather > 0 else None)
    w_paw = region_weight(paw_mask, paw_feather if paw_feather > 0 else None)
    w_face = region_weight(face_mask)
    cap_to_body(ear_mask, args.region_detail_ear_body_cap, "ears", w_ear)
    cap_to_body(leg_mask, args.region_detail_leg_body_cap, "legs", w_leg)
    cap_to_body(paw_mask, args.region_detail_paw_body_cap, "paws", w_paw)
    compact_regions(("ears", "ear"), args.region_detail_ear_scale, args.region_detail_ear_keep, w_ear)
    compact_regions(("legs", "leg"), args.region_detail_leg_scale, args.region_detail_leg_keep, w_leg)
    compact_regions(("paws", "paw"), args.region_detail_paw_scale, args.region_detail_paw_keep, w_paw)
    cap_to_body(face_mask, args.region_detail_face_body_cap, "face", w_face)
    compact_regions(("face",), args.region_detail_face_scale, args.region_detail_face_keep, w_face)

    tail = root_reg == "tail"
    tail_base = root_reg == "tail_base"
    if tail.any():
        cap_to_body(tail, args.region_detail_tail_body_cap, "tail")
        # whole-tail thinning (feathered at the base) BEFORE the distal taper, so a
        # dense curled tail loses bulk everywhere instead of only at an ill-defined tip
        compact_regions(("tail",), args.region_detail_tail_uniform_scale,
                        args.region_detail_tail_uniform_keep, region_weight(tail))
        tail_roots = roots[tail]
        if tail_base.any():
            base_center = roots[tail_base].mean(0)
            tail_center = tail_roots.mean(0)
            axis = tail_center - base_center
        else:
            centered = tail_roots - tail_roots.mean(0, keepdims=True)
            _, _, vh = np.linalg.svd(centered.astype(np.float64), full_matrices=False)
            axis = vh[0].astype(np.float32)
        axis = unit(axis[None])[0]
        if str(getattr(args, "region_detail_tail_metric", "axis")) == "base_dist" and tail_base.any():
            # Outward Euclidean distance from the tail_base attachment. A curled
            # tail is a blob along any single axis, so axis-projection leaves most
            # of it at t~0; distance-from-base follows the curl and puts the true
            # (farthest) tip at t=1 so the taper actually reaches it.
            proj = np.linalg.norm(tail_roots - roots[tail_base].mean(0)[None], axis=1).astype(np.float32)
        else:
            proj = tail_roots @ axis
        q0 = float(np.percentile(proj, float(args.region_detail_tail_start_pct)))
        q1 = float(np.percentile(proj, float(args.region_detail_tail_end_pct)))
        if q1 <= q0 + 1e-9:
            t = np.ones(len(proj), np.float32)
        else:
            t = np.clip((proj - q0) / (q1 - q0), 0.0, 1.0).astype(np.float32)
        # Concentrate the taper toward the tip. pow>1 keeps the proximal tail at
        # ~body length (smooth junction with tail_base) and pulls length+density
        # down sharply only near the distal end -> the tip comes to a point
        # instead of rounding into a dense pom-pom. Density can taper even harder
        # (keep_pow) so the very tip thins out faster than it shortens.
        pw = max(float(getattr(args, "region_detail_tail_taper_pow", 1.0) or 1.0), 1e-3)
        kpw = max(float(getattr(args, "region_detail_tail_keep_pow", 1.0) or 1.0), 1e-3)
        t_len = np.power(t, pw).astype(np.float32)
        t_keep = np.power(t, pw * kpw).astype(np.float32)
        tip_keep = float(getattr(args, "region_detail_tail_tip_keep", -1.0))
        if tip_keep < 0:
            tip_keep = float(args.region_detail_tail_keep)
        tail_scale = 1.0 - (1.0 - float(args.region_detail_tail_scale)) * t_len
        tail_keep = 1.0 - (1.0 - tip_keep) * t_keep
        scale[tail] *= tail_scale.astype(np.float32)
        keep_prob[tail] *= tail_keep.astype(np.float32)
        details.append(
            "tail:"
            f"{int(tail.sum())} distal_mean={float(t.mean()):.3f} pow={pw:.2f}/{kpw:.2f} "
            f"scale_tip={float(np.min(tail_scale)):.3f} keep_tip={float(np.min(tail_keep)):.3f} "
            "tail_base_protected"
        )

    scale = np.clip(scale, 0.05, 4.0).astype(np.float32)
    keep_prob = np.clip(keep_prob, 0.05, 1.0).astype(np.float32)
    strands = (strands[:, :1] + (strands - strands[:, :1]) * scale[:, None, None]).astype(np.float32)
    radius = np.maximum(radius * scale, 1e-7).astype(np.float32)

    rng = np.random.default_rng(int(args.region_detail_seed))
    keep = rng.random(n) < keep_prob
    if not keep.any():
        keep[np.argmax(keep_prob)] = True
    dropped = int((~keep).sum())
    if dropped:
        strands = strands[keep]
        radius = radius[keep]
        root_uv = root_uv[keep]
        root_face = root_face[keep]
        root_bary = root_bary[keep]
        root_patch = root_patch[keep]
        root_row = root_row[keep]
        color = color[keep]
        color_tip = color_tip[keep]
        scale = scale[keep]
        keep_prob = keep_prob[keep]
    details.append(f"dropped={dropped}")
    summary = "; ".join(details) if details else "generic:no_matching_regions"
    return (
        strands,
        radius,
        root_uv,
        root_face,
        root_bary,
        root_patch,
        root_row,
        color,
        color_tip,
        scale.astype(np.float32),
        keep_prob.astype(np.float32),
        summary,
    )


def hash01_2d(ix, iy, salt=0):
    # Deterministic vectorized hash for stable world-space clump jitter.
    x = np.asarray(ix, np.float32) * 12.9898 + np.asarray(iy, np.float32) * 78.233 + float(salt) * 37.719
    return np.mod(np.sin(x) * 43758.5453, 1.0).astype(np.float32)


def apply_tail_length_policy(strands, radius, root_row, patch_region, args):
    """Make all tail-region strands share one final arclength.

    This is deliberately applied after generic region-detail cleanup.  If it ran
    earlier, distal tail tapering would immediately reintroduce piecewise length
    bands.  The operation is generic: by default it treats tail and tail_base as
    one continuous tail region.
    """
    policy = str(args.tail_length_policy)
    if policy == "none" or len(strands) == 0:
        return strands, radius, "none", 0.0, 0
    if policy != "uniform":
        raise ValueError(f"unknown tail length policy: {args.tail_length_policy}")

    reg = np.asarray(patch_region).astype(str)
    row = np.asarray(root_row, np.int64)
    root_reg = np.full(len(row), "", dtype=object)
    valid = (row >= 0) & (row < len(reg))
    root_reg[valid] = reg[row[valid]]
    names = [x.strip() for x in str(args.tail_length_regions).split(",") if x.strip()]
    mask = np.isin(root_reg, names)
    if not mask.any():
        return strands, radius, f"uniform:no matching regions {','.join(names)}", 0.0, 0

    arc = np.linalg.norm(np.diff(strands, axis=1), axis=2).sum(1)
    good = mask & np.isfinite(arc) & (arc > 1e-9)
    if not good.any():
        return strands, radius, "uniform:no valid tail arcs", 0.0, int(mask.sum())

    target = float(np.median(arc[good]))
    if float(args.tail_length_body_cap) > 0:
        body = (root_reg == "body") & np.isfinite(arc) & (arc > 1e-9)
        if body.any():
            target = min(target, float(np.median(arc[body])) * float(args.tail_length_body_cap))
    if target <= 1e-9:
        return strands, radius, "uniform:bad target", 0.0, int(mask.sum())

    scale = np.ones(len(strands), np.float32)
    scale[good] = (target / np.maximum(arc[good], 1e-9)).astype(np.float32)
    root = strands[:, :1].copy()
    strands = strands.copy()
    strands[good] = root[good] + (strands[good] - root[good]) * scale[good, None, None]

    # Preserve each fiber's relative thickness.  This keeps the radius/arc
    # contract intact when flattening tail length variance.
    radius = radius.copy()
    radius[good] = np.maximum(radius[good] * scale[good], 1e-7).astype(np.float32)

    # Distal LENGTH taper applied AFTER uniformizing: the proximal tail keeps the
    # uniform target (so no piecewise length bands reappear) but the distal end
    # shortens toward the tip, so a curled tail comes to a point ("taper to tip")
    # instead of a uniform-length fuzzy club. Distance is measured outward from
    # the tail_base centroid (robust for curled tails where a single axis fails).
    taper_extra = ""
    tip_scale = float(getattr(args, "tail_length_tip_scale", 1.0) or 1.0)
    tb = (root_reg == "tail_base")
    tl = (root_reg == "tail") & good
    if tip_scale < 1.0 and tb.any() and tl.any():
        base_c = strands[tb, 0].mean(0)
        d = np.linalg.norm(strands[tl, 0] - base_c[None], axis=1)
        s0 = float(np.percentile(d, float(getattr(args, "tail_length_tip_start_pct", 30.0))))
        s1 = float(np.percentile(d, float(getattr(args, "tail_length_tip_end_pct", 98.0))))
        tt = np.clip((d - s0) / max(s1 - s0, 1e-9), 0.0, 1.0).astype(np.float32)
        pw = max(float(getattr(args, "tail_length_tip_pow", 1.5) or 1.5), 1e-3)
        fac = (1.0 - (1.0 - tip_scale) * np.power(tt, pw)).astype(np.float32)
        idx = np.where(tl)[0]
        r0 = strands[idx, :1]
        strands[idx] = r0 + (strands[idx] - r0) * fac[:, None, None]
        radius[idx] = np.maximum(radius[idx] * fac, 1e-7).astype(np.float32)
        taper_extra = f" +distal_taper tip={tip_scale:.2f} pow={pw:.1f} tipmin={float(fac.min()):.2f}"

    arc2 = np.linalg.norm(np.diff(strands, axis=1), axis=2).sum(1)
    summary = (
        f"uniform regions={','.join(names)} roots={int(good.sum())} "
        f"target={target:.5f} std {float(np.std(arc[good])):.5f}->{float(np.std(arc2[good])):.5f}{taper_extra}"
    )
    return strands.astype(np.float32), radius.astype(np.float32), summary, target, int(good.sum())


def nearest_stable_clump_seeds(xy, flow_spacing, cross_spacing, jitter, seed_salt):
    """Nearest jittered infinite-grid seed in world-projected flow/cross coords.

    Unlike the legacy per-patch seed grid, this has no patch-local random phase,
    so neighboring patches see the same clump lanes where their flow fields agree.
    """
    ci0 = np.floor(xy[:, 0] / flow_spacing).astype(np.int32)
    cj0 = np.floor(xy[:, 1] / cross_spacing).astype(np.int32)
    best_seed = None
    best_d2 = None
    for di in (-1, 0, 1, 2):
        for dj in (-1, 0, 1, 2):
            ci = ci0 + di
            cj = cj0 + dj
            sx = ci.astype(np.float32) * flow_spacing
            sy = cj.astype(np.float32) * cross_spacing
            if jitter > 0:
                sx += (hash01_2d(ci, cj, seed_salt) - 0.5) * flow_spacing * jitter
                sy += (hash01_2d(ci, cj, seed_salt + 17) - 0.5) * cross_spacing * jitter
            seed = np.stack([sx, sy], 1).astype(np.float32)
            d2 = ((xy - seed) ** 2).sum(1)
            if best_d2 is None:
                best_d2 = d2
                best_seed = seed
            else:
                take = d2 < best_d2
                best_d2[take] = d2[take]
                best_seed[take] = seed[take]
    return best_seed


def apply_flow_clumping(
    local,
    rootp,
    root_row,
    root_patch,
    tr,
    br,
    patch_centers,
    patch_t,
    patch_b,
    patch_region,
    reg_map,
    prior,
    psize,
    target_arclen,
    args,
    rng,
):
    strength = float(np.clip(args.clump_strength, 0.0, 1.0))
    if strength <= 0.0 or len(local) == 0:
        return np.zeros(len(local), np.float32)

    size = float(args.clump_size)
    if size <= 0.0:
        size = max(0.18 * float(psize), 0.45 * max(float(target_arclen), 1e-6))
    size = max(size, 1e-5)
    aspect = max(float(args.clump_flow_aspect), 1.0)
    flow_spacing = size * aspect
    cross_spacing = size
    max_shift = max(float(target_arclen), size) * float(args.clump_tip_frac)
    max_shift = max(max_shift, 1e-6)
    along_frac = float(np.clip(args.clump_along_frac, 0.0, 1.0))
    jitter = float(np.clip(args.clump_jitter, 0.0, 0.95))
    power = max(float(args.clump_power), 0.1)
    tprof = (np.linspace(0.0, 1.0, local.shape[1], dtype=np.float32) ** power)[None, :]
    root_shift_norm = np.zeros(len(local), np.float32)

    for p in np.unique(root_patch):
        idx = np.nonzero(root_patch == p)[0]
        if len(idx) < 8:
            continue
        row = int(np.bincount(root_row[idx].astype(np.int64)).argmax())
        rname = patch_region[row] if row < len(patch_region) else "body"
        rid = reg_map.get(str(rname), -1)
        theta = direction_for_patch(prior, row, rid, args.dir_coh_gate)
        c, s = np.cos(theta), np.sin(theta)
        pt = patch_t[int(p)].astype(np.float32)
        pb = patch_b[int(p)].astype(np.float32)
        flow = unit((c * pt + s * pb)[None])[0]
        cross = unit((-s * pt + c * pb)[None])[0]

        xy = np.stack([rootp[idx] @ flow, rootp[idx] @ cross], 1).astype(np.float32)
        nearest = nearest_stable_clump_seeds(
            xy,
            flow_spacing,
            cross_spacing,
            jitter,
            int(args.seed) + 7919,
        )
        delta = nearest - xy
        delta[:, 0] *= along_frac
        nrm = np.linalg.norm(delta, axis=1)
        big = nrm > max_shift
        if big.any():
            delta[big] *= (max_shift / np.maximum(nrm[big], 1e-9))[:, None]
            nrm[big] = max_shift
        disp = delta[:, :1] * flow[None] + delta[:, 1:2] * cross[None]
        dx = (disp * tr[idx]).sum(1).astype(np.float32) * strength
        dy = (disp * br[idx]).sum(1).astype(np.float32) * strength
        local[idx, :, 0] += dx[:, None] * tprof
        local[idx, :, 1] += dy[:, None] * tprof
        root_shift_norm[idx] = np.sqrt(dx * dx + dy * dy)

    return root_shift_norm


def clump_density_multiplier(surf_p, valid_p, center, tangent, bitangent, theta, psize, target_arclen, args, rng):
    contrast = float(np.clip(args.clump_density_contrast, 0.0, 0.98))
    if contrast <= 0.0:
        return 1.0
    size = float(args.clump_size)
    if size <= 0.0:
        size = max(0.18 * float(psize), 0.45 * max(float(target_arclen), 1e-6))
    size = max(size, 1e-5)
    sigma = max(size * float(args.clump_density_sigma), 1e-6)
    c, s = np.cos(theta), np.sin(theta)
    cross = unit((-s * tangent + c * bitangent)[None])[0]
    y = (surf_p * cross[None, None]).sum(2)
    phase = float(np.mod(float(args.seed) * 0.61803398875, 1.0) * size)
    d = np.abs(((y + phase + 0.5 * size) % size) - 0.5 * size)
    lane = np.exp(-0.5 * (d / sigma) ** 2).astype(np.float32)
    ok = valid_p > 0.5
    if ok.any():
        lane = lane / max(float(lane[ok].mean()), 1e-6)
    mult = 1.0 + contrast * (lane - 1.0)
    mult = np.clip(mult, float(args.clump_density_min), float(args.clump_density_max)).astype(np.float32)
    if ok.any():
        mult = mult / max(float(mult[ok].mean()), 1e-6)
    return mult


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True, help="v2_region_prior.npz")
    ap.add_argument("--patch_params", default="", help="optional Step2 patch_params.npz; optimized flow_delta overrides patch_dir during grow")
    ap.add_argument("--patch_param_vars", default="all",
                    help="comma-separated patch_params fields to consume from --patch_params. Default 'all' preserves legacy behavior. Use 'flow' for the locked mainline so Step2 cannot accidentally alter rho/length/lift.")
    ap.add_argument("--z64_map_path", default="", help="optional patch_z64_map.npy memmap; overrides/augments prior path without storing the large map in the npz")
    ap.add_argument("--z64_profile_npz", default="",
                    help="optional single-patch fur npz containing z64_map. When set, every root samples this profile map by its local root_uv instead of using per-patch z64 means. Use this for adopted-anchor character tests.")
    ap.add_argument("--strand_profile_npz", default="",
                    help="STAMPING route: a single adopted 3D patch (p1) fur npz with strands+root_uv. Each dog root copies the nearest p1 strand offset (by local root_uv) directly as its 3D shape -- BYPASSING z64 decode/groom entirely. The patch's baked character (curl/lift/clump/radius/local-mess) transfers as-is; Step2 only rotates by flow_delta and rescales length. p1 strands are canonicalized so their mean tangent = +x, making flow_delta the absolute direction.")
    ap.add_argument("--strand_profile_uv_crop", default="",
                    help="optional 'u0,v0,u1,v1' crop inside --strand_profile_npz root_uv. The cropped reliable 3D swatch is renormalized to [0,1]^2 before stamping, so only the good predicted sub-patch is tiled/adopted across target patches.")
    ap.add_argument("--strand_profile_lift_scale", type=float, default=1.0,
                    help="direct 3D stamping only: multiply the adopted swatch local normal/z component before world-length rescale. This tests/fixes insufficient standoff without changing density, direction, color, or latent decoding.")
    ap.add_argument("--direction_profile_npz", default="",
                    help="optional single-patch fur npz containing direction_vec_map. Its local direction residuals are added on top of the macro Step2/patch direction, preserving the anchor patch's visual groom texture.")
    ap.add_argument("--direction_profile_strength", type=float, default=1.0,
                    help="strength for --direction_profile_npz local axial residuals; 0 disables, 1 preserves the anchor residual exactly")
    ap.add_argument("--direction_profile_conf_gate", type=float, default=0.25,
                    help="minimum direction_conf_map value to apply profile residual")
    ap.add_argument("--appearance_profile_npz", default="",
                    help="optional single-patch fur npz with photo_color_map/color_map. Extracts zero-mean high-frequency luminance as an adopted patch visual profile.")
    ap.add_argument("--appearance_profile_strength", type=float, default=0.0,
                    help="strength of the adopted luminance/groove profile; 0 disables. Applied to root-density lanes and per-root color brightness.")
    ap.add_argument("--appearance_profile_clip", type=float, default=0.35,
                    help="max fractional multiplier from the adopted appearance profile")
    ap.add_argument("--pack", default="", help="surface_patch_dataset.npz; default from prior source_pack")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--codec_path", default="patchlocks/models/fur_strand_codec_v6upright.pt")
    ap.add_argument("--target_strands", type=int, default=0,
                    help="<=0 (DEFAULT): DENSITY-INTEGRAL -- root count = sum(calibrated final_density * PoU_share) * base_density, i.e. the PHYSICAL hair count the density implies (~millions for a whole dog; transferable, no magic number). >0: force this exact total (rescales density, was the old footgun -- 60k-300k = ~168 roots/patch = GRAINY; use only for lightweight test renders).")
    ap.add_argument("--base_density", type=float, default=1.0, help="density-integral multiplier (used when --target_strands <= 0); 1.0 = the calibrated physical density. Raise for a fuller coat, lower for a lighter/faster render.")
    ap.add_argument("--max_patches", type=int, default=0)
    ap.add_argument("--points", type=int, default=32)
    ap.add_argument("--decode_chunk", type=int, default=32768)
    ap.add_argument("--target_arclen", type=float, default=0.0,
                    help="optional global median arclength target. <=0 keeps the adopted/decoded strand's own length and only applies explicit patch/scale-contract ratios; pass 0.029 explicitly for legacy beagle-style global locking.")
    ap.add_argument("--length_mode", default="region_smooth", choices=["raw", "region", "patch_smooth", "region_smooth"],
                    help="consume diffusion-decoded relative length: raw preserves decoded length; region/patch_smooth/region_smooth stabilize it")
    ap.add_argument("--length_smooth_k", type=int, default=8)
    ap.add_argument("--length_smooth_iters", type=int, default=2)
    ap.add_argument("--no_pou", action="store_true",
                    help="ABLATION: disable partition-of-unity normalization across overlapping patches "
                         "(share = raw cosine window, not normalized to sum 1) -> seams / uneven coverage / "
                         "bald patch edges. Demonstrates that the surface grow is not naive patch-pasting.")
    ap.add_argument("--length_region_blend", type=float, default=0.75,
                    help="for region_smooth: 1=region mean only, 0=raw patch ratio before spatial smoothing")
    ap.add_argument("--length_ratio_clip", type=float, default=1.8)
    ap.add_argument("--scale_contract", default="none", choices=["none", "patch_length_remap", "patch_window_footprint"],
                    help="opt-in 2D-to-3D scale contract. patch_window_footprint uses crop_px/native_patch_px: the diffusion patch is treated as a visual window whose footprint can cover multiple geodesic patches. patch_length_remap is the older model-reference scale.")
    ap.add_argument("--scale_contract_clip", type=float, default=4.0,
                    help="clip scale-contract multipliers to [1/clip, clip]")
    ap.add_argument("--scale_contract_density_exp", type=float, default=0.0,
                    help="when >0, multiply root density by scale_contract**exp. exp=1 is a conservative visual-density correction; exp=2 preserves roots per scaled patch area but can create very large surfpacks.")
    ap.add_argument("--scale_contract_length_exp", type=float, default=1.0,
                    help="when scale_contract is enabled, multiply strand length by scale_contract**exp. Use 0 when length is already correct and only density/radius need visual-scale correction.")
    ap.add_argument("--scale_contract_radius_exp", type=float, default=0.0,
                    help="when >0, multiply final strand radius by scale_contract**exp per root. This makes individual fibers visible at the 2D patch scale without necessarily lengthening the coat.")
    ap.add_argument("--radius", type=float, default=0.0013)
    ap.add_argument("--radius_arc_ratio", type=float, default=0.0,
                    help="if >0, set each strand radius to radius_arc_ratio * its final arclength. This preserves an adopted patch's relative fiber thickness when target_arclen changes.")
    ap.add_argument("--radius_profile_npz", default="",
                    help="optional single-patch fur npz used to infer radius_arc_ratio = median(radius)/median(arclen). Overrides fixed --radius unless --radius_arc_ratio is explicitly set.")
    ap.add_argument("--dir_coh_gate", type=float, default=0.08)
    ap.add_argument("--direction_mode", default="external", choices=["external", "latent"],
                    help="external: canon-dir consumption, rotate decoded local xy by patch/region axial direction (measured or model-predicted). latent: V2-clean absdir consumption, keep decoded z64 xy direction as-is so direction stays inside the latent.")
    ap.add_argument("--z_mode", default="abs", choices=["abs", "clamp", "raw"],
                    help="consume codec normal coordinate: abs preserves lift while forcing it outside the surface")
    ap.add_argument("--normal_lift_scale", type=float, default=1.0,
                    help="scale decoded local normal lift after --z_mode; lower values make short coats lie closer to the body")
    ap.add_argument("--lift_mode", default="z64", choices=["z64", "external"],
                    help="z64 (legacy): standoff/normal-rise comes from decoded z64 (scaled) -- z64 IMPLICITLY decides absolute lift, tends too fluffy (beagle height 0.0143 vs GT 0.0072). external: DECOUPLE lift from z64 -- keep only z64 in-plane shape (curl/bend/flow), set the normal rise to an EXPLICIT profile = lift_height * (normalized in-plane arclength). z64 no longer decides absolute height; lift is calibrated like length/density.")
    ap.add_argument("--lift_height", type=float, default=0.0072,
                    help="external-lift target normal rise at the strand tip (world units). GT beagle real fur ~0.0072 (V2 legacy 0.0143 = 2x too fluffy, V1-style 0.0014 = too flat).")
    ap.add_argument("--torso_lift_k", type=float, default=0.0,
                    help="extra body/torso-only loft for the 'fluffy' look. Adds a normal rise = clip(torso_lift_k * strand_arclen, 0, torso_lift_cap) ONLY on torso regions (see --torso_lift_regions). Rise is proportional to that strand's final arclength (longer fur lofts more) and capped so it never blows up. 0 = off (legs/face/tail/ears untouched). Applied ON TOP of the existing z64/strand-profile lift, after the length rescale.")
    ap.add_argument("--torso_lift_cap", type=float, default=0.010,
                    help="hard cap (world units) on the extra torso loft from --torso_lift_k, so 'proportional to length' never gets too large.")
    ap.add_argument("--torso_lift_regions", default="body",
                    help="comma list of region names that count as torso for --torso_lift_k (default: body).")
    ap.add_argument("--torso_lift_frac", type=float, default=1.0,
                    help="fraction (0..1) of torso strands that actually get lifted; the rest stay flat. <1 gives the natural 'some hairs stand up, most lie flat' plush look instead of lofting the whole coat uniformly. 1.0 = lift all torso strands.")
    ap.add_argument("--torso_lift_mode", default="random", choices=["random", "smooth"],
                    help="random keeps the legacy per-strand Bernoulli loft. smooth converts the torso mask into a continuous patch-neighborhood field and interprets torso_lift_frac as mean amplitude, avoiding speckle and hard region seams.")
    ap.add_argument("--torso_lift_smooth_k", type=int, default=8,
                    help="nearest patch count for --torso_lift_mode smooth")
    ap.add_argument("--torso_lift_smooth_iters", type=int, default=4,
                    help="surface-field smoothing iterations for --torso_lift_mode smooth")
    ap.add_argument("--length_soft_caps", default="",
                    help="region:metres,... soft maximum strand length per region (e.g. body:0.03,tail:0.05); "
                         "length beyond the cap is compressed by --length_soft_cap_slope. Empty = off")
    ap.add_argument("--length_soft_cap_slope", type=float, default=0.35)
    ap.add_argument("--torso_lift_seed", type=int, default=12345,
                    help="RNG seed for the --torso_lift_frac random strand selection (deterministic).")
    ap.add_argument("--comb_blend", type=float, default=0.0,
                    help="blend decoded local strands toward a coherent measured-direction stroke; fixes stippled short-fur renders")
    ap.add_argument("--comb_tangent_frac", type=float, default=0.78,
                    help="analytic stroke tangent chord as a fraction of target_arclen for --comb_blend")
    ap.add_argument("--comb_cross_damp", type=float, default=0.15,
                    help="damp cross-flow wiggle in the analytic stroke target")
    ap.add_argument("--clump_strength", type=float, default=0.0,
                    help="grow-time locks: bend strand mid/tips toward local clump cells; 0 keeps legacy grow")
    ap.add_argument("--clump_size", type=float, default=0.0,
                    help="world-space cross-flow spacing between clump lanes; <=0 auto from patch size/target length")
    ap.add_argument("--clump_flow_aspect", type=float, default=2.8,
                    help="clump cell aspect along measured flow; >1 makes elongated lock lanes instead of round star clumps")
    ap.add_argument("--clump_along_frac", type=float, default=0.18,
                    help="fraction of along-flow convergence; keep small to preserve measured groom direction")
    ap.add_argument("--clump_tip_frac", type=float, default=0.55,
                    help="max tip lateral shift as fraction of max(target_arclen, clump_size)")
    ap.add_argument("--clump_power", type=float, default=1.7,
                    help="tip profile exponent: higher keeps roots/midpoints steadier and moves mostly tips")
    ap.add_argument("--clump_jitter", type=float, default=0.35,
                    help="jitter clump cell centers to avoid visible grid locking")
    ap.add_argument("--clump_density_contrast", type=float, default=0.0,
                    help="modulate root sampling into subtle flow-aligned lanes; creates lock gaps before tip clumping")
    ap.add_argument("--clump_density_sigma", type=float, default=0.30,
                    help="lane width as a fraction of clump_size for --clump_density_contrast")
    ap.add_argument("--clump_density_min", type=float, default=0.45,
                    help="minimum normalized root density multiplier for lock gaps")
    ap.add_argument("--clump_density_max", type=float, default=2.2,
                    help="maximum normalized root density multiplier on lock lanes")
    ap.add_argument("--root_sink", type=float, default=0.0,
                    help="move rendered strand geometry slightly below the surface normal so root caps do not read as dots")
    ap.add_argument("--region_detail_policy", default="none", choices=["none", "generic"],
                    help="generic post-grow region cleanup: protect body/tail_base, compact ears/paws and distal tail only")
    ap.add_argument("--region_detail_tail_scale", type=float, default=0.68,
                    help="generic policy: distal tail strand/radius scale at the tail tip")
    ap.add_argument("--region_detail_tail_keep", type=float, default=0.88,
                    help="generic policy: distal tail root keep probability at the tail tip")
    ap.add_argument("--region_detail_tail_body_cap", type=float, default=1.12,
                    help="generic policy: cap tail median arclength to this multiple of body median; <=0 disables")
    ap.add_argument("--region_detail_tail_start_pct", type=float, default=35.0,
                    help="generic policy: tail projection percentile where distal taper starts")
    ap.add_argument("--region_detail_tail_end_pct", type=float, default=96.0,
                    help="generic policy: tail projection percentile where distal taper reaches full strength")
    ap.add_argument("--region_detail_tail_uniform_scale", type=float, default=1.0,
                    help="generic policy: length/radius scale applied to the WHOLE tail (feathered at the base), on top of the distal taper. <1 shortens a long fuzzy tail overall. NOTE: when --tail_length_policy uniform is on, the final length is set by that policy, so this mainly affects radius; use --tail_length_body_cap to shorten a uniformized tail")
    ap.add_argument("--region_detail_tail_uniform_keep", type=float, default=1.0,
                    help="generic policy: root keep probability applied to the WHOLE tail (feathered at the base). <1 thins a dense curled tail everywhere so it stops reading as a pom-pom (the user's 'reduce density')")
    ap.add_argument("--region_detail_tail_metric", default="axis", choices=["axis", "base_dist"],
                    help="how distal 'tip-ness' t is measured along the tail. axis=projection onto the base->tail axis (good for straight tails). base_dist=Euclidean distance outward from the tail_base centroid (robust for CURLED/balled tails where a single axis cannot separate base from tip)")
    ap.add_argument("--region_detail_tail_taper_pow", type=float, default=1.0,
                    help="generic policy: exponent on the distal-tail ramp; >1 keeps the proximal tail near body length and concentrates the length+density falloff toward the tip so it comes to a point instead of balling up")
    ap.add_argument("--region_detail_tail_keep_pow", type=float, default=1.0,
                    help="generic policy: extra exponent (on top of taper_pow) for the tail DENSITY falloff, so the very tip thins out faster than it shortens")
    ap.add_argument("--region_detail_tail_tip_keep", type=float, default=-1.0,
                    help="generic policy: root keep probability at the very tail tip (<0 falls back to --region_detail_tail_keep); lower = sparser pointed tip, kills the pom-pom")
    ap.add_argument("--region_detail_feather", type=float, default=0.0,
                    help="generic policy: world-space half-width of a smoothstep transition band across region boundaries (ears/paws/face). 0 = hard mask (legacy). >0 feathers the length/density change so refined regions blend smoothly into the body with no visible seam")
    ap.add_argument("--region_detail_ear_scale", type=float, default=0.78,
                    help="generic policy: compact ear strand/radius scale")
    ap.add_argument("--region_detail_ear_keep", type=float, default=0.95,
                    help="generic policy: compact ear root keep probability")
    ap.add_argument("--region_detail_ear_body_cap", type=float, default=0.82,
                    help="generic policy: cap ear median arclength to this multiple of body median; <=0 disables")
    ap.add_argument("--region_detail_leg_scale", type=float, default=1.0,
                    help="individual-refine policy: compact leg strand/radius scale. Default 1.0 leaves legs unchanged; use with --region_detail_leg_feather to smooth the body/leg seam.")
    ap.add_argument("--region_detail_leg_keep", type=float, default=1.0,
                    help="individual-refine policy: leg root keep probability. Default 1.0 leaves legs unchanged.")
    ap.add_argument("--region_detail_leg_body_cap", type=float, default=0.0,
                    help="individual-refine policy: cap leg median arclength to this multiple of body median; <=0 disables.")
    ap.add_argument("--region_detail_leg_feather", type=float, default=0.0,
                    help="individual-refine policy: leg-specific feather half-width (world units). Spreads leg/body length and density changes across the shoulder/belly transition.")
    ap.add_argument("--region_detail_paw_scale", type=float, default=0.78,
                    help="generic policy: compact paw strand/radius scale")
    ap.add_argument("--region_detail_paw_keep", type=float, default=1.0,
                    help="generic policy: compact paw root keep probability")
    ap.add_argument("--region_detail_paw_body_cap", type=float, default=0.82,
                    help="generic policy: cap paw median arclength to this multiple of body median; <=0 disables")
    ap.add_argument("--region_detail_paw_feather", type=float, default=0.0,
                    help="generic policy: paw-specific feather half-width (world units); overrides --region_detail_feather for the paw band only. Wider here spreads the short-paw->long-leg length change over a longer zone so it stops reading as a 'sock' cuff, without widening the tuned tail/ear bands. 0 = use global feather")
    ap.add_argument("--region_detail_face_scale", type=float, default=1.0,
                    help="generic policy: compact face/muzzle strand/radius scale (1.0=off); <1 shortens muzzle fuzz")
    ap.add_argument("--region_detail_face_keep", type=float, default=1.0,
                    help="generic policy: compact face root keep probability (1.0=off)")
    ap.add_argument("--region_detail_face_body_cap", type=float, default=0.0,
                    help="generic policy: cap face median arclength to this multiple of body median; <=0 disables")
    ap.add_argument("--region_detail_seed", type=int, default=2027,
                    help="generic policy RNG seed for deterministic root thinning")
    ap.add_argument("--tail_length_policy", default="none", choices=["none", "uniform"],
                    help="post-detail tail cleanup. uniform makes all roots in --tail_length_regions share one final arclength; useful for thin curved tails where patchwise length bands show up as segments")
    ap.add_argument("--tail_length_regions", default="tail,tail_base",
                    help="comma-separated patch_region names treated as one tail for --tail_length_policy uniform")
    ap.add_argument("--tail_length_body_cap", type=float, default=0.0,
                    help="optional cap: uniform tail target <= body median arclength * this value; <=0 disables")
    ap.add_argument("--tail_length_tip_scale", type=float, default=1.0,
                    help="distal tail LENGTH taper applied AFTER uniformizing (1.0=off). <1 shortens fur toward the tip so a curled tail tapers to a point instead of a uniform-length club; survives the uniform policy that otherwise flattens length")
    ap.add_argument("--tail_length_tip_start_pct", type=float, default=30.0,
                    help="percentile of outward base-distance where the distal length taper starts (proximal tail below this keeps full uniform length)")
    ap.add_argument("--tail_length_tip_end_pct", type=float, default=98.0,
                    help="percentile of outward base-distance where the distal length taper reaches --tail_length_tip_scale")
    ap.add_argument("--tail_length_tip_pow", type=float, default=1.5,
                    help="exponent on the distal length-taper ramp; >1 concentrates the shortening toward the very tip")
    ap.add_argument("--region_flow_policy", default="none", choices=["none", "generic"],
                    help="generic pre-grow flow cleanup for thin/small regions; stabilizes Step2 flow residuals on ears/paws/tail without changing fur character")
    ap.add_argument("--region_flow_smooth_k", type=int, default=10,
                    help="generic flow cleanup: same-region KNN size")
    ap.add_argument("--region_flow_iters", type=int, default=2,
                    help="generic flow cleanup: smoothing iterations")
    ap.add_argument("--region_flow_ear_smooth", type=float, default=0.65,
                    help="generic flow cleanup: smoothing strength for ears")
    ap.add_argument("--region_flow_ear_damp", type=float, default=0.55,
                    help="generic flow cleanup: residual damping around ear region mean")
    ap.add_argument("--region_flow_tail_smooth", type=float, default=0.55,
                    help="generic flow cleanup: smoothing strength for tail+tail_base")
    ap.add_argument("--region_flow_tail_damp", type=float, default=0.65,
                    help="generic flow cleanup: residual damping around tail region mean")
    ap.add_argument("--region_flow_paw_smooth", type=float, default=0.35,
                    help="generic flow cleanup: light smoothing strength for paws")
    ap.add_argument("--region_flow_paw_damp", type=float, default=0.85,
                    help="generic flow cleanup: light residual damping around paw region mean")
    ap.add_argument("--backfill_min_face_roots", type=int, default=0,
                    help="ensure at least this many roots on selected mesh faces by inheriting the nearest patch prior")
    ap.add_argument("--backfill_scope", default="covered", choices=["covered", "all"],
                    help="faces eligible for --backfill_min_face_roots")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    prior = np.load(args.prior, allow_pickle=True)
    pack_path = infer_pack(prior, args.pack)
    pack = np.load(pack_path, allow_pickle=True)

    surf = pack["surf"].astype(np.float32)
    normg = pack["normg"].astype(np.float32)
    valid = pack["valid"].astype(np.float32)
    cface = pack["cell_face"].astype(np.int32)
    cbary = pack["cell_bary"].astype(np.float32)
    tt = pack["Tt"].astype(np.float32)
    bb = pack["Bb"].astype(np.float32) if "Bb" in pack.files else unit(np.cross(pack["Nn"], tt))
    psize = float(pack["psize"]) if "psize" in pack.files else 1.0
    h = valid.shape[-1]

    rows = patch_rows(prior, len(surf), args.max_patches)
    if not rows:
        raise RuntimeError("no prior patch rows map to the surface pack")
    row_ids = np.array([r for r, _p in rows], np.int64)
    patch_ids = np.array([p for _r, p in rows], np.int64)

    z64 = prior["patch_z64"].astype(np.float32)
    if z64.shape[1] != 64:
        raise ValueError(f"prior patch_z64 must be (P,64), got {z64.shape}")
    z64_map = None
    z64_profile = None
    if args.z64_profile_npz:
        prof = np.load(args.z64_profile_npz, allow_pickle=True)
        if "z64_map" not in prof.files:
            raise KeyError(f"--z64_profile_npz must contain z64_map: {args.z64_profile_npz}")
        z64_profile = prof["z64_map"].astype(np.float32)
        if z64_profile.ndim != 3 or z64_profile.shape[-1] != 64:
            raise ValueError(f"profile z64_map must be (H,W,64), got {z64_profile.shape}")
        print(f"  z64 source: single adopted profile map {z64_profile.shape} from {args.z64_profile_npz}", flush=True)
    strand_profile = None
    strand_profile_angle = 0.0
    if args.strand_profile_npz:
        from scipy.spatial import cKDTree
        sp_prof = np.load(args.strand_profile_npz, allow_pickle=True)
        if "strands" not in sp_prof.files or "root_uv" not in sp_prof.files:
            raise KeyError(f"--strand_profile_npz must contain strands and root_uv: {args.strand_profile_npz}")
        _ps = sp_prof["strands"].astype(np.float32)                              # (M, P0, 3) planar local swatch (root z=0)
        _puv = sp_prof["root_uv"].astype(np.float32)                             # (M, 2) in [0,1]^2
        strand_profile_crop = np.array([0.0, 0.0, 1.0, 1.0], np.float32)
        if args.strand_profile_uv_crop:
            vals = [float(x) for x in args.strand_profile_uv_crop.replace(",", " ").split()]
            if len(vals) != 4:
                raise ValueError("--strand_profile_uv_crop must be 'u0,v0,u1,v1'")
            u0, v0, u1, v1 = vals
            if not (0.0 <= u0 < u1 <= 1.0 and 0.0 <= v0 < v1 <= 1.0):
                raise ValueError(f"bad --strand_profile_uv_crop={args.strand_profile_uv_crop}; expected 0<=u0<u1<=1 and 0<=v0<v1<=1")
            keep = (_puv[:, 0] >= u0) & (_puv[:, 0] <= u1) & (_puv[:, 1] >= v0) & (_puv[:, 1] <= v1)
            if int(keep.sum()) < 32:
                raise ValueError(f"--strand_profile_uv_crop kept only {int(keep.sum())} roots; choose a larger reliable swatch")
            _ps = _ps[keep]
            _puv = _puv[keep].copy()
            _puv[:, 0] = (_puv[:, 0] - u0) / max(u1 - u0, 1e-9)
            _puv[:, 1] = (_puv[:, 1] - v0) / max(v1 - v0, 1e-9)
            _puv = np.clip(_puv, 0.0, 1.0)
            strand_profile_crop = np.array([u0, v0, u1, v1], np.float32)
            print(
                f"  strand profile crop: uv=({u0:.3f},{v0:.3f},{u1:.3f},{v1:.3f}) "
                f"kept {len(_ps)} reliable source strands and renormalized to [0,1]^2",
                flush=True,
            )
        _off = _ps - _ps[:, :1]                                                  # offsets in p1 local frame (x,y in-plane, z normal)
        if _off.shape[1] != args.points:
            _idx = np.linspace(0, _off.shape[1] - 1, args.points).round().astype(np.int64)
            _off = _off[:, _idx]
        if float(args.strand_profile_lift_scale) != 1.0:
            _off[..., 2] *= float(args.strand_profile_lift_scale)
        _ang = np.arctan2(_off[:, -1, 1], _off[:, -1, 0])                        # per-strand tip tangent angle
        _ma = float(np.arctan2(np.sin(_ang).mean(), np.cos(_ang).mean()))        # circular mean tangent
        strand_profile_angle = _ma
        _cc, _ss = np.cos(-_ma), np.sin(-_ma)
        _ox, _oy = _off[..., 0].copy(), _off[..., 1].copy()
        _off[..., 0] = _cc * _ox - _ss * _oy                                     # rotate mean tangent -> +x so flow_delta is absolute
        _off[..., 1] = _ss * _ox + _cc * _oy
        strand_profile = {"offsets": _off.astype(np.float32), "tree": cKDTree(_puv), "crop": strand_profile_crop}
        print(
            f"  STRAND PROFILE STAMP: {len(_off)} p1 strands x {args.points}pts, "
            f"mean-tangent canonicalized (rot {np.degrees(_ma):.1f}deg), "
            f"lift_scale={float(args.strand_profile_lift_scale):.2f}, "
            f"bypassing z64/groom; from {args.strand_profile_npz}",
            flush=True,
        )
    elif z64_profile is None and "patch_z64_map" in prior.files:
        z64_map = prior["patch_z64_map"].astype(np.float32)
        if z64_map.ndim != 4 or z64_map.shape[0] != len(z64) or z64_map.shape[-1] != 64:
            raise ValueError(f"prior patch_z64_map must be (P,H,W,64), got {z64_map.shape}; P={len(z64)}")
        print(f"  z64 source: full patch map {z64_map.shape} (root_uv bilinear sampling)", flush=True)
    else:
        zmap_path = args.z64_map_path
        if not zmap_path and "patch_z64_map_path" in prior.files:
            zmap_path = str(np.asarray(prior["patch_z64_map_path"]).reshape(-1)[0])
        if zmap_path:
            if not os.path.exists(zmap_path):
                raise FileNotFoundError(f"patch_z64_map_path not found: {zmap_path}")
            z64_map = np.load(zmap_path, mmap_mode="r")
            if z64_map.ndim != 4 or z64_map.shape[0] != len(z64) or z64_map.shape[-1] != 64:
                raise ValueError(f"patch_z64_map npy must be (P,H,W,64), got {z64_map.shape}; P={len(z64)}")
            print(f"  z64 source: external npy map {z64_map.shape} from {zmap_path} (root_uv bilinear sampling)", flush=True)
    if z64_map is None and z64_profile is None:
        print("  z64 source: per-patch mean fallback (legacy; no patch-local texture)", flush=True)
    density = prior["final_density"].astype(np.float32)
    patch_region = prior["patch_region"].astype(str) if "patch_region" in prior.files else np.full(len(density), "body")
    if "centers" in pack.files and len(pack["centers"]) >= len(density):
        patch_centers_all = pack["centers"].astype(np.float32)[:len(density)]
    else:
        patch_centers_all = surf.mean(axis=(1, 2)).astype(np.float32)[:len(density)]
    scale_contract = np.ones(len(density), np.float32)
    scale_contract_source = "none"
    if args.scale_contract == "patch_length_remap":
        if "patch_length_remap" not in prior.files or len(prior["patch_length_remap"]) != len(density):
            raise KeyError("--scale_contract patch_length_remap requires prior['patch_length_remap'] with one value per patch")
        clip = max(float(args.scale_contract_clip), 1.0)
        scale_contract = np.clip(prior["patch_length_remap"].astype(np.float32), 1.0 / clip, clip)
        scale_contract_source = "patch_length_remap"
        print(
            "  scale contract: patch_length_remap "
            f"pct={np.percentile(scale_contract, [5,25,50,75,95]).round(3).tolist()} "
            f"clip={clip:.2f} density_exp={float(args.scale_contract_density_exp):.2f} "
            f"length_exp={float(args.scale_contract_length_exp):.2f} "
            f"radius_exp={float(args.scale_contract_radius_exp):.2f}",
            flush=True,
        )
    elif args.scale_contract == "patch_window_footprint":
        need = ["patch_crop_px", "patch_native_px"]
        missing = [k for k in need if k not in prior.files or len(prior[k]) != len(density)]
        if missing:
            raise KeyError(f"--scale_contract patch_window_footprint requires prior fields {need}; missing {missing}")
        native = prior["patch_native_px"].astype(np.float32)
        crop = prior["patch_crop_px"].astype(np.float32)
        raw = np.ones(len(density), np.float32)
        ok = np.isfinite(native) & np.isfinite(crop) & (native > 1e-3) & (crop > 1e-3)
        raw[ok] = crop[ok] / np.maximum(native[ok], 1e-6)
        if "patch_feature_from" in prior.files:
            src = prior["patch_feature_from"].astype(np.int32)
            inh = (src >= 0) & (src < len(raw)) & ~ok
            raw[inh] = raw[src[inh]]
        clip = max(float(args.scale_contract_clip), 1.0)
        scale_contract = np.clip(raw, 1.0 / clip, clip).astype(np.float32)
        scale_contract_source = "patch_crop_px/patch_native_px"
        print(
            "  scale contract: patch_window_footprint=crop/native "
            f"raw_pct={np.percentile(raw[np.isfinite(raw)], [5,25,50,75,95]).round(3).tolist()} "
            f"pct={np.percentile(scale_contract, [5,25,50,75,95]).round(3).tolist()} "
            f"clip={clip:.2f} density_exp={float(args.scale_contract_density_exp):.2f} "
            f"length_exp={float(args.scale_contract_length_exp):.2f} "
            f"radius_exp={float(args.scale_contract_radius_exp):.2f}",
            flush=True,
        )
    if args.scale_contract != "none":
        if float(args.scale_contract_density_exp) != 0.0:
            dens_mult = np.power(scale_contract, float(args.scale_contract_density_exp)).astype(np.float32)
            density = density * dens_mult
            print(
                "  density scale contract applied: "
                f"mult pct={np.percentile(dens_mult, [5,25,50,75,95]).round(3).tolist()}",
                flush=True,
            )
    reg_map = region_lookup(prior)
    opt_flow_delta = None
    opt_length = None
    opt_lift = None
    region_flow_summary = "none"
    if args.patch_params:
        pp = np.load(args.patch_params, allow_pickle=True)
        _pv = str(args.patch_param_vars).strip().lower()
        _consume_all = _pv in ("", "all", "*")
        _consume = {x.strip() for x in _pv.split(",") if x.strip()}

        def full_param(key, default):
            out = np.full(len(density), float(default), np.float32)
            if key not in pp.files:
                return out
            arr = pp[key].astype(np.float32)
            if len(arr) == len(density):
                return arr.astype(np.float32)
            if "used_patch" in pp.files and len(arr) == len(pp["used_patch"]):
                used = pp["used_patch"].astype(np.int64)
                ok = (used >= 0) & (used < len(out))
                out[used[ok]] = arr[ok]
                return out
            raise ValueError(f"{args.patch_params}:{key} shape {arr.shape} cannot map to P={len(density)}")

        if "flow_delta" in pp.files and (_consume_all or "flow" in _consume or "flow_delta" in _consume):
            opt_flow_delta = full_param("flow_delta", 0.0)
            print(f"  direction source override: Step2 optimized flow_delta from {args.patch_params}", flush=True)
        if "rho" in pp.files and (_consume_all or "rho" in _consume or "density" in _consume):
            rho_resid = full_param("rho", 1.0)
            density = density * np.clip(rho_resid, 0.0, 4.0)
            print(f"  density source: prior final_density x Step2 rho residual [{rho_resid.min():.3f},{rho_resid.max():.3f}]", flush=True)
        if "length" in pp.files and (_consume_all or "length" in _consume or "len" in _consume):
            opt_length = full_param("length", 1.0)
            print(f"  length source: Step2 length residual [{opt_length.min():.3f},{opt_length.max():.3f}]", flush=True)
        if "lift" in pp.files and (_consume_all or "lift" in _consume):
            opt_lift = full_param("lift", 1.0)
            print(f"  lift source: Step2 lift residual [{opt_lift.min():.3f},{opt_lift.max():.3f}]", flush=True)
    if opt_flow_delta is not None:
        opt_flow_delta, region_flow_summary = smooth_region_flow_delta(opt_flow_delta, patch_region, patch_centers_all, args)
        if str(args.region_flow_policy) != "none":
            print(f"  region_flow={args.region_flow_policy} | {region_flow_summary}", flush=True)
    if "patch_dir_source" in prior.files:
        src = str(np.asarray(prior["patch_dir_source"]).reshape(-1)[0])
        print(f"  direction source: external axial patch_dir ({src})", flush=True)
    dir_vec_map = None
    dir_vec_map_conf = None
    dir_profile_vec = None
    dir_profile_conf = None
    dir_profile_theta0 = 0.0
    if args.direction_profile_npz:
        prof = np.load(args.direction_profile_npz, allow_pickle=True)
        if "direction_vec_map" not in prof.files:
            raise KeyError(f"--direction_profile_npz must contain direction_vec_map: {args.direction_profile_npz}")
        dir_profile_vec = prof["direction_vec_map"].astype(np.float32)
        if dir_profile_vec.ndim != 3 or dir_profile_vec.shape[-1] != 2:
            raise ValueError(f"profile direction_vec_map must be (H,W,2), got {dir_profile_vec.shape}")
        if "direction_conf_map" in prof.files:
            dir_profile_conf = prof["direction_conf_map"].astype(np.float32)
            if dir_profile_conf.shape != dir_profile_vec.shape[:2]:
                raise ValueError(f"profile direction_conf_map must be {dir_profile_vec.shape[:2]}, got {dir_profile_conf.shape}")
        vx, vy = dir_profile_vec[..., 0], dir_profile_vec[..., 1]
        ax = np.stack([vx * vx - vy * vy, 2.0 * vx * vy], -1)
        ww = (dir_profile_conf if dir_profile_conf is not None else np.ones(dir_profile_vec.shape[:2], np.float32))[..., None]
        mean_ax = (ax * ww).reshape(-1, 2).sum(0) / (ww.reshape(-1, 1).sum(0) + 1e-9)
        dir_profile_theta0 = float(0.5 * np.arctan2(mean_ax[1], mean_ax[0]))
        print(
            f"  direction profile: {dir_profile_vec.shape} theta0={np.degrees(dir_profile_theta0):.1f}deg "
            f"strength={args.direction_profile_strength:.2f} from {args.direction_profile_npz}",
            flush=True,
        )
    if "patch_dir_vec_map" in prior.files:
        dir_vec_map = prior["patch_dir_vec_map"].astype(np.float32)
        if dir_vec_map.ndim != 4 or dir_vec_map.shape[0] != len(z64) or dir_vec_map.shape[-1] != 2:
            raise ValueError(f"prior patch_dir_vec_map must be (P,H,W,2), got {dir_vec_map.shape}; P={len(z64)}")
        if "patch_dir_vec_map_conf" in prior.files:
            dir_vec_map_conf = prior["patch_dir_vec_map_conf"].astype(np.float32)
            if dir_vec_map_conf.shape != dir_vec_map.shape[:3]:
                raise ValueError(f"prior patch_dir_vec_map_conf must be {dir_vec_map.shape[:3]}, got {dir_vec_map_conf.shape}")
        print(f"  direction source: full patch vector maps {dir_vec_map.shape} (root_uv bilinear sampling)", flush=True)

    appearance_profile = None
    if args.appearance_profile_npz and float(args.appearance_profile_strength) != 0.0:
        prof = np.load(args.appearance_profile_npz, allow_pickle=True)
        if "photo_color_map" in prof.files:
            cm = prof["photo_color_map"].astype(np.float32)
        elif "color_map" in prof.files:
            cm = prof["color_map"].astype(np.float32)[..., :3]
        else:
            raise KeyError(f"--appearance_profile_npz must contain photo_color_map or color_map: {args.appearance_profile_npz}")
        if cm.ndim != 3 or cm.shape[-1] < 3:
            raise ValueError(f"appearance color map must be (H,W,3+), got {cm.shape}")
        lum = 0.299 * cm[..., 0] + 0.587 * cm[..., 1] + 0.114 * cm[..., 2]
        # Tiny dependency-free blur.  Subtracting it keeps fur streaks and
        # suppresses broad markings, so a white/brown boundary is not tiled over
        # the whole animal as a color patch.
        blur = lum.astype(np.float32).copy()
        for _ in range(5):
            blur = (
                blur
                + np.roll(blur, 1, 0)
                + np.roll(blur, -1, 0)
                + np.roll(blur, 1, 1)
                + np.roll(blur, -1, 1)
            ) / 5.0
        hf = lum - blur
        scale_hf = float(np.percentile(np.abs(hf), 95))
        if scale_hf <= 1e-6:
            scale_hf = float(np.std(hf) + 1e-6)
        app = 1.0 + float(args.appearance_profile_strength) * hf / scale_hf
        app = np.clip(app, 1.0 - float(args.appearance_profile_clip), 1.0 + float(args.appearance_profile_clip)).astype(np.float32)
        app = app / max(float(app.mean()), 1e-6)
        appearance_profile = app
        print(
            f"  appearance profile: {appearance_profile.shape} mult "
            f"[{appearance_profile.min():.3f},{np.median(appearance_profile):.3f},{appearance_profile.max():.3f}] "
            f"from {args.appearance_profile_npz}",
            flush=True,
        )

    if "final_color_root_tip" in prior.files and prior["final_color_root_tip"].shape[1] >= 6:
        color6 = prior["final_color_root_tip"].astype(np.float32)
    elif "patch_color_root_tip" in prior.files and prior["patch_color_root_tip"].shape[1] >= 6:
        color6 = prior["patch_color_root_tip"].astype(np.float32)
    else:
        color6 = np.tile(np.array([[0.45, 0.32, 0.20, 0.48, 0.36, 0.24]], np.float32), (len(density), 1))

    ii, jj = np.meshgrid(np.arange(h), np.arange(h))
    ug = (ii + 0.5) / h
    vg = (jj + 0.5) / h
    pouwin = (np.cos(np.pi * (ug - 0.5)) * np.cos(np.pi * (vg - 0.5))).clip(1e-3, None).astype(np.float32)
    appearance_grid = None
    if appearance_profile is not None:
        grid_uv = np.stack([ug.ravel(), vg.ravel()], 1).astype(np.float32)
        appearance_grid = _bilinear(appearance_profile[..., None], grid_uv)[:, 0].reshape(h, h).astype(np.float32)
        appearance_grid = appearance_grid / max(float(appearance_grid.mean()), 1e-6)

    nf = int(cface.max()) + 1
    wp = np.zeros((len(rows), nf), np.float32)
    for ri, (_row, p) in enumerate(rows):
        fv = cface[p].ravel()
        wv = (pouwin * (valid[p] > 0.5)).ravel()
        ok = fv >= 0
        np.maximum.at(wp[ri], fv[ok], wv[ok])
    vsum = wp.sum(0)

    total_weight = 0.0
    for ri, (row, p) in enumerate(rows):
        cf = np.clip(cface[p], 0, nf - 1)
        others = np.maximum(vsum[cf] - wp[ri][cf], 0.0)
        share = pouwin if args.no_pou else pouwin / np.maximum(pouwin + others, 1e-6)
        total_weight += float((valid[p] * share * max(float(density[row]), 0.0)).sum())
    root_scale = (float(args.target_strands) / max(total_weight, 1e-6)) if args.target_strands > 0 else float(args.base_density)
    _mode = f"FORCED total={args.target_strands}" if args.target_strands > 0 else f"DENSITY-INTEGRAL base_density={args.base_density}"
    print(f"  root mode: {_mode} | density-integral natural count ~{int(total_weight):,} (x base_density); ~{int(total_weight)//max(len(rows),1):,} roots/patch", flush=True)

    rng = np.random.default_rng(args.seed)
    all_uv, all_row, all_patch, all_rf, all_rb, all_rootp, all_nr, all_t, all_b = [], [], [], [], [], [], [], [], []
    for ri, (row, p) in enumerate(rows):
        cf = np.clip(cface[p], 0, nf - 1)
        others = np.maximum(vsum[cf] - wp[ri][cf], 0.0)
        share = pouwin if args.no_pou else pouwin / np.maximum(pouwin + others, 1e-6)
        lam = root_scale * max(float(density[row]), 0.0) * valid[p] * share
        if appearance_grid is not None:
            lam = lam * appearance_grid
        if args.clump_density_contrast > 0:
            rname = patch_region[row] if row < len(patch_region) else "body"
            rid = reg_map.get(str(rname), -1)
            theta = direction_for_patch(prior, int(row), rid, args.dir_coh_gate)
            mult = clump_density_multiplier(
                surf[p],
                valid[p],
                pack["centers"].astype(np.float32)[p],
                tt[p],
                bb[p],
                theta,
                psize,
                args.target_arclen,
                args,
                rng,
            )
            lam = lam * mult
        base = np.floor(lam).astype(np.int32)
        extra = (rng.uniform(0, 1, lam.shape) < (lam - base)).astype(np.int32)
        nc = (base + extra).ravel()
        nz = np.nonzero(nc)[0]
        if len(nz) == 0:
            continue
        cells = np.repeat(nz, nc[nz])
        cj, ci = np.divmod(cells, h)
        uv = np.stack([(ci + rng.uniform(0, 1, len(cells))) / h, (cj + rng.uniform(0, 1, len(cells))) / h], 1).astype(np.float32)
        ti = np.clip((uv[:, 0] * h).astype(np.int32), 0, h - 1)
        tj = np.clip((uv[:, 1] * h).astype(np.int32), 0, h - 1)
        all_uv.append(uv)
        all_row.append(np.full(len(uv), row, np.int32))
        all_patch.append(np.full(len(uv), p, np.int32))
        all_rf.append(cface[p][tj, ti].astype(np.int32))
        all_rb.append(cbary[p][tj, ti].astype(np.float32))
        all_rootp.append(_bilinear(surf[p], uv).astype(np.float32))
        nr = unit(_bilinear(normg[p], uv).astype(np.float32))
        tr = tt[p][None].repeat(len(uv), 0)
        tr = tr - nr * (tr * nr).sum(1, keepdims=True)
        bad = np.linalg.norm(tr, axis=1) < 1e-5
        if bad.any():
            tr[bad] = bb[p]
        tr = unit(tr)
        br = unit(np.cross(nr, tr))
        all_nr.append(nr.astype(np.float32))
        all_t.append(tr.astype(np.float32))
        all_b.append(br.astype(np.float32))

    if not all_uv:
        raise RuntimeError("sampled zero roots")
    root_uv = np.concatenate(all_uv)
    root_row = np.concatenate(all_row)
    root_patch = np.concatenate(all_patch)
    root_face = np.concatenate(all_rf)
    root_bary = np.concatenate(all_rb)
    rootp = np.concatenate(all_rootp)
    nr = np.concatenate(all_nr)
    tr = np.concatenate(all_t)
    br = np.concatenate(all_b)

    backfill_n = 0
    if args.backfill_min_face_roots > 0:
        mesh_path = infer_mesh_path(pack)
        if not mesh_path or not os.path.exists(mesh_path):
            raise RuntimeError("--backfill_min_face_roots requires a valid mesh path")
        verts, faces = load_obj_tri_mesh(mesh_path)
        if len(faces) < nf:
            raise RuntimeError(f"mesh has {len(faces)} faces but pack references {nf}")

        counts = np.bincount(root_face[root_face >= 0], minlength=len(faces))
        if args.backfill_scope == "all":
            eligible = np.arange(len(faces), dtype=np.int64)
        else:
            eligible = np.nonzero(vsum > 0)[0].astype(np.int64)
        eligible = eligible[eligible < len(faces)]
        deficit = np.maximum(args.backfill_min_face_roots - counts[eligible], 0).astype(np.int32)
        fill_faces = np.repeat(eligible, deficit)
        if len(fill_faces):
            pts, bary = sample_face_roots(verts, faces, fill_faces, rng)
            fcent = verts[faces[fill_faces]].mean(1)
            centers = pack["centers"].astype(np.float32)[patch_ids]
            fill_row = nearest_rows_for_points(fcent, centers, row_ids).astype(np.int32)
            row_to_patch = np.full(len(prior["final_density"]), -1, np.int32)
            row_to_patch[row_ids] = patch_ids
            fill_patch = row_to_patch[fill_row]
            badp = fill_patch < 0
            if badp.any():
                fill_patch[badp] = patch_ids[0]

            tri = faces[fill_faces]
            fn = unit(np.cross(verts[tri[:, 1]] - verts[tri[:, 0]], verts[tri[:, 2]] - verts[tri[:, 0]])).astype(np.float32)
            ft = tt[fill_patch].astype(np.float32)
            ft = ft - fn * (ft * fn).sum(1, keepdims=True)
            bad = np.linalg.norm(ft, axis=1) < 1e-5
            if bad.any():
                alt = np.tile(np.array([[1.0, 0.0, 0.0]], np.float32), (bad.sum(), 1))
                near = np.abs((alt * fn[bad]).sum(1)) > 0.9
                alt[near] = np.array([0.0, 1.0, 0.0], np.float32)
                ft[bad] = alt - fn[bad] * (alt * fn[bad]).sum(1, keepdims=True)
            ft = unit(ft).astype(np.float32)
            fb = unit(np.cross(fn, ft)).astype(np.float32)

            root_uv = np.concatenate([root_uv, np.zeros((len(fill_faces), 2), np.float32)], 0)
            root_row = np.concatenate([root_row, fill_row], 0)
            root_patch = np.concatenate([root_patch, fill_patch.astype(np.int32)], 0)
            root_face = np.concatenate([root_face, fill_faces.astype(np.int32)], 0)
            root_bary = np.concatenate([root_bary, bary.astype(np.float32)], 0)
            rootp = np.concatenate([rootp, pts.astype(np.float32)], 0)
            nr = np.concatenate([nr, fn], 0)
            tr = np.concatenate([tr, ft], 0)
            br = np.concatenate([br, fb], 0)
            backfill_n = len(fill_faces)

    strand_profile_world_scale = 1.0
    if strand_profile is not None:
        # STAMPING route: copy the adopted p1 strand offsets directly.  This
        # bypasses z64 decode, z64 lift handling, comb, and clump.  Step2 only
        # supplies the macro flow angle; length is handled by the global/patch
        # rescale below.
        print("  local strand source: STRAND PROFILE STAMP (skip z64 codec decode)", flush=True)
        _pidx = strand_profile["tree"].query(root_uv)[1]
        local = strand_profile["offsets"][_pidx].astype(np.float32).copy()
        # Swatches built by build_canon_swatch are in normalized patch-local
        # units (root_uv over [0,1]^2). Convert that local unit to world units
        # using the geodesic patch size before applying the 2D crop footprint.
        # This replaces the old hidden target_arclen trick:
        #   old: target_arclen ~= median(raw_swatch_arc) * psize
        #   new: local *= psize, then scale_contract_length_exp handles crop/native
        strand_profile_world_scale = float(psize)
        local *= strand_profile_world_scale
        print(f"  strand profile unit scale: local_patch_unit * psize={strand_profile_world_scale:.6f}", flush=True)
        if opt_flow_delta is not None:
            _th = np.zeros(len(local), np.float32)
            _vr = (root_row >= 0) & (root_row < len(opt_flow_delta))
            _th[_vr] = opt_flow_delta[root_row[_vr]].astype(np.float32)
            _ct, _st = np.cos(_th), np.sin(_th)
            _lx, _ly = local[..., 0].copy(), local[..., 1].copy()
            local[..., 0] = _ct[:, None] * _lx - _st[:, None] * _ly
            local[..., 1] = _st[:, None] * _lx + _ct[:, None] * _ly
        clump_shift = np.zeros(len(local), np.float32)
    else:
        if z64_profile is not None:
            z_roots = _bilinear(z64_profile, root_uv).astype(np.float32)
        elif z64_map is not None:
            z_roots = np.empty((len(root_uv), 64), np.float32)
            for row in np.unique(root_row):
                m = root_row == row
                z_roots[m] = _bilinear(z64_map[int(row)], root_uv[m]).astype(np.float32)
        else:
            z_roots = z64[root_row]
        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if device == "auto":
            device = "cpu"
        ck = torch.load(args.codec_path, map_location=device, weights_only=False)
        cnorm = {k: torch.tensor(v, device=device, dtype=torch.float32) for k, v in ck["fur_norm"].items()}
        codec = StrandCodec(do_vae=False, decode_type="dir", scale_init=30.0, nr_verts_per_strand=256,
                             nr_values_to_decode=255, dim_per_value_decoded=3).to(device).eval()
        codec.load_state_dict(ck["state_dict"])
        # The upstream decoder keeps a few plain tensors instead of registered
        # buffers, so module.to(device) does not move them.
        if hasattr(codec.decoder, "t"):
            codec.decoder.t = codec.decoder.t.to(device)
        if hasattr(codec.decoder, "start_positions"):
            codec.decoder.start_positions = codec.decoder.start_positions.to(device)
        local = decode_z64(codec, cnorm, z_roots, args.decode_chunk, args.points, device)
        # The v6 upright codec stores normal displacement in local z.  V2's decoded
        # z64 can land on either sign; negative z goes under the dog mesh and renders
        # as bare skin/stipple.  `abs` keeps the learned lift magnitude while forcing
        # the strand outside the surface.
        if args.lift_mode == "external":
            # DECOUPLE standoff from z64 (parallels the direction decouple): z64 is not lift-equivariant
            # -- it bakes an absolute normal-rise that grow consumes too high. Keep ONLY z64's in-plane
            # shape (curl/bend/flow in x,y); ZERO the normal component here so the global arclen rescale
            # below only stretches the IN-PLANE shape. The explicit GT-calibrated lift is added back AFTER
            # the rescale (search "external lift add"), so lift_height is immune to the length scaling.
            local[..., 2] = 0.0
        else:
            if args.z_mode == "abs":
                local[..., 2] = np.abs(local[..., 2])
            elif args.z_mode == "clamp":
                local[..., 2] = np.maximum(local[..., 2], 0.0)
            local[..., 2] *= float(args.normal_lift_scale)
            if "patch_lift_scale" in prior.files:                                           # z64-mode region lift multiplier (external mode applies it after the rescale)
                lift_scale = prior["patch_lift_scale"].astype(np.float32)
                if len(lift_scale) == len(density):
                    local[..., 2] *= np.clip(lift_scale[root_row], 0.25, 2.5)[:, None]

    if strand_profile is None and args.direction_mode == "external":
        def add_direction_profile(theta_base, uv):
            if dir_profile_vec is None or float(args.direction_profile_strength) == 0.0:
                # every caller indexes the result per root (c[:, None]), so a scalar base must still
                # broadcast to one angle per root here -- same shape contract as the path below.
                return (np.full(len(uv), float(theta_base), np.float32)
                        if np.ndim(theta_base) == 0 else np.asarray(theta_base, np.float32))
            if np.ndim(theta_base) == 0:
                theta = np.full(len(uv), float(theta_base), np.float32)
            else:
                theta = np.asarray(theta_base, np.float32).copy()
            dvp = _bilinear(dir_profile_vec, uv).astype(np.float32)
            dn = np.linalg.norm(dvp, axis=1)
            good = dn > 1e-6
            if dir_profile_conf is not None:
                dc = _bilinear(dir_profile_conf[..., None], uv)[:, 0].astype(np.float32)
                good &= dc >= float(args.direction_profile_conf_gate)
            if good.any():
                tp = np.arctan2(dvp[good, 1], dvp[good, 0]).astype(np.float32)
                theta[good] = theta[good] + float(args.direction_profile_strength) * axial_delta(tp, dir_profile_theta0).astype(np.float32)
            return theta

        for row in np.unique(root_row):
            m = root_row == row
            rname = patch_region[row] if row < len(patch_region) else "body"
            rid = reg_map.get(str(rname), -1)
            xy = local[m, :, :2].copy()
            if opt_flow_delta is not None and int(row) < len(opt_flow_delta):
                theta = add_direction_profile(float(opt_flow_delta[int(row)]), root_uv[m])
                c, s = np.cos(theta), np.sin(theta)
                local[m, :, 0] = c[:, None] * xy[:, :, 0] - s[:, None] * xy[:, :, 1]
                local[m, :, 1] = s[:, None] * xy[:, :, 0] + c[:, None] * xy[:, :, 1]
                continue
            fallback_theta = direction_for_patch(prior, int(row), rid, args.dir_coh_gate)
            if dir_vec_map is not None and int(row) < len(dir_vec_map):
                dv = _bilinear(dir_vec_map[int(row)], root_uv[m]).astype(np.float32)
                dn = np.linalg.norm(dv, axis=1)
                good = dn > 1e-6
                if dir_vec_map_conf is not None:
                    dc = _bilinear(dir_vec_map_conf[int(row)][..., None], root_uv[m])[:, 0].astype(np.float32)
                    good &= dc >= float(args.dir_coh_gate)
                theta = np.full(len(dv), fallback_theta, np.float32)
                theta[good] = np.arctan2(dv[good, 1], dv[good, 0]).astype(np.float32)
                theta = add_direction_profile(theta, root_uv[m])
                c, s = np.cos(theta), np.sin(theta)
                local[m, :, 0] = c[:, None] * xy[:, :, 0] - s[:, None] * xy[:, :, 1]
                local[m, :, 1] = s[:, None] * xy[:, :, 0] + c[:, None] * xy[:, :, 1]
            else:
                theta = add_direction_profile(fallback_theta, root_uv[m])
                c, s = np.cos(theta), np.sin(theta)
                local[m, :, 0] = c[:, None] * xy[:, :, 0] - s[:, None] * xy[:, :, 1]
                local[m, :, 1] = s[:, None] * xy[:, :, 0] + c[:, None] * xy[:, :, 1]
    elif strand_profile is None:
        print("  direction source: latent absdir (skip measured-direction rotate-back)", flush=True)

    if strand_profile is None and args.comb_blend > 0:
        blend = float(np.clip(args.comb_blend, 0.0, 1.0))
        tline = np.linspace(0.0, 1.0, local.shape[1], dtype=np.float32)[None, :]
        # Direction has already been rotated into local x/y.  The codec still
        # contributes lift/curl, while this stroke target guarantees that body
        # patches read as fur strokes rather than random stipple.
        target_chord = max(float(args.target_arclen), 1e-6) * float(args.comb_tangent_frac)
        stroke = local.copy()
        stroke[:, :, 0] = tline * target_chord
        stroke[:, :, 1] = stroke[:, :, 1] * float(args.comb_cross_damp)
        local = (1.0 - blend) * local + blend * stroke

    if strand_profile is None:
        clump_shift = apply_flow_clumping(
            local,
            rootp,
            root_row,
            root_patch,
            tr,
            br,
            pack["centers"].astype(np.float32),
            tt,
            bb,
            patch_region,
            reg_map,
            prior,
            psize,
            args.target_arclen,
            args,
            rng,
        )

    strands = rootp[:, None] + local[..., 0:1] * tr[:, None] + local[..., 1:2] * br[:, None] + local[..., 2:3] * nr[:, None]
    arc = np.linalg.norm(np.diff(strands, axis=1), axis=2).sum(1)
    length_raw_ratio = np.ones(len(density), np.float32)
    length_ratio = np.ones(len(density), np.float32)
    need_length_contract = args.scale_contract != "none" and float(args.scale_contract_length_exp) != 0.0
    need_length_prior = ("patch_length_ratio" in prior.files) or (opt_length is not None and len(opt_length) == len(density))
    need_global_arclen = float(args.target_arclen) > 0.0
    if len(arc) and (need_global_arclen or need_length_prior or need_length_contract):
        med = float(np.median(arc))
        if med > 1e-9:
            scale = float(args.target_arclen) / med if need_global_arclen else 1.0
            smooth_centers = pack["centers"].astype(np.float32)[patch_ids] if "centers" in pack.files else pack["surf"].mean(axis=(1, 2))[patch_ids].astype(np.float32)
            if need_global_arclen or need_length_prior:
                length_raw_ratio, length_ratio = build_length_ratios(
                    arc,
                    root_row,
                    len(density),
                    patch_region,
                    row_ids,
                    smooth_centers,
                    args.length_mode,
                    args.length_smooth_k,
                    args.length_smooth_iters,
                    args.length_region_blend,
                    args.length_ratio_clip,
                )
            else:
                # Pure footprint scaling: preserve the adopted/decoded swatch's
                # own length distribution and multiply only by the 2D->3D
                # scale contract. Do not normalize it back to a hidden global
                # median.
                length_raw_ratio = np.ones(len(density), np.float32)
                length_ratio = np.ones(len(density), np.float32)
            if "patch_length_ratio" in prior.files:
                opt_len = prior["patch_length_ratio"].astype(np.float32)
                if len(opt_len) == len(density):
                    length_ratio = length_ratio * np.clip(opt_len, 1.0 / max(float(args.length_ratio_clip), 1.0), max(float(args.length_ratio_clip), 1.0))
                    root_med = float(np.median(length_ratio[root_row])) if len(root_row) else 1.0
                    if root_med > 1e-9:
                        length_ratio = length_ratio / root_med
            if opt_length is not None and len(opt_length) == len(density):
                length_ratio = length_ratio * np.clip(opt_length, 1.0 / max(float(args.length_ratio_clip), 1.0), max(float(args.length_ratio_clip), 1.0))
                root_med = float(np.median(length_ratio[root_row])) if len(root_row) else 1.0
                if root_med > 1e-9:
                    length_ratio = length_ratio / root_med
            if args.scale_contract != "none" and float(args.scale_contract_length_exp) != 0.0:
                # This is intentionally absolute, not median-normalized away:
                # patch_length_remap encodes how the same adopted 3D patch should
                # be resized when the real image patch occupies fewer/more pixels
                # than the model's reference crop.  Keep this exponent at 0 when
                # the coat length is already visually correct and only root
                # density / fiber radius need scale correction.
                length_ratio = length_ratio * np.power(scale_contract.astype(np.float32), float(args.scale_contract_length_exp)).astype(np.float32)
            # Replace the noisy raw patch median length ratio with the stabilized
            # diffusion-derived ratio.  The scalar `scale` only fixes world units.
            root_len_scale = scale * length_ratio[root_row] / np.maximum(length_raw_ratio[root_row], 1e-6)
            strands = strands[:, :1] + (strands - strands[:, :1]) * root_len_scale[:, None, None]
            arc = np.linalg.norm(np.diff(strands, axis=1), axis=2).sum(1)
        else:
            scale = 1.0
    else:
        scale = 1.0
    if args.lift_mode == "external":
        # external lift add: NOW the in-plane shape carries the final (rescaled) arclength and z is still 0,
        # so (strands-root) is purely in-plane. Add the explicit GT-calibrated standoff along the surface
        # normal: rise = lift_height * (normalized in-plane arclength). Immune to the arclen rescale above.
        inpl = np.linalg.norm(np.diff(strands - strands[:, :1], axis=1), axis=2)
        sprof = np.concatenate([np.zeros((len(strands), 1), np.float32), np.cumsum(inpl, axis=1)], axis=1)
        sprof = sprof / np.maximum(sprof[:, -1:], 1e-9)
        lh = np.full(len(strands), float(args.lift_height), np.float32)
        if "patch_lift_scale" in prior.files:                                               # region-level explicit lift multiplier (aggregated like length/density)
            lsc = prior["patch_lift_scale"].astype(np.float32)
            if len(lsc) == len(density):
                lh = lh * np.clip(lsc[root_row], 0.25, 2.5)
        if opt_lift is not None and len(opt_lift) == len(density):
            lh = lh * np.clip(opt_lift[root_row], 0.25, 2.5)
        strands = strands + (lh[:, None] * sprof)[..., None] * nr[:, None]
    if args.root_sink > 0:
        strands = strands - nr[:, None, :] * float(args.root_sink)
    # Recompute after external lift/root sink so the saved QA metadata reports
    # the actual 3D strand length, not the pre-lift in-plane length.
    arc = np.linalg.norm(np.diff(strands, axis=1), axis=2).sum(1)

    # ---- torso-only loft (the "fluffy" look) -----------------------------------
    # Add a gentle extra normal rise ONLY on torso regions, proportional to each
    # strand's final arclength (longer fur lofts more) and hard-capped so it never
    # blows up. Legs/face/tail/ears are left at their existing lift. This is the
    # honest way to recover V10's volume without over-lengthening the coat.
    if float(args.torso_lift_k) > 0.0:
        torso_names = {s.strip() for s in str(args.torso_lift_regions).split(",") if s.strip()}
        root_reg = np.full(len(strands), "body", dtype=object)
        rr_ok = (root_row >= 0) & (root_row < len(patch_region))
        root_reg[rr_ok] = patch_region[root_row[rr_ok]].astype(str)
        torso = np.array([r in torso_names for r in root_reg], dtype=bool)
        frac = float(args.torso_lift_frac)
        if args.torso_lift_mode == "smooth":
            # Smooth the binary region label over the patch centers. The field is
            # continuous across body/face/leg boundaries and constant for all
            # roots owned by a patch, so it cannot introduce per-strand speckle.
            patch_pos = prior["patch_pos"].astype(np.float32)
            patch_torso = np.array([r in torso_names for r in patch_region], np.float32)
            pcount = len(patch_torso)
            k = min(max(int(args.torso_lift_smooth_k), 1), max(pcount - 1, 1))
            d2 = ((patch_pos[:, None, :] - patch_pos[None, :, :]) ** 2).sum(axis=2)
            nn = np.argpartition(d2, kth=k, axis=1)[:, : k + 1]
            patch_weight = patch_torso.copy()
            for _ in range(max(int(args.torso_lift_smooth_iters), 0)):
                patch_weight = 0.35 * patch_weight + 0.65 * patch_weight[nn].mean(axis=1)
            root_weight = np.zeros(len(strands), np.float32)
            root_weight[rr_ok] = patch_weight[root_row[rr_ok]]
            root_weight *= np.clip(frac, 0.0, 1.0)
            sel = root_weight > 1e-4
        else:
            # Legacy mode: only a random fraction of torso strands loft.
            sel = torso
            root_weight = sel.astype(np.float32)
            keep = np.random.RandomState(int(args.torso_lift_seed)).random(len(strands)) < max(frac, 0.0)
            if frac < 1.0:
                sel = torso & keep
                root_weight = sel.astype(np.float32)
        amt = np.clip(float(args.torso_lift_k) * arc, 0.0, float(args.torso_lift_cap)).astype(np.float32)
        amt = amt * root_weight
        if amt.max() > 0:
            seg = np.linalg.norm(np.diff(strands, axis=1), axis=2)
            prof = np.concatenate([np.zeros((len(strands), 1), np.float32), np.cumsum(seg, axis=1)], axis=1)
            prof = prof / np.maximum(prof[:, -1:], 1e-9)          # 0 at root -> 1 at tip
            strands = strands + (amt[:, None] * prof)[..., None] * nr[:, None]
            arc = np.linalg.norm(np.diff(strands, axis=1), axis=2).sum(1)
            print(f"torso_lift: mode={args.torso_lift_mode} regions={sorted(torso_names)} k={args.torso_lift_k} cap={args.torso_lift_cap} "
                  f"frac={args.torso_lift_frac} lifted={int(sel.sum())}/{int(torso.sum())} torso roots "
                  f"({int(sel.sum())}/{len(torso)} total), rise med/p90="
                  f"{float(np.median(amt[sel])) if sel.any() else 0.0:.5f}/"
                  f"{float(np.percentile(amt[sel],90)) if sel.any() else 0.0:.5f}")

    c6 = color6[root_row]
    color = np.clip(c6[:, :3], 0, 1).astype(np.float32)
    color_tip = np.clip(c6[:, 3:6] if c6.shape[1] >= 6 else color * 1.04, 0, 1).astype(np.float32)
    appearance_root = None
    if appearance_profile is not None:
        appearance_root = _bilinear(appearance_profile[..., None], root_uv)[:, 0].astype(np.float32)
        color = np.clip(color * appearance_root[:, None], 0, 1).astype(np.float32)
        color_tip = np.clip(color_tip * appearance_root[:, None], 0, 1).astype(np.float32)
    radius_mode = "fixed"
    radius_arc_ratio = float(args.radius_arc_ratio)
    if radius_arc_ratio <= 0 and args.radius_profile_npz:
        prof = np.load(args.radius_profile_npz, allow_pickle=True)
        if "strands" not in prof.files or "radius" not in prof.files:
            raise KeyError(f"--radius_profile_npz must contain strands and radius: {args.radius_profile_npz}")
        ps = prof["strands"].astype(np.float32)
        pr = prof["radius"].astype(np.float32)
        parc = np.linalg.norm(np.diff(ps, axis=1), axis=2).sum(1)
        med_arc = float(np.median(parc)) if len(parc) else 0.0
        med_rad = float(np.median(pr)) if len(pr) else 0.0
        if med_arc <= 1e-9 or med_rad <= 0:
            raise ValueError(f"bad radius profile medians: radius={med_rad} arc={med_arc} from {args.radius_profile_npz}")
        radius_arc_ratio = med_rad / med_arc
    if radius_arc_ratio > 0:
        radius_mode = "arc_ratio"
        radius = np.maximum(radius_arc_ratio * arc, 1e-7).astype(np.float32)
    else:
        radius = np.full(len(strands), float(args.radius) * float(prior["global_radius_scale"]) if "global_radius_scale" in prior.files else float(args.radius), np.float32)
    if args.scale_contract != "none" and float(args.scale_contract_radius_exp) != 0.0:
        radius = radius * np.power(scale_contract[root_row], float(args.scale_contract_radius_exp)).astype(np.float32)
        radius_mode = f"{radius_mode}+scale_contract"

    (
        strands,
        radius,
        root_uv,
        root_face,
        root_bary,
        root_patch,
        root_row,
        color,
        color_tip,
        region_detail_scale,
        region_detail_keep_prob,
        region_detail_summary,
    ) = apply_region_detail_policy(
        strands,
        radius,
        root_uv,
        root_face,
        root_bary,
        root_patch,
        root_row,
        color,
        color_tip,
        patch_region,
        args,
    )
    if args.region_detail_policy != "none":
        arc = np.linalg.norm(np.diff(strands, axis=1), axis=2).sum(1)

    (
        strands,
        radius,
        tail_length_summary,
        tail_length_target,
        tail_length_roots,
    ) = apply_tail_length_policy(
        strands,
        radius,
        root_row,
        patch_region,
        args,
    )
    if args.tail_length_policy != "none":
        arc = np.linalg.norm(np.diff(strands, axis=1), axis=2).sum(1)
        print(f"  tail_length={args.tail_length_policy} | {tail_length_summary}", flush=True)

    # ---- soft per-region maximum length (realistic fur depth) -------------------
    # Per-patch length ratios (Step 2, silhouette refinement) can leave a long tail of very long
    # strands that read as spikes; compress the length beyond a per-region cap, keeping the shape.
    if args.length_soft_caps:
        caps = {}
        for tok in args.length_soft_caps.split(","):
            name, val = tok.split(":")
            caps[name.strip()] = float(val)
        rr_ok = (root_row >= 0) & (root_row < len(patch_region))
        root_reg = np.full(len(strands), "", dtype=object)
        root_reg[rr_ok] = patch_region[root_row[rr_ok]].astype(str)
        cap = np.array([caps.get(r, np.inf) for r in root_reg], np.float64)
        over = arc > cap
        new_arc = arc.astype(np.float64).copy()
        new_arc[over] = cap[over] + (arc[over] - cap[over]) * float(args.length_soft_cap_slope)
        fac = (new_arc / np.maximum(arc, 1e-9)).astype(np.float32)
        strands = strands[:, :1] + (strands - strands[:, :1]) * fac[:, None, None]
        arc = np.linalg.norm(np.diff(strands, axis=1), axis=2).sum(1)
        print("  length_soft_caps: " + "; ".join(
            f"{r}: {int((over & (root_reg == r)).sum())}/{int((root_reg == r).sum())} over {c * 100:.1f} cm"
            for r, c in caps.items()), flush=True)

    op = os.path.join(args.out, "surfpack_v2fur.npz")
    np.savez(
        op,
        strands=strands.astype(np.float32),
        root_uv=root_uv.astype(np.float32),
        color=color,
        color_tip=color_tip,
        radius=radius,
        root_face=root_face.astype(np.int32),
        root_bary=root_bary.astype(np.float32),
        root_patch=root_patch.astype(np.int32),
        root_prior_row=root_row.astype(np.int32),
        v2_prior=np.asarray(args.prior),
        patch_params=np.asarray(args.patch_params),
        strand_profile_npz=np.asarray(os.path.abspath(args.strand_profile_npz) if args.strand_profile_npz else ""),
        strand_profile_mode=np.asarray("direct_3d_patch" if strand_profile is not None else ""),
        strand_profile_canonical_angle=np.float32(strand_profile_angle),
        strand_profile_uv_crop=(strand_profile["crop"].astype(np.float32) if strand_profile is not None else np.array([0.0, 0.0, 1.0, 1.0], np.float32)),
        strand_profile_lift_scale=np.float32(args.strand_profile_lift_scale),
        z64_profile_npz=np.asarray(os.path.abspath(args.z64_profile_npz) if args.z64_profile_npz else ""),
        direction_profile_npz=np.asarray(os.path.abspath(args.direction_profile_npz) if args.direction_profile_npz else ""),
        direction_profile_strength=np.float32(args.direction_profile_strength),
        direction_profile_theta0=np.float32(dir_profile_theta0),
        appearance_profile_npz=np.asarray(os.path.abspath(args.appearance_profile_npz) if args.appearance_profile_npz else ""),
        appearance_profile_strength=np.float32(args.appearance_profile_strength),
        appearance_profile_clip=np.float32(args.appearance_profile_clip),
        source_pack=np.asarray(pack_path),
        mesh_path=np.asarray(infer_mesh_path(pack)),
        target_strands=np.int32(args.target_strands),
        root_density_scale=np.float32(root_scale),
        scale_contract=np.asarray(args.scale_contract),
        scale_contract_source=np.asarray(scale_contract_source),
        scale_contract_clip=np.float32(args.scale_contract_clip),
        scale_contract_density_exp=np.float32(args.scale_contract_density_exp),
        scale_contract_length_exp=np.float32(args.scale_contract_length_exp),
        scale_contract_radius_exp=np.float32(args.scale_contract_radius_exp),
        strand_profile_world_scale=np.float32(strand_profile_world_scale),
        patch_scale_contract=scale_contract.astype(np.float32),
        target_arclen=np.float32(args.target_arclen),
        direction_mode=np.asarray(args.direction_mode),
        arclen_median=np.float32(np.median(arc) if len(arc) else 0.0),
        arclen_scale=np.float32(scale),
        length_mode=np.asarray(args.length_mode),
        length_raw_ratio=length_raw_ratio.astype(np.float32),
        length_ratio=length_ratio.astype(np.float32),
        patch_lift_scale=(prior["patch_lift_scale"].astype(np.float32) if "patch_lift_scale" in prior.files else np.ones(len(density), np.float32)),
        patch_length_ratio=(prior["patch_length_ratio"].astype(np.float32) if "patch_length_ratio" in prior.files else np.ones(len(density), np.float32)),
        length_smooth_k=np.int32(args.length_smooth_k),
        length_smooth_iters=np.int32(args.length_smooth_iters),
        length_region_blend=np.float32(args.length_region_blend),
        length_ratio_clip=np.float32(args.length_ratio_clip),
        radius_mode=np.asarray(radius_mode),
        radius_arc_ratio=np.float32(radius_arc_ratio),
        radius_profile_npz=np.asarray(os.path.abspath(args.radius_profile_npz) if args.radius_profile_npz else ""),
        backfill_roots=np.int32(backfill_n),
        backfill_min_face_roots=np.int32(args.backfill_min_face_roots),
        backfill_scope=np.asarray(args.backfill_scope),
        comb_blend=np.float32(args.comb_blend),
        lift_mode=np.asarray(args.lift_mode), lift_height=np.float32(args.lift_height),
        z_mode=np.asarray(args.z_mode), normal_lift_scale=np.float32(args.normal_lift_scale),
        torso_lift_mode=np.asarray(args.torso_lift_mode),
        torso_lift_k=np.float32(args.torso_lift_k),
        torso_lift_cap=np.float32(args.torso_lift_cap),
        torso_lift_frac=np.float32(args.torso_lift_frac),
        torso_lift_smooth_k=np.int32(args.torso_lift_smooth_k),
        torso_lift_smooth_iters=np.int32(args.torso_lift_smooth_iters),
        comb_tangent_frac=np.float32(args.comb_tangent_frac),
        comb_cross_damp=np.float32(args.comb_cross_damp),
        clump_strength=np.float32(args.clump_strength),
        clump_size=np.float32(args.clump_size),
        clump_flow_aspect=np.float32(args.clump_flow_aspect),
        clump_along_frac=np.float32(args.clump_along_frac),
        clump_tip_frac=np.float32(args.clump_tip_frac),
        clump_power=np.float32(args.clump_power),
        clump_jitter=np.float32(args.clump_jitter),
        clump_density_contrast=np.float32(args.clump_density_contrast),
        clump_density_sigma=np.float32(args.clump_density_sigma),
        clump_density_min=np.float32(args.clump_density_min),
        clump_density_max=np.float32(args.clump_density_max),
        clump_shift_median=np.float32(np.median(clump_shift) if len(clump_shift) else 0.0),
        clump_shift_p90=np.float32(np.percentile(clump_shift, 90) if len(clump_shift) else 0.0),
        root_sink=np.float32(args.root_sink),
        region_detail_policy=np.asarray(args.region_detail_policy),
        region_detail_summary=np.asarray(region_detail_summary),
        region_flow_policy=np.asarray(args.region_flow_policy),
        region_flow_summary=np.asarray(region_flow_summary),
        region_detail_scale=region_detail_scale.astype(np.float32),
        length_soft_caps=np.asarray(args.length_soft_caps),
        length_soft_cap_slope=np.float32(args.length_soft_cap_slope),
        region_detail_keep_prob=region_detail_keep_prob.astype(np.float32),
        region_detail_tail_scale=np.float32(args.region_detail_tail_scale),
        region_detail_tail_keep=np.float32(args.region_detail_tail_keep),
        region_detail_tail_body_cap=np.float32(args.region_detail_tail_body_cap),
        region_detail_ear_scale=np.float32(args.region_detail_ear_scale),
        region_detail_ear_keep=np.float32(args.region_detail_ear_keep),
        region_detail_ear_body_cap=np.float32(args.region_detail_ear_body_cap),
        region_detail_leg_scale=np.float32(args.region_detail_leg_scale),
        region_detail_leg_keep=np.float32(args.region_detail_leg_keep),
        region_detail_leg_body_cap=np.float32(args.region_detail_leg_body_cap),
        region_detail_leg_feather=np.float32(args.region_detail_leg_feather),
        region_detail_paw_scale=np.float32(args.region_detail_paw_scale),
        region_detail_paw_keep=np.float32(args.region_detail_paw_keep),
        region_detail_paw_body_cap=np.float32(args.region_detail_paw_body_cap),
        tail_length_policy=np.asarray(args.tail_length_policy),
        tail_length_regions=np.asarray(args.tail_length_regions),
        tail_length_body_cap=np.float32(args.tail_length_body_cap),
        tail_length_summary=np.asarray(tail_length_summary),
        tail_length_target=np.float32(tail_length_target),
        tail_length_roots=np.int32(tail_length_roots),
    )
    print(f"V2 GROW -> {op}", flush=True)
    print(
        f"  patches={len(rows)} roots={len(strands)} root_scale={root_scale:.4f} "
        f"arclen_med={np.median(arc):.5f} radius_med={np.median(radius):.6f} "
        f"radius/arc={np.median(radius) / max(float(np.median(arc)), 1e-9):.4f} "
        f"radius_mode={radius_mode} backfill={backfill_n} color_mean={color.mean(0)}",
        flush=True,
    )
    if args.region_detail_policy != "none":
        print(f"  region_detail={args.region_detail_policy} | {region_detail_summary}", flush=True)
    if args.clump_strength > 0:
        print(
            f"  clump strength={args.clump_strength:.3f} size={args.clump_size:.5f} "
            f"shift_med={np.median(clump_shift):.5f} shift_p90={np.percentile(clump_shift, 90):.5f}",
            flush=True,
        )
    mesh_path = infer_mesh_path(pack)
    if mesh_path:
        print(f"  mesh={mesh_path}", flush=True)


if __name__ == "__main__":
    main()
