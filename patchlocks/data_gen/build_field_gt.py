"""Step-1 (field-first) GT builder -> SMOOTHED REALIZED-FIELD GT (NOT the generator's analytic field).
For every rendered sample we ESTIMATE the fur fields from the saved (realized) strands by binning
per-strand attributes to a grid + weighted Gaussian smoothing. This is the right target: the field is the
smoothed MEAN; the per-strand stochastic residual (which made per-texel latent unpredictable) averages
away. NOTE: this is used for BOTH the v3 (stage1) and v4 batches -- it is an estimate from realized
strands, not the exact generator field (see fur_fields.py for true analytic-field export, currently unused).
Saves field_gt_16.npz / field_gt_32.npz with: lambda_map, direction_map(3), length_mu, curl_mu,
color_map(3), radius_mu, valid_mask.
Run:  python patchlocks/data_gen/build_field_gt.py patchlocks/datasets/stage1/renders
"""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2

ap = argparse.ArgumentParser(); ap.add_argument("renders")
a = ap.parse_args()
dirs = sorted(d for d in glob.glob(a.renders + "/*") if os.path.exists(f"{d}/patch.npz"))
print(f"{len(dirs)} patches", flush=True)


def strand_attrs(S, color, radius):
    """Per-strand: root-flow direction (unit3), arc length, curl ratio, color(3), radius."""
    L = S.shape[1]; kr = max(1, L // 6)
    d = S[:, kr] - S[:, 0]; d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)  # root tangent
    seg = np.linalg.norm(S[:, 1:] - S[:, :-1], axis=2); arc = seg.sum(1)                          # arc length
    chord = np.linalg.norm(S[:, -1] - S[:, 0], axis=1)
    curl = 1.0 - chord / np.maximum(arc, 1e-9)                                                    # 0=straight
    tip_off = (S[:, -1, :2] - S[:, 0, :2])                                                        # net lateral (tip-root) -> attractor/clump pull
    return (d.astype(np.float32), arc.astype(np.float32), curl.astype(np.float32),
            color.astype(np.float32), radius.astype(np.float32), tip_off.astype(np.float32))


def smooth_field(uv, attrs, res, sigma):
    """Weighted bin+blur: returns dict of smoothed maps + count map. attrs: dict name->(N,C)."""
    ti = np.clip((uv[:, 0] * res).astype(int), 0, res - 1)
    tj = np.clip((uv[:, 1] * res).astype(int), 0, res - 1)
    cnt = np.zeros((res, res), np.float32); np.add.at(cnt, (tj, ti), 1.0)
    k = int(max(3, round(sigma * 4)) | 1)
    cnt_b = cv2.GaussianBlur(cnt, (k, k), sigma)
    out = {"lambda_map": cnt_b.copy()}                                   # smoothed density (expected count proxy)
    for name, v in attrs.items():
        C = v.shape[1]; acc = np.zeros((res, res, C), np.float32)
        np.add.at(acc, (tj, ti), v)
        accb = cv2.GaussianBlur(acc, (k, k), sigma)
        if accb.ndim == 2: accb = accb[..., None]                        # cv2 drops singleton channel
        out[name] = accb / np.maximum(cnt_b[..., None], 1e-6)            # weighted mean -> (res,res,C)
    valid = (cnt_b > 0.05 * cnt_b.max()).astype(np.float32)
    out["valid_mask"] = valid
    return out


ok = 0
for d in dirs:
    p = np.load(f"{d}/patch.npz", allow_pickle=True)
    S = p["strands"].astype(np.float32); uv = p["root_uv"].astype(np.float32)
    color = p["color"].astype(np.float32)
    radius = p["radius"].astype(np.float32).reshape(-1)
    if len(radius) == 1: radius = np.full(len(S), float(radius[0]), np.float32)
    dvec, arc, curl, col, rad, tip_off = strand_attrs(S, color, radius)
    attrs = {"direction_map": dvec, "length_mu": arc[:, None], "curl_mu": curl[:, None],
             "color_map": col, "radius_mu": rad[:, None], "flow2d": tip_off}
    for res, sigma in ((16, 0.9), (32, 1.3), (64, 1.8)):
        f = smooth_field(uv, attrs, res, sigma)
        dn = f["direction_map"]; f["direction_map"] = dn / np.maximum(np.linalg.norm(dn, axis=2, keepdims=True), 1e-9)
        np.savez(f"{d}/field_gt_{res}.npz", **{k: v.astype(np.float32) for k, v in f.items()})
    ok += 1
    if ok % 150 == 0: print(f"  {ok}/{len(dirs)}", flush=True)

z = np.load(f"{dirs[0]}/field_gt_32.npz")
print(f"wrote field_gt_16/32 for {ok} patches")
print("keys/shapes:", {k: z[k].shape for k in z.files})
print(f"lambda mean {z['lambda_map'].mean():.2f} max {z['lambda_map'].max():.2f} | length mean {z['length_mu'][z['valid_mask']>.5].mean():.3f} | "
      f"curl mean {z['curl_mu'][z['valid_mask']>.5].mean():.3f} | valid {z['valid_mask'].mean():.3f}")
