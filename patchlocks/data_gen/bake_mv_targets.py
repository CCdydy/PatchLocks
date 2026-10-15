"""Bake the SHARED (view-independent) V13 multi-view training targets ONCE per render dir.

V13 keeps the 4-view renders intact (do NOT use explode_singleview_rotaug.py, which collapses
4 -> 4 single-view dirs and kills cross-view correspondence). Each renders_mv/<dir> already has:
  patch.npz (the single shared 3D fur patch, root_uv frame), cameras.json (4 views), rgb_0..3.png.
The codec z64 target and the smoothed field GT are functions of patch.npz ONLY (keyed on root_uv),
so they are SHARED across all 4 views and are baked exactly once per dir -- leaving every view's
rgb + cameras.json untouched.

This is the canonical bake_latent_batch.py (z64 + density + mask) FUSED with build_field_gt.py
(lambda/direction/length/curl/color/radius/flow2d + valid_mask) so a single pass over each dir
writes everything train_v11_clean.py needs:
    <dir>/<out-name>            -> keys: target (H,H,64), density (H,H,1), mask (H,H)
    <dir>/field_gt_{16,32,64}.npz -> keys: lambda_map, direction_map(3), length_mu(1), curl_mu(1),
                                           color_map(3), radius_mu(1), flow2d(2), valid_mask

Run (difflocks env, from repo root):
  python patchlocks/data_gen/bake_mv_targets.py patchlocks/datasets/v11_longfur/renders_mv \
    --codec patchlocks/models/fur_strand_codec_v6upright.pt \
    --out-name target_latent64_v6codec.npz \
    --H 64 --limit 20            # drop --limit to bake all dirs

NOTE on H: train_v11_clean.py loads <out-name>["target"] at --res and requires field_gt_{res}.npz.
The default res is 64, so the codec latent is baked at H=64; field GT is always written at 16/32/64.
"""
import os, sys, glob, json, time, argparse
sys.path.insert(0, os.getcwd())
import numpy as np, cv2, torch
from models.strand_codec import StrandCodec, normalize_gt_data
from utils.strand_util import compute_dirs

torch.set_grad_enabled(False)


# ----- codec z64 latent (mirrors patchlocks/models/bake_latent_batch.py) -----
def resample(s, M=256):
    seg = np.linalg.norm(np.diff(s, axis=0), axis=1); cl = np.concatenate([[0], np.cumsum(seg)])
    tt = np.linspace(0, cl[-1] + 1e-9, M)
    return np.stack([np.interp(tt, cl, s[:, i]) for i in range(3)], 1)


def bake_latent(D, codec, norm, H, out_name, straight=False):
    p = np.load(f"{D}/patch.npz", allow_pickle=True)
    S = p["strands"].astype(np.float32); uv = p["root_uv"].astype(np.float32)
    S256 = np.stack([resample(s) for s in S]).astype(np.float32); S256 = S256 - S256[:, :1]
    if straight:                                                                    # ablation: drop curl, keep dir+len+elevation
        S256 = (np.linspace(0, 1, S256.shape[1], dtype=np.float32)[None, :, None] * S256[:, -1:, :]).astype(np.float32)
    pos = torch.tensor(S256, device="cuda")
    gt = {"strand_positions": pos, "strand_directions": compute_dirs(pos, append_last_dir=False)}
    z = codec.encoder(normalize_gt_data(gt, norm))["z"].cpu().numpy()
    ti = np.clip((uv[:, 0] * (H - 1)).round().astype(int), 0, H - 1)
    tj = np.clip((uv[:, 1] * (H - 1)).round().astype(int), 0, H - 1)
    T = np.zeros((H, H, 64)); cnt = np.zeros((H, H))
    np.add.at(T, (tj, ti), z); np.add.at(cnt, (tj, ti), 1.0); occ = cnt > 0; T[occ] /= cnt[occ, None]
    np.savez(f"{D}/{out_name}", target=T.astype(np.float32), density=cnt[..., None].astype(np.float32), mask=occ)


# ----- smoothed realized-field GT (mirrors patchlocks/data_gen/build_field_gt.py) -----
def strand_attrs(S, color, radius):
    L = S.shape[1]; kr = max(1, L // 6)
    d = S[:, kr] - S[:, 0]; d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    seg = np.linalg.norm(S[:, 1:] - S[:, :-1], axis=2); arc = seg.sum(1)
    chord = np.linalg.norm(S[:, -1] - S[:, 0], axis=1)
    curl = 1.0 - chord / np.maximum(arc, 1e-9)
    tip_off = (S[:, -1, :2] - S[:, 0, :2])
    return (d.astype(np.float32), arc.astype(np.float32), curl.astype(np.float32),
            color.astype(np.float32), radius.astype(np.float32), tip_off.astype(np.float32))


def smooth_field(uv, attrs, res, sigma):
    ti = np.clip((uv[:, 0] * res).astype(int), 0, res - 1)
    tj = np.clip((uv[:, 1] * res).astype(int), 0, res - 1)
    cnt = np.zeros((res, res), np.float32); np.add.at(cnt, (tj, ti), 1.0)
    k = int(max(3, round(sigma * 4)) | 1)
    cnt_b = cv2.GaussianBlur(cnt, (k, k), sigma)
    out = {"lambda_map": cnt_b.copy()}
    for name, v in attrs.items():
        C = v.shape[1]; acc = np.zeros((res, res, C), np.float32)
        np.add.at(acc, (tj, ti), v)
        accb = cv2.GaussianBlur(acc, (k, k), sigma)
        if accb.ndim == 2: accb = accb[..., None]
        out[name] = accb / np.maximum(cnt_b[..., None], 1e-6)
    out["valid_mask"] = (cnt_b > 0.05 * cnt_b.max()).astype(np.float32)
    return out


def bake_field(D):
    p = np.load(f"{D}/patch.npz", allow_pickle=True)
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
        np.savez(f"{D}/field_gt_{res}.npz", **{k: v.astype(np.float32) for k, v in f.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="renders_mv directory (each child has patch.npz + cameras.json + rgb_0..N.png)")
    ap.add_argument("--codec", default="patchlocks/models/fur_strand_codec_v6upright.pt")
    ap.add_argument("--out-name", "--latent-name", dest="out_name", default="target_latent64_v6codec.npz")
    ap.add_argument("--H", type=int, default=64, help="codec latent grid res (train default --res 64)")
    ap.add_argument("--straight", action="store_true", help="ablation: encode straight root->tip (drops curl)")
    ap.add_argument("--limit", type=int, default=0, help="bake only the first N dirs (0 = all); for a trial run")
    ap.add_argument("--skip-latent", action="store_true")
    ap.add_argument("--skip-field", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="re-bake even if targets already exist")
    a = ap.parse_args()

    dirs = sorted(d for d in glob.glob(a.root + "/*")
                  if os.path.isdir(d) and os.path.exists(f"{d}/patch.npz") and os.path.exists(f"{d}/cameras.json"))
    if a.limit > 0: dirs = dirs[:a.limit]
    if not dirs:
        raise FileNotFoundError(f"no dirs with patch.npz + cameras.json under {a.root}")

    ck = torch.load(a.codec, weights_only=False)
    norm = {k: torch.tensor(v, device="cuda", dtype=torch.float32) for k, v in ck["fur_norm"].items()}
    codec = StrandCodec(do_vae=False, decode_type="dir", scale_init=30.0, nr_verts_per_strand=256,
                        nr_values_to_decode=255, dim_per_value_decoded=3).cuda().eval()
    codec.load_state_dict(ck["state_dict"])
    print(f"codec={a.codec} | out={a.out_name} | H={a.H} | dirs={len(dirs)} | straight={a.straight}", flush=True)

    t0 = time.time(); done = 0
    for i, D in enumerate(dirs):
        has_lat = os.path.exists(f"{D}/{a.out_name}")
        has_fld = all(os.path.exists(f"{D}/field_gt_{r}.npz") for r in (16, 32, 64))
        if not a.skip_latent and (a.overwrite or not has_lat):
            bake_latent(D, codec, norm, a.H, a.out_name, straight=a.straight)
        if not a.skip_field and (a.overwrite or not has_fld):
            bake_field(D)
        done += 1
        if (i + 1) % 10 == 0:
            dt = time.time() - t0
            print(f"  {i+1}/{len(dirs)} | {dt:.1f}s | {dt/(i+1):.2f}s/dir", flush=True)
    dt = time.time() - t0
    print(f"BAKE-MV-DONE {done} dirs in {dt:.1f}s ({dt/max(done,1):.2f}s/dir). "
          f"Per-dir kept: rgb_0..N + cameras.json untouched; added {a.out_name} + field_gt_16/32/64.npz", flush=True)


if __name__ == "__main__":
    main()
