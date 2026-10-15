"""V16 (V14-family) per-patch inference -> DENSE predicted patch.npz for the Blender visual gate.

Given a held-out patch dir and a v16 ckpt, this samples the FROZEN v16/E1 full-mode CFG
diffusion (the EXACT sampler v14_common.build_env / eval_e1_2d_match.py use, w=3) to produce a
64x64x64 z64 FIELD, then runs infer_v10_pixel.py's EXACT DENSE downstream on that field:

  1. distribute ~rho strands/texel (rho-weighted from field_gt_64.npz lambda_map -> matches the GT
     patch's ~34k strand count; falls back to uniform --density 4.0 if no field).
  2. BILINEAR-sample (_bilinear from field_to_strands) the v16 z64 field at each strand root.
  3. decode z64 -> root-centered 3D strands with the frozen v6upright codec.
  4. place each strand at its surface root [(u-0.5)*size, (v-0.5)*size, 0], +Z up (EXACT GT frame).
  5. COLOR each strand from the INPUT PHOTO projected at its root_uv (RGBPROJ un-standardized) --
     so the prediction is colored like the input photo, NOT a flat tan. (v16 has no generated color.)

This replaces the old 1-strand-per-texel emit (a sparse ~4k tan net) with the dense, photo-colored
v10-style output (tens of thousands of strands), so the Blender render reads as DENSE COLORED FUR.

Writes a predicted patch.npz with the keys render_patch_samecam.py reads:
  strands   (N, L, 3)  float32  -- root-centered offset + texel root, +Z up, L matched to GT
  root_uv   (N, 2)     float32  -- root (u,v) in [0,1]
  color     (N, 3)     float32  -- photo-projected input color at each root, in [0,1]
  color_tip (N, 3)     float32  -- slightly lighter tip color
  radius    (N,)       float32  -- GT field radius_mu (per-root), or GT patch radius scale, or default
  size      scalar     float32  -- = GT patch size

HONEST CAVEAT: the diffusion does NOT set ABSOLUTE strand direction (z64 carries the codec-CANONICAL
in-plane direction). The dense predicted fur therefore points in the codec-canonical sweep, NOT
necessarily the image's fur direction -- this gate shows GEOMETRY/CHARACTER (length, lift, curl,
clump, density) + photo color, like the V10 reference, not absolute direction.

Run (repo root, difflocks env):
  python patchlocks/v16/infer_v16_patch.py \
    --model patchlocks/datasets/v16/e2a_cw15/v14_best.pt \
    --renders patchlocks/datasets/v11_longfur/renders_mv \
    --patch_dir patchlocks/datasets/v11_longfur/renders_mv/rd_0007_ultra_short_bicolor \
    --out <out_dir>           # writes <out_dir>/patch.npz
"""
import os, sys, json, argparse
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "patchlocks/v14")
sys.path.insert(0, "patchlocks/data_gen")
import numpy as np
import torch
from v14_common import build_env
from field_to_strands import _bilinear

CFG_W = 3.0
NEUTRAL_TAN = np.array([0.62, 0.50, 0.36], np.float32)  # fallback fur color where photo is degenerate
DEFAULT_RADIUS = 0.0007359216


def _model_dir(model_path):
    """build_env loads E1 net from <e1_dir>/v14_best.pt and the fuser from <e1_dir>/b3_fuser.pt.
    Accept either the dir or the v14_best.pt file as --model."""
    p = model_path
    if os.path.isfile(p) and p.endswith(".pt"):
        return os.path.dirname(p)
    return p


def distribute_roots(rho_map, size, H, seed):
    """STRATIFIED rho-weighted root sampling -- IDENTICAL math to infer_v10_pixel.py / field_to_strands:
    per-cell count = floor(lam) + Bernoulli(frac). Returns (uv (M,2), roots (M,3))."""
    rng = np.random.default_rng(seed)
    lam = np.clip(rho_map, 0, None)
    base = np.floor(lam).astype(np.int32)
    extra = (rng.uniform(0, 1, lam.shape) < (lam - base)).astype(np.int32)
    nc = (base + extra).reshape(-1)
    cells = np.repeat(np.nonzero(nc)[0], nc[nc > 0])
    if len(cells) == 0:
        raise RuntimeError("rho map sampled zero roots")
    cj, ci = np.divmod(cells, H)                                  # row=v, col=u
    uv = np.stack([(ci + rng.uniform(0, 1, len(cells))) / H,
                   (cj + rng.uniform(0, 1, len(cells))) / H], 1).astype(np.float32)
    M = len(uv)
    roots = np.stack([(uv[:, 0] - 0.5) * size, (uv[:, 1] - 0.5) * size, np.zeros(M)], 1).astype(np.float32)
    return uv, roots


def build_v16_env(model, renders, codec_path="patchlocks/models/fur_strand_codec_v6upright.pt",
                  latent_name="target_latent64_v6codec.npz", verbose=True):
    """Build the frozen v16 env ONCE (DINO + conditions + sampler + codec). The visual gate calls this
    once and reuses the returned env across ALL picked patches (the per-patch env rebuild was the
    ~3min/patch bottleneck). Returns the v14_common Env namespace."""
    e1_dir = _model_dir(model)
    if verbose:
        print(f"[infer_v16] build_env(renders={renders}, e1_dir={e1_dir}) [shared, built once] ...", flush=True)
    return build_env(renders=renders, e1_dir=e1_dir, codec_path=codec_path,
                     latent_name=latent_name, verbose=verbose)


def infer_patch_with_env(env, patch_dir, out_dir, cfg=CFG_W, seed=-1,
                         codec_path="patchlocks/models/fur_strand_codec_v6upright.pt",
                         rho_source="field_gt", density=4.0, color_source="photo",
                         default_radius=DEFAULT_RADIUS):
    """Sample + write the DENSE photo-colored patch.npz for ONE patch using an ALREADY-BUILT env.
    This is the per-patch fast path the visual gate uses (no env rebuild). Returns (out_npz, M)."""
    patch_dir = os.path.abspath(patch_dir).rstrip("/")
    os.makedirs(out_dir, exist_ok=True)
    gt = np.load(f"{patch_dir}/patch.npz", allow_pickle=True)
    L = int(gt["strands"].shape[1]); size_gt = float(gt["size"])
    gt_radius = float(np.ravel(gt["radius"].astype(np.float32))[0]) if "radius" in gt.files else default_radius
    a = argparse.Namespace(out=out_dir, cfg=cfg, seed=seed, codec_path=codec_path,
                           rho_source=rho_source, density=density, color_source=color_source,
                           default_radius=default_radius)
    return run_one(env, patch_dir, a, L, size_gt, gt_radius)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="patchlocks/datasets/v16/e2a_cw15/v14_best.pt",
                    help="v16 ckpt (v14_best.pt) or its dir; b3_fuser.pt must sit alongside")
    ap.add_argument("--renders", default="patchlocks/datasets/v11_longfur/renders_mv")
    ap.add_argument("--patch_dir", required=True, help="the held-out render dir to predict")
    ap.add_argument("--out", required=True, help="dir to write predicted patch.npz")
    ap.add_argument("--cfg", type=float, default=CFG_W)
    ap.add_argument("--seed", type=int, default=-1, help="-1 => use the patch global index (matches eval_e1)")
    ap.add_argument("--codec_path", default="patchlocks/models/fur_strand_codec_v6upright.pt")
    ap.add_argument("--latent_name", default="target_latent64_v6codec.npz")
    ap.add_argument("--rho_source", choices=["field_gt", "uniform"], default="field_gt",
                    help="field_gt: rho-weighted from field_gt_<H>.npz lambda_map (~GT density). uniform: --density per texel")
    ap.add_argument("--density", type=float, default=4.0, help="uniform strands per texel when --rho_source=uniform")
    ap.add_argument("--default_radius", type=float, default=DEFAULT_RADIUS,
                    help="fallback radius scale if GT patch.npz/field has none")
    ap.add_argument("--color_source", choices=["photo", "neutral"], default="photo",
                    help="photo: paint each strand from the input photo at root_uv (default). neutral: flat tan.")
    a = ap.parse_args()

    patch_dir = os.path.abspath(a.patch_dir).rstrip("/")
    e1_dir = _model_dir(a.model)
    os.makedirs(a.out, exist_ok=True)

    # ---- GT patch.npz: lock onto exact L, size, radius scale, and frame ----
    gt = np.load(f"{patch_dir}/patch.npz", allow_pickle=True)
    L = int(gt["strands"].shape[1])
    size_gt = float(gt["size"])
    gt_radius = float(np.ravel(gt["radius"].astype(np.float32))[0]) if "radius" in gt.files else a.default_radius
    gt_n = int(gt["strands"].shape[0])
    print(f"[infer_v16] GT patch L={L} size={size_gt} radius_scale={gt_radius} N_gt={gt_n}", flush=True)

    # ---- frozen v16 env (DINO + conditions + sampler + codec), identical to eval_e1_2d_match ----
    print(f"[infer_v16] build_env(renders={a.renders}, e1_dir={e1_dir}) ...", flush=True)
    env = build_env(renders=a.renders, e1_dir=e1_dir,
                    codec_path=a.codec_path, latent_name=a.latent_name, verbose=True)
    run_one(env, patch_dir, a, L, size_gt, gt_radius)
    del env
    torch.cuda.empty_cache()


def sample_z_field(env, gidx, cfg, seed):
    """Sample the v16 diffusion -> REAL (un-standardized) z64 FIELD as (H,H,64) numpy."""
    jj = torch.tensor([gidx], device=env.idx_dev)
    gc = env.fetch(env.G_COND, jj); gl = env.fetch(env.GLOB, jj)
    rc = env.fetch(env.RHOC, jj); rp = env.fetch(env.RGBPROJ, jj); ry = env.fetch(env.RAYC, jj)
    z_pred = env.sample_z(gc, gl, rc, rp, ry, cfg=cfg, seed=seed)          # (1,64,H,H) standardized
    z_real = (z_pred[:, :64] * env.xs + env.xm)[0].permute(1, 2, 0).contiguous().cpu().numpy()  # (H,H,64) real z64
    return z_real


def photo_map_for(env, gidx):
    """Recover the view-fused INPUT PHOTO color map (H,H,3) in [0,1] from build_env's standardized RGBPROJ.
    build_env stores RGBPROJ = (proj_rgb - 0.5)/0.25  ->  proj_rgb = RGBPROJ*0.25 + 0.5 (= infer_v10's RGBRAW)."""
    jj = torch.tensor([gidx], device=env.idx_dev)
    rgbproj = env.fetch(env.RGBPROJ, jj)[0].permute(1, 2, 0).cpu().numpy()  # (H,H,3) standardized
    return np.clip(rgbproj * 0.25 + 0.5, 0.0, 1.0).astype(np.float32)


def decode_dense(env, z_real, uv, codec_path):
    """Bilinear-sample the v16 z64 FIELD at each root uv, decode root-centered strands. Returns (M,P,3)."""
    from models.strand_codec import StrandCodec
    dev = env.dev
    _lck = torch.load(codec_path, weights_only=False)
    _lcn = {k: torch.tensor(v, device=dev, dtype=torch.float32) for k, v in _lck["fur_norm"].items()}
    _codec = StrandCodec(do_vae=False, decode_type="dir", scale_init=30.0,
                         nr_verts_per_strand=256, nr_values_to_decode=255, dim_per_value_decoded=3).to(dev).eval()
    _codec.load_state_dict(_lck["state_dict"])
    for p in _codec.parameters():
        p.requires_grad = False
    z = torch.tensor(_bilinear(z_real, uv), device=dev, dtype=torch.float32)   # (M,64) bilinear-sampled field
    with torch.no_grad():
        # decode in chunks to bound memory at tens of thousands of strands
        recs = []
        CH = 20000
        for s in range(0, len(z), CH):
            pr = _codec.decoder(z[s:s + CH], None, _lcn)
            r = pr["strand_positions"] if "strand_positions" in pr else pr[[k for k in pr if "pos" in k.lower()][0]]
            recs.append((r - r[:, :1]).float().cpu().numpy())
    return np.concatenate(recs, 0)                                            # (M,P,3) root-centered


def resample_arc(rec, L):
    """Resample each (P,3) strand to L points by ARC length (matches infer_v10_pixel's interp)."""
    M = rec.shape[0]
    out = np.zeros((M, L, 3), np.float32)
    for n in range(M):
        s = rec[n] - rec[n][:1]
        seg = np.linalg.norm(np.diff(s, axis=0), axis=1)
        cl = np.concatenate([[0], np.cumsum(seg)])
        tt = np.linspace(0, cl[-1] + 1e-9, L)
        out[n] = np.stack([np.interp(tt, cl, s[:, c]) for c in range(3)], 1)
    return out


def run_one(env, patch_dir, a, L, size_gt, gt_radius):
    """Sample the v16 z64 field for one patch (already in env.dirs) and write the DENSE photo-colored patch.npz."""
    H = env.H
    base = os.path.basename(patch_dir.rstrip("/"))
    gidx = None
    for i, d in enumerate(env.dirs):
        if os.path.basename(d.rstrip("/")) == base:
            gidx = i
            break
    if gidx is None:
        raise SystemExit(f"patch dir {base} not in env.dirs ({len(env.dirs)} dirs); excluded by latent/field/pose filter?")
    seed = gidx if a.seed < 0 else a.seed
    print(f"[infer_v16] patch gidx={gidx} (held-out={'yes' if gidx in env.vi.tolist() else 'NO'}) seed={seed}", flush=True)

    # ---- v16 diffusion -> z64 FIELD (the diffusion part, unchanged) ----
    z_real = sample_z_field(env, gidx, a.cfg, seed)                            # (H,H,64) real z64 field
    print(f"[infer_v16] sampled z64 field {z_real.shape}", flush=True)

    # ---- rho-weighted DENSE root distribution (EXACT v10 / field_to_strands math) ----
    radius_map = np.full((H, H), float(a.default_radius), np.float32)
    if a.rho_source == "field_gt":
        fg = np.load(f"{patch_dir}/field_gt_{H}.npz")
        rho_map = fg["lambda_map"].astype(np.float32)
        if "radius_mu" in fg.files:
            rm = fg["radius_mu"].astype(np.float32)
            radius_map = rm[..., 0] if rm.ndim == 3 else rm
    else:
        rho_map = np.full((H, H), float(a.density), np.float32)
    uv, roots = distribute_roots(rho_map, size_gt, H, seed)
    M = len(uv)
    print(f"[infer_v16] DENSE roots: {M} strands (rho_source={a.rho_source}, rho mean={rho_map.mean():.2f})", flush=True)

    # ---- bilinear-sample z64 field at each root -> decode -> arc-resample to L ----
    rec = decode_dense(env, z_real, uv, a.codec_path)                         # (M,P,3) root-centered
    outp = resample_arc(rec, L)                                               # (M,L,3) root-centered, arc

    # ---- place at surface roots [(u-.5)size,(v-.5)size,0], +Z up; clamp into surface; pin root ----
    strands = (outp - outp[:, :1] + roots[:, None]).astype(np.float32)
    strands[..., 2] = np.maximum(strands[..., 2], 0.0)
    strands[:, 0] = roots
    root_uv = uv.astype(np.float32)

    # ---- COLOR: paint each strand from the INPUT PHOTO at its root_uv (v10 --color_source photo path) ----
    # Mirrors infer_v10_pixel's `--color_source photo` (col = photo at root_uv), with NO recoloring of dark
    # fur: only TRULY degenerate texels (no view saw them -> exactly ~0 projected RGB) fall back to tan. The
    # GT coat itself has ~25% near-black (dark bicolor) strands, so a large threshold would WASH OUT real dark
    # fur and stop matching the photo. Keep the photo color, dark regions included.
    if a.color_source == "photo":
        photo_map = photo_map_for(env, gidx)                                  # (H,H,3) view-fused input photo
        col = _bilinear(photo_map, root_uv).astype(np.float32)
        bad = (col.max(1) < 1e-3)                                             # only no-view (literally ~0) texels
        n_bad = int(bad.sum())
        if bad.any():
            col[bad] = NEUTRAL_TAN
    else:
        col = np.tile(NEUTRAL_TAN, (M, 1)).astype(np.float32); n_bad = M
    color = np.clip(col, 0.0, 1.0).astype(np.float32)
    color_tip = np.clip(color * 0.78 + 0.22, 0.0, 1.0).astype(np.float32)     # slightly lighter tip

    radius = _bilinear(radius_map[..., None], root_uv)[:, 0].astype(np.float32)

    out_npz = os.path.join(a.out, "patch.npz")
    np.savez(out_npz,
             strands=strands.astype(np.float32),
             root_uv=root_uv.astype(np.float32),
             color=color,
             color_tip=color_tip,
             radius=radius,
             size=np.float32(size_gt))
    print(f"[infer_v16] wrote {out_npz}  N={M} L={L} size={size_gt}", flush=True)

    # ---- verify vs GT: keys / dtypes / frame / density / color ----
    pred = np.load(out_npz, allow_pickle=True)
    for k in ["strands", "root_uv", "color", "color_tip", "radius", "size"]:
        assert k in pred.files, f"missing key {k} in predicted npz"
    assert pred["strands"].dtype == np.float32 and pred["strands"].ndim == 3 and pred["strands"].shape[1] == L
    assert pred["root_uv"].shape[1] == 2 and pred["color"].shape[1] == 3 and pred["color_tip"].shape[1] == 3
    assert float(pred["size"]) == size_gt
    pr_roots = pred["strands"][:, 0, :]; pr_ru = pred["root_uv"]
    exp = np.stack([(pr_ru[:, 0] - 0.5) * size_gt, (pr_ru[:, 1] - 0.5) * size_gt, np.zeros(len(pr_ru))], -1)
    frame_err = float(np.abs(pr_roots - exp).mean())
    arc = np.linalg.norm(np.diff(strands, axis=1), axis=-1).sum(1)
    print(f"[infer_v16] VERIFY  keys OK dtype=float32 L={L} size={float(pred['size'])} "
          f"root-frame mean|err|={frame_err:.2e} (~0)", flush=True)
    print(f"[infer_v16] DENSE strand count N={M}  | color range [{color.min():.3f},{color.max():.3f}] "
          f"mean {color.mean(0).round(3)}  fallback_tan={n_bad}/{M}  "
          f"arc={np.median(arc)/size_gt:.3f}x  z[{strands[...,2].min():.4f},{strands[...,2].max():.4f}]", flush=True)
    return out_npz, M


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
