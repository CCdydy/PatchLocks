#!/usr/bin/env python3
"""QUANTITATIVE v16 (paper ckpt) vs V10 (locked stable) on the SAME seed-0 held-out test set.

Per deployment fur PARAMETER {arc, curl, clump, straight, lift_arc, tip_spread}, for BOTH models:
  - nMAE(pred, GT)  = mean|pred-GT| / std(GT)   (the "loss"/gap to GT; lower = better; ~1 = as bad as
    predicting the GT mean; 0 = perfect)
  - corr(pred, GT)  = Pearson correlation       (higher = better; the model tracks the per-patch variation)

Headline = a per-param table {param | V10 nMAE | v16 nMAE | v16 improvement | V10 corr | v16 corr} plus an
overall mean-nMAE / mean-corr summary and a one-line verdict.

=================================  FEASIBILITY  (same-patches: YES)  =================================
v16 and V10 are evaluated on the IDENTICAL held-out renders_mv patches, decoded through the IDENTICAL
codec + texel-sampling logic. This is apples-to-apples SAME-PATCH:

  * v16  : v14_common.build_env loads the v16 ckpt (e2a_cw15/v14_best.pt + b3_fuser.pt), rebuilds the
           seed-0 PATCH-LEVEL held-out split (env.vi) on renders_mv, the EXACT full-mode CFG Heun
           sampler (env.sample_z, w=3, fixed per-patch noise seed=gidx), and env.decode_params (codec
           v6upright). GT = env.Xn (the baked v6codec target latent) decoded on the SAME texels.

  * V10  : the LOCKED stable diffusion (patchlocks/v2_locked/V10_PATCH_STABLE/{stage3_model.pt,b3_fuser.pt},
           xd=70 = z64+6color, iter 30000). V10 is a SINGLE-VIEW model (its B3Fuser vemb is [1,192] -- it
           was trained on renders_sv / v8b_realdog with one view per patch), so it cannot ingest all 4
           renders_mv views at once. We therefore run V10 on a SINGLE-VIEW shim of each held-out patch:
           a temp dir holding view-0's cameras.json + rgb_0.png + field_gt_64.npz (its exact native input).
           SAME patch geometry -> SAME GT, SAME sampled texels. V10 conditions on RGBPROJ + its own
           B3-DINO fuser + rho (NO RAYC / NO texture_prior), targets the SAME v6upright codec latent as
           v16, and saves z64_map (raw, un-standardized z64). We RE-STANDARDIZE that z64 by v16's xm/xs and
           push it through the SAME env.decode_params at the SAME sampled texels -> the codec+texel math
           is byte-identical for both models; only the z64 differs. (env.decode_params does z*xs+xm
           internally, so re-standardizing by xm/xs round-trips V10's raw z64 losslessly.)

  Why this is fair: a deployment-capability metric asks "given this held-out patch's image(s), how close
  is each model's predicted geometry to GT?". Both models see image(s) OF THE SAME PATCH and are scored
  by the same decoded params on the same texels and the same GT. V10 was trained single-view (renders_sv)
  and is run here on view 0 of the renders_mv test patches; it was never trained on these patches, so they
  are genuinely held-out for V10 too. v16 sees all 4 views (its design); V10 sees 1 view (its design) --
  each is run with the conditioning it was built for. This is the standard "run the older model's native
  inference on the new test set" protocol, NOT apples-to-oranges; the only asymmetry (1 vs 4 input views)
  is intrinsic to V10's single-view architecture and is reported explicitly.

Run (SMOKE first, then full):
  EVAL_SMOKE=2  python patchlocks/v16/eval_v16_vs_v10.py
  EVAL_SMOKE=0  python patchlocks/v16/eval_v16_vs_v10.py   # full subset
"""
import os, sys, json, glob, tempfile, subprocess, shutil
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "patchlocks/v14"); sys.path.insert(0, "patchlocks/data_gen")
import numpy as np, torch
from v14_common import build_env

# ---- config ----
RENDERS  = "patchlocks/datasets/v11_longfur/renders_mv"
V16_DIR  = "patchlocks/datasets/v16/e2a_cw15"          # holds v14_best.pt + b3_fuser.pt (the v16 paper ckpt)
V10_DIR  = "patchlocks/v2_locked/V10_PATCH_STABLE"     # locked stable: stage3_model.pt + b3_fuser.pt (xd=70)
PYBIN    = os.environ.get("PY", sys.executable)
CFG_W    = 3.0
LEN_K    = 32
STEPS    = 40
OUT_JSON = "patchlocks/v16/eval_v16_vs_v10.json"

# How many held-out patches to score. SMOKE caps it (end-to-end check); 0/unset -> NSUB (the tractable
# full-run subset; V10 per-patch inference is a subprocess so we cap at NSUB rather than all ~240).
SMOKE = int(os.environ.get("EVAL_SMOKE", "0"))
NSUB  = int(os.environ.get("EVAL_NSUB", "80"))

# deployment params (decode key -> display name) -- the v14/v16 convention (env.decode_params keys)
ATTRS = [("arc", "arc(length)"), ("curl", "curl"), ("clump", "clump"),
         ("straight", "straight"), ("lift_arc", "lift_arc"), ("tip_spread", "tip_spread")]
PKEYS = [k for k, _ in ATTRS]


def pcorr(u, v):
    u = np.asarray(u, float); v = np.asarray(v, float)
    m = np.isfinite(u) & np.isfinite(v); u, v = u[m], v[m]
    if len(u) < 3 or u.std() < 1e-9 or v.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(u, v)[0, 1])


def nmae(pred, gt):
    """Normalized MAE: mean|pred-GT| / std(GT). 0=perfect, ~1=as bad as predicting the GT mean."""
    p = np.asarray(pred, float); g = np.asarray(gt, float)
    m = np.isfinite(p) & np.isfinite(g); p, g = p[m], g[m]
    if len(g) < 2 or g.std() < 1e-12:
        return float("nan")
    return float(np.mean(np.abs(p - g)) / (g.std() + 1e-12))


def make_single_view_dir(src_dir, sv_dir):
    """V10's locked ckpt is single-view (vemb=[1,192]); it cannot ingest renders_mv's 4 views. Build a
    1-view shim of `src_dir`: cameras.json keeping ONLY view 0, plus links to rgb_0.png + field_gt_64.npz +
    patch.npz. SAME patch geometry/GT/image-view-0 -> still same-patch, V10 just runs in its native mode."""
    os.makedirs(sv_dir, exist_ok=True)
    meta = json.load(open(f"{src_dir}/cameras.json"))
    meta1 = dict(meta); meta1["views"] = [meta["views"][0]]
    json.dump(meta1, open(f"{sv_dir}/cameras.json", "w"))
    for f in ("rgb_0.png", "field_gt_64.npz", "patch.npz"):
        s = f"{src_dir}/{f}"; dst = f"{sv_dir}/{f}"
        if os.path.exists(s) and not os.path.exists(dst):
            os.symlink(os.path.abspath(s), dst)
    return sv_dir


def run_v10_infer(patch_dir, out_dir):
    """Run the locked V10 per-patch inference (single-view native mode) on one renders_mv held-out patch.
    Returns the un-standardized z64_map (H,H,64) it generated, or None on failure. Uses field_gt rho
    (synthetic-QA fair: geometry/z64 come purely from the diffusion sample; rho only conditions density,
    which we don't score)."""
    sv_dir = make_single_view_dir(patch_dir, f"{out_dir}_sv")
    cmd = [PYBIN, "patchlocks/inference/infer_v10_pixel.py",
           "--geo_dir", V10_DIR,
           "--model", f"{V10_DIR}/stage3_model.pt",
           "--patch_dir", sv_dir, "--out", out_dir,
           "--steps", str(STEPS), "--seed", "0",
           "--rho_source", "field_gt", "--rho_npz", f"{sv_dir}/field_gt_64.npz"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    pz = f"{out_dir}/patch.npz"
    if not os.path.exists(pz):
        print(f"    [V10 infer FAIL] {os.path.basename(patch_dir)}: {r.stderr[-300:]}", flush=True)
        return None
    d = np.load(pz, allow_pickle=True)
    if "z64_map" not in d.files:
        print(f"    [V10 no z64_map] {os.path.basename(patch_dir)}", flush=True)
        return None
    return np.asarray(d["z64_map"], np.float32)   # (H,H,64) RAW (un-standardized) z64


def main():
    print(f"\n===== build env (v16 ckpt = {V16_DIR}) on {RENDERS} =====", flush=True)
    env = build_env(renders=RENDERS, e1_dir=V16_DIR, len_k=LEN_K, verbose=True)
    H, dev, xm, xs = env.H, env.dev, env.xm, env.xs
    vi = env.vi.detach().cpu().numpy().tolist()
    n_total = len(vi)
    n = min(SMOKE, n_total) if SMOKE > 0 else min(NSUB, n_total)
    print(f"\nheld-out (seed-0 val): {n_total} dirs; scoring {n} "
          f"({'SMOKE' if SMOKE>0 else f'subset NSUB={NSUB}'})", flush=True)

    # per-model, per-param accumulators of patch-level medians
    acc = {who: {k: [] for k in PKEYS} for who in ("v10", "v16", "gt")}
    n_ok = 0
    tmp = tempfile.mkdtemp(prefix="v10_eval_")
    try:
        for r in range(n):
            gidx = int(vi[r]); jj = env.vi[r:r + 1]
            d = env.dirs[gidx]
            # ---- conditions for v16 sampler ----
            gc = env.fetch(env.G_COND, jj); gl = env.fetch(env.GLOB, jj)
            rc = env.fetch(env.RHOC, jj); rp = env.fetch(env.RGBPROJ, jj); ry = env.fetch(env.RAYC, jj)
            mm = env.fetch(env.MASK, jj)
            # SAME masked texels for v16 / V10 / GT (so all three decode identical roots)
            ib, ihw, kept = env.sample_texels(mm, LEN_K, seed=1000000 + gidx)
            if ib is None or not bool(kept.any()):
                print(f"  [{r+1}/{n}] {os.path.basename(d)}: no masked texels, skip", flush=True)
                continue
            # ---- v16: full CFG sampler (fixed per-patch noise) -> standardized z64 ----
            z_v16 = env.sample_z(gc, gl, rc, rp, ry, cfg=CFG_W, seed=gidx)
            # ---- GT: baked v6codec target latent (standardized) ----
            z_gt = env.fetch(env.Xn, jj)
            # ---- V10: run its inference subprocess on the SAME dir -> raw z64 -> re-standardize ----
            z10_raw = run_v10_infer(d, f"{tmp}/p{r}")
            if z10_raw is None:
                continue
            # raw (H,H,64) -> (1,64,H,H), standardize by v16 xm/xs so env.decode_params' z*xs+xm round-trips
            z10 = torch.tensor(z10_raw, device=dev).permute(2, 0, 1)[None]   # (1,64,H,H) raw
            z10_std = (z10 - xm) / xs                                        # standardized for decode_params

            p16 = env.decode_params(z_v16, ib, ihw, LEN_K)
            pgt = env.decode_params(z_gt, ib, ihw, LEN_K)
            p10 = env.decode_params(z10_std, ib, ihw, LEN_K)
            for k in PKEYS:
                acc["v16"][k].append(float(p16[k][0]))
                acc["gt"][k].append(float(pgt[k][0]))
                acc["v10"][k].append(float(p10[k][0]))
            n_ok += 1
            if (r + 1) % 10 == 0 or SMOKE > 0:
                print(f"  [{r+1}/{n}] {os.path.basename(d)} ok (n_ok={n_ok})", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        del env; torch.cuda.empty_cache()

    if n_ok < 2:
        print(f"\n*** only {n_ok} patches scored -- need >=2; aborting metrics ***", flush=True)
        return

    # ---- per-param metrics ----
    gt = acc["gt"]
    per_param = {}
    for k, disp in ATTRS:
        v10_nmae = nmae(acc["v10"][k], gt[k]); v16_nmae = nmae(acc["v16"][k], gt[k])
        v10_corr = pcorr(acc["v10"][k], gt[k]); v16_corr = pcorr(acc["v16"][k], gt[k])
        impr = (v10_nmae - v16_nmae) / v10_nmae * 100 if np.isfinite(v10_nmae) and v10_nmae > 1e-9 else float("nan")
        per_param[k] = dict(display=disp, v10_nmae=v10_nmae, v16_nmae=v16_nmae,
                            v16_nmae_improvement_pct=impr, v10_corr=v10_corr, v16_corr=v16_corr)

    def _mean(vals):
        vv = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vv)) if vv else float("nan")

    mean_v10_nmae = _mean([per_param[k]["v10_nmae"] for k, _ in ATTRS])
    mean_v16_nmae = _mean([per_param[k]["v16_nmae"] for k, _ in ATTRS])
    mean_v10_corr = _mean([per_param[k]["v10_corr"] for k, _ in ATTRS])
    mean_v16_corr = _mean([per_param[k]["v16_corr"] for k, _ in ATTRS])
    overall_nmae_impr = (mean_v10_nmae - mean_v16_nmae) / mean_v10_nmae * 100 if mean_v10_nmae > 1e-9 else float("nan")

    wins = sum(1 for k, _ in ATTRS
               if np.isfinite(per_param[k]["v16_nmae"]) and np.isfinite(per_param[k]["v10_nmae"])
               and per_param[k]["v16_nmae"] < per_param[k]["v10_nmae"])
    verdict = (f"v16 beats V10 on {wins}/{len(ATTRS)} params; mean nMAE {mean_v10_nmae:.3f}->{mean_v16_nmae:.3f} "
               f"({overall_nmae_impr:+.0f}% lower error), mean corr {mean_v10_corr:+.2f}->{mean_v16_corr:+.2f}.")

    feasibility = ("SAME-PATCHES (apples-to-apples on geometry): v16 and V10 scored on the IDENTICAL "
                   "seed-0 held-out renders_mv patches, SAME codec + SAME sampled texels + SAME GT. "
                   "V10's per-patch inference (infer_v10_pixel.py) runs on a SINGLE-VIEW shim (view 0) of "
                   "each held-out patch because the locked V10 ckpt is single-view (vemb=[1,192], trained "
                   "on renders_sv) and cannot ingest renders_mv's 4 views. V10's raw z64_map is "
                   "re-standardized by v16 xm/xs and decoded by the same env.decode_params. Only the "
                   "predicted z64 differs. The one intrinsic asymmetry: v16 conditions on all 4 views, V10 "
                   "on 1 view -- each run in its native designed mode (not a confound, it is V10's "
                   "single-view capability being measured).")

    # ---- markdown table ----
    def f3(x, sgn=False):
        if not np.isfinite(x): return "nan"
        return (f"{x:+.3f}" if sgn else f"{x:.3f}")
    lines = []
    lines.append("| param | V10 nMAE | v16 nMAE | v16 improvement | V10 corr | v16 corr |")
    lines.append("|---|---|---|---|---|---|")
    for k, disp in ATTRS:
        p = per_param[k]
        impr = f"{p['v16_nmae_improvement_pct']:+.0f}%" if np.isfinite(p["v16_nmae_improvement_pct"]) else "nan"
        lines.append(f"| {disp} | {f3(p['v10_nmae'])} | {f3(p['v16_nmae'])} | {impr} | "
                     f"{f3(p['v10_corr'], True)} | {f3(p['v16_corr'], True)} |")
    lines.append(f"| **MEAN** | **{f3(mean_v10_nmae)}** | **{f3(mean_v16_nmae)}** | "
                 f"**{overall_nmae_impr:+.0f}%** | **{f3(mean_v10_corr, True)}** | **{f3(mean_v16_corr, True)}** |")
    table_md = "\n".join(lines)

    print("\n" + "=" * 88)
    print(f"v16 (paper) vs V10 (locked stable) -- SAME-PATCH held-out, n={n_ok}, CFG w={CFG_W}")
    print("nMAE = mean|pred-GT|/std(GT) (lower=better, 0=perfect, ~1=predict-the-mean); corr=Pearson")
    print("=" * 88)
    print(table_md)
    print("\nOVERALL: mean nMAE {:.3f} (V10) -> {:.3f} (v16)  =  {:+.0f}% lower error | "
          "mean corr {:+.2f} (V10) -> {:+.2f} (v16)".format(
              mean_v10_nmae, mean_v16_nmae, overall_nmae_impr, mean_v10_corr, mean_v16_corr))
    print(f"\nVERDICT: {verdict}")
    print(f"\nFEASIBILITY: {feasibility}")

    out = dict(
        meta=dict(n_scored=n_ok, n_requested=n, cfg_w=CFG_W, steps=STEPS, len_k=LEN_K,
                  smoke=SMOKE, nsub=NSUB, renders=RENDERS, v16_dir=V16_DIR, v10_dir=V10_DIR,
                  metric="nMAE=mean|pred-GT|/std(GT); corr=Pearson",
                  params=[d for _, d in ATTRS], same_patches=True),
        feasibility=feasibility,
        per_param=per_param,
        overall=dict(mean_v10_nmae=mean_v10_nmae, mean_v16_nmae=mean_v16_nmae,
                     overall_nmae_improvement_pct=overall_nmae_impr,
                     mean_v10_corr=mean_v10_corr, mean_v16_corr=mean_v16_corr,
                     v16_param_wins=wins, n_params=len(ATTRS)),
        verdict=verdict,
        table_markdown=table_md,
    )
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"\nsaved {OUT_JSON}", flush=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
