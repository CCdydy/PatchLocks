"""V16 Clump-Lite — E1 (V14 full-mode z64 diffusion) + two clump levers. KEEPS E1 INTACT.

This file is patchlocks/v14/train_v14.py (the E1 recipe: full-mode z64 diffusion as the ONLY
generator, sigma_max=80, P_mean=-0.4, CFG cond_drop+w, EMA, PURE EDM-weighted MSE, same +7 raydir
condition, same Heun sampler, composite-score eval + early-stop) with THREE ADDITIVE clump enhancements,
all behind flags that default OFF so the E1 path is byte-for-byte preserved:

  ADD 1 (--clump_cw <float>, 0=off): clump-sensitive z64 CHANNEL WEIGHTING. EXTENDS the existing
    clump_ch_w / CW tensor: up-weights the clump-sensitive z64 channels (from diag_clump_channels.json
    'suggested': tier-2.0 set [7,16,24,29,37,38,52,54] at weight clump_cw, tier-3.0 set [24,29,38,54]
    at a higher sub-weight) in the LATENT EDM-MSE only (CW multiplies ||z_pred-z_gt||^2). Capped at 4.
    This is NOT a decoded-param loss on the diffused output D (the V11/V12 teacher-forced trap is forbidden).

  ADD 2 (--clump_bucket 0/1, 0=off): clump-LEVEL bucket sampler. At startup decodes GT z64 -> per-patch
    clump (reuse decode_params), buckets train patches into 4 clump quartiles, samples BALANCED across
    buckets (replaces E1's hf_oversample sampling weights). A per-patch coherence proxy (neighbor tangent
    alignment / tangled-rate) is computed so tangled patches are NOT oversampled (avoid teaching 'more
    mess = more clump'). Buckets by clump LEVEL, NOT coat type (coat-balanced = the V15 failure).

  ADD 3 (always on in eval): clump-COHERENCE eval metrics. In the held-out composite-score eval, in
    ADDITION to the score, prints clump corr + clump std-ratio, coherence (neighbor alignment /
    parallelism / tangled-rate) of the SAMPLED output vs GT, and the strong-attr corrs (arc/lift/straight/curl).

HARD RULES (red lines): NO decoded-param/len loss on the diffused output D; NO aux style head; NO
coat-balanced sampler; NO group-dropout; NO SPADE; do NOT special-case curl. Everything is E1 + the two
clump levers above. Do NOT touch V10-V15 files/ckpts.

Original V13 docstring follows:
V13 — multi-view, raydir-in-TBN conditioned z64 regressor (+ optional residual diffusion).

Derived from train_v11_clean.py via TARGETED edits (do NOT touch V11/V12). The thesis (V13_SPEC §1):
PatchLocks's length/character gap is an INFORMATION gap, not a capacity gap -- a single view is
systematically ambiguous for regressing the 3D fur latent because the condition never encodes the
observation direction relative to the local fur TBN frame. V13 closes the loop:
  - CONDITION (§1a item1, §1b): add per-texel per-view `raydir_local` (ray surf->cam expressed in the
    patch TBN) + `grazing = ray.n` + per-view `vis` mask + a per-view POSE token. The output z64 already
    lives in TBN (codec v6upright); now the input shares that frame.
  - FUSER (§5): MVFuser replaces B3Fuser's `vemb=Parameter(K,dim)` (which PINS the view count) with a
    POSE-conditioned token (MLP of pose_view) -> visibility-masked softmax over V*S tokens + a learned
    `unseen` token. Accepts VARIABLE V (train view-dropout V in {1..K}; deploy more). semb kept.
  - REG-AS-BASE (§1a items 2,3,5; §2 gate): the `reg` net (trained-then-discarded in V11) is promoted to
    a SAVED deterministic base. --single_view runs the single-view baseline; default multi-view.
    --no_diffusion (V13-1 default) trains reg-only.
  - EVAL (§1a item4, §7): held-out decode eval reports ONLY the 7 deployment params, split short vs long.
  - HARD RULES (§6): decoded-character/param losses live on the ZERO-INPUT reg base ONLY; the diffused D
    NEVER enables param_w/len_w.

Original V10 docstring follows:
Stage-3 v10 PIXEL-conditioned diffusion (geometry + COLOR). Single-image. Target is the 70-ch UV texture
[z64 geometry (64) + LIT root/tip color (6)] baked on the patch UV/root_uv. Color target is the LIT color
projected from the SAME input image (root = RGBPROJ at scalp texel; tip = image sampled at the GT tip
projection) — NOT canonical albedo, which the model could not de-light from a lit input and so did not
generalize (held-out root MSE 0.071 > per-patch-mean baseline 0.026). The lit target is well-posed (input
determines target) and still yields an optimizable per-patch colormap + occluded fill-in. Conditioned on RAW
PIXELS (RGBPROJ: input RGB back-projected onto the scalp shell — carries fine detail + color DINO washes out) +
B3 multi-shell DINO backprojection G(u,v) + global DINO token, with rho/density still a CLEAN condition (not a
generation target). EDM (Karras) precond + Heun sampler. Color loss is masked by the color-valid mask (CMASK =
root exists AND texel visible), separate from the z64 geometry mask (MASK); z64 and color losses + metrics are
logged SEPARATELY (--color_w weights color).
NOT YET in the target (next version): rho/rho_delta, radius/thickness, material roughness/sheen.
Run:  python patchlocks/training/stage3_v10_pixel.py --iters 50000 --renders <single-view renders> --out <dir>
"""
import os, sys, glob, json, argparse
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "patchlocks/data_gen")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from models.strand_codec import StrandCodec
from field_to_strands import sample_from_fields, _bilinear

ap = argparse.ArgumentParser()
ap.add_argument("--renders", default="patchlocks/datasets/v4/renders"); ap.add_argument("--res", type=int, default=32)
ap.add_argument("--dino_res", type=int, default=448); ap.add_argument("--shells", type=int, default=5); ap.add_argument("--dim", type=int, default=192)
ap.add_argument("--unet_dim", type=int, default=256, help="V11 denoiser width; B3 condition dim stays --dim")
ap.add_argument("--no_attn", action="store_true", help="disable V11 bottleneck self-attention ablation")
ap.add_argument("--iters", type=int, default=6000); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--overfit", type=int, default=0)
ap.add_argument("--steps", type=int, default=40); ap.add_argument("--nsamples", type=int, default=8); ap.add_argument("--out", default="patchlocks/datasets/stage3")
ap.add_argument("--sigma_min", type=float, default=0.01); ap.add_argument("--sigma_max", type=float, default=5.0)
ap.add_argument("--hf_w", type=float, default=0.0, help="high-freq (gradient) residual loss weight (clump detail); 0=off")
ap.add_argument("--spec_w", type=float, default=0.0, help="spectral high-freq ENERGY loss; LEAVE 0 -- DIFF already has hf energy, energy match just adds noise (v5_lowpass_diag)")
ap.add_argument("--hf_oversample", type=float, default=1.0, help="batch-sampling weight for high-freq coat archetypes (wiry/wet/double); 1=uniform")
ap.add_argument("--clump_ch_w", type=float, default=1.0, help="up-weight the clump-structure-bearing z64 channels (ch37/29/38..) in latent+gradient recon loss; 1=off")
ap.add_argument("--color_w", type=float, default=1.0, help="v10: weight on the root/tip color loss (masked by CMASK = color-valid texels), separate from the z64 geometry loss")
ap.add_argument("--no_color", action="store_true", help="ablation: drop the color prediction term -> z64-ONLY (64ch) target. Isolates geometry/direction from the joint color task (still PIXEL-conditioned on RGBPROJ).")
ap.add_argument("--len_w", type=float, default=0.0, help="decoded length+height matching loss (counters diffusion under-generating strand length/standing); 0=off")
ap.add_argument("--elev_w", type=float, default=0.0, help="decoded rise/arc matching loss for upright coats; 0=off. V11.1: SET 0 -- it over-lifts long draping fur (+70%)")
ap.add_argument("--param_w", type=float, default=0.0, help="V11.1: differentiable decoded-param MATCHING loss to GT for {arc,lift_arc,curl,tip_spread,clump,straight} (relative-error, the deployment params). Directly supervises the character stats z64-MSE can't hold. 0=off")
ap.add_argument("--param_weights", default="", help="comma list for V11.2 decoded-param weights, e.g. arc=0.5,straight=0.5,lift_arc=1,curl=3,tip_spread=4,clump=4. Empty = all 1.")
ap.add_argument("--len_k", type=int, default=32, help="texels per patch decoded for the length/height loss")
ap.add_argument("--b3_pre_iters", type=int, default=2000, help="supervised B3 fuser pretrain steps before freezing G_COND")
ap.add_argument("--mode", choices=["direct", "residual"], default="direct", help="direct: p([z64,rho]|G); residual: old anchored mu(G)+diffusion residual")
ap.add_argument("--sample_temp", type=float, default=1.0, help="multiply initial sampling noise; cheap diversity probe")
ap.add_argument("--cond_drop", type=float, default=0.0, help="drop G/global during diffusion training; enables CFG when >0")
ap.add_argument("--cfg", type=float, default=1.0, help="classifier-free guidance scale at sampling time; direct mode only")
ap.add_argument("--rho_w", type=float, default=1.0, help="density texture MSE weight")
ap.add_argument("--rho_mass_w", type=float, default=0.0, help="extra global root-count loss weight; useful at H64+")
ap.add_argument("--decode_thr", type=float, default=0.3, help="relative rho threshold for decode mask; 0 uses all positive rho")
ap.add_argument("--gcond_batch", type=int, default=8, help="batch size for frozen B3 G_COND precompute; lower avoids H64 peak-memory OOM")
ap.add_argument("--cache_device", choices=["cuda", "cpu"], default="cuda", help="where to keep precomputed dataset tensors; cpu is slower but allows 14k/H64 rotaug to coexist with other GPU jobs")
ap.add_argument("--max_dirs", type=int, default=0, help="subsample to at most this many dirs (coat-balanced stride) to BOUND the in-RAM feature cache; 0 = all. ~4.9MB/dir, so 6000 -> ~29GB cache")
ap.add_argument("--eval_batch", type=int, default=64, help="validation sampling batch size; keeps held-out rotaug eval from OOM")
ap.add_argument("--resume_b3", action="store_true", help="reuse an existing b3_fuser.pt in --out when shapes match")
ap.add_argument("--resume_stage3", default="", help="resume diffusion net/reg weights from a stage3_model*.pt checkpoint; optimizer state is restarted, LR scheduler continues from checkpoint iter")
ap.add_argument("--save_every", type=int, default=5000, help="write stage3_model_latest.pt every N iters for pause/visual QA; 0 disables")
ap.add_argument("--latent_name", default="target_latent64.npz", help="baked latent filename in each render dir (use a v6codec re-bake without overwriting the old latents)")
ap.add_argument("--codec_path", default="patchlocks/models/fur_strand_codec.pt", help="codec used for the differentiable len/height loss; MUST match the codec that baked --latent_name")
# ===== V13 additions =====
ap.add_argument("--pose_name", default="v13_pose_64.npz", help="V13 per-dir pose sidecar (keys raydir_tbn (K,H,H,3), pose_view (K,12), vis_frustum (K,H,H))")
ap.add_argument("--no_diffusion", action="store_true", help="V13-1 default: train ONLY the deterministic reg base (residual diffusion OFF until reg is stable, §1a item5)")
ap.add_argument("--single_view", action="store_true", help="V13 §1a item2 baseline: condition on view 0 ONLY (the control that quantifies the multi-view gain). Default = multi-view (all K).")
ap.add_argument("--view_dropout", action="store_true", help="MVFuser robustness (§5): per-step random V in {1..Kuse} subset of views (1->many + single-view fallback). Off = always all Kuse views.")
ap.add_argument("--reg_iters", type=int, default=0, help="warm up reg-only for this many steps BEFORE the residual target switches on (§8 risk6). 0 = warm for a.iters when --no_diffusion else min(iters,8000).")
ap.add_argument("--reg_param_w", type=float, default=1.0, help="V13 §6: decoded 7-param relerr loss weight ON THE REG BASE ONLY (zero-input, no teacher-forcing). 0=off")
ap.add_argument("--reg_len_w", type=float, default=1.0, help="V13 §6: decoded |arc/height| length loss on the REG BASE ONLY. 0=off")
ap.add_argument("--reg_param_weights", default="curl=3,clump=4,arc=1,straight=1,lift_arc=1,tip_spread=1", help="per-param weights for the REG-base decoded loss (V13 §6 curl3/clump4)")
ap.add_argument("--eval_every", type=int, default=2000, help="V13 §6: run the 7-param held-out decode eval every N steps, save-best + early-stop")
ap.add_argument("--early_stop", type=int, default=3, help="V13 §6/§7: stop if held-out 7-param nMAE has not improved for this many evals (over-fits ~it10000). 0=off")
ap.add_argument("--seed", type=int, default=0, help="training stochasticity seed (model init + data order). The held-out SPLIT stays fixed at 0 for fair cross-run eval; vary this for seed-repeats.")
# ===== V14 additions (V14_DIFFUSION_SPEC Parts A/B/D/E/F; experiments E0/E1/E3). All additive, behind flags. =====
ap.add_argument("--diff_mode", choices=["full", "residual"], default="full",
                help="V14 Part E (PRIMARY=full). full: diffuse z64 DIRECTLY (target=standardized x0, sigma_data=1, NO reg base / NO z_base subtraction / NO z_base stem concat; reg/reg_iters/resume-reg machinery bypassed). residual: the V13-2 path (reg base + (x0-z_base)/res_std).")
ap.add_argument("--p_mean", type=float, default=-0.4,
                help="V14 Part A: mean of the log-sigma training draw (sigma=exp(randn*p_std + p_mean)). V13 hardcoded -1.2 (low-sigma refiner); -0.4 puts mass in the high-sigma region. E0 may pass -1.2 to reproduce V13's draw.")
ap.add_argument("--p_std", type=float, default=1.2, help="V14 Part A: std of the log-sigma training draw (V13 hardcoded 1.2).")
ap.add_argument("--drop_group", choices=["none", "appearance", "geometry", "density", "global"], default="none",
                help="V14 Part B GROUP ABLATION: zero ONLY one named condition group EVERY step. appearance={RGBPROJ,G_COND,GLOB}, geometry={raydir,grazing,vis}, density={rho}, global={GLOB}. none=off. (--cond_drop is the separate CFG random ALL-group drop.)")
ap.add_argument("--ema_decay", type=float, default=0.0,
                help="V14 Part D: EMA decay for a shadow copy of the diffusion net (0=off; E1 uses 0.999). When >0 the EMA weights drive eval + are saved.")
ap.add_argument("--score_eval_n", type=int, default=240, help="V14 Part F: number of held-out patches to SAMPLE+decode for the composite-score eval.")
# ===== V16 Clump-Lite additions (ADD 1 + ADD 2). Both default OFF -> E1 path preserved. =====
ap.add_argument("--clump_cw", type=float, default=0.0,
                help="V16 ADD 1: clump-sensitive z64 CHANNEL WEIGHTING. 0=off (E1). >0 = elevated weight on the clump-sensitive z64 "
                     "channels in the LATENT EDM-MSE (NOT a decoded-param loss on D). tier-2.0 set [7,16,24,29,37,38,52,54] gets "
                     "clump_cw; tier-3.0 set [24,29,38,54] gets a higher sub-weight (1.25x, capped at 4); 1.0 elsewhere. Try 1.5/2.0/3.0. "
                     "Channels from patchlocks/v14/diag_clump_channels.json 'suggested'. EXTENDS the legacy --clump_ch_w path (which stays).")
ap.add_argument("--clump_cw_json", default="patchlocks/v14/diag_clump_channels.json",
                help="V16 ADD 1: JSON with the clump-sensitive channel tiers (reads 'suggested'.'tier_channel_sets').")
ap.add_argument("--clump_cw_t3_mul", type=float, default=1.25,
                help="V16 ADD 1: tier-3.0 channels get clump_cw * this (sub-weight above the tier-2.0 weight); whole CW capped at 4.")
ap.add_argument("--clump_bucket", type=int, default=0, choices=[0, 1],
                help="V16 ADD 2: clump-LEVEL bucket sampler. 0=off (E1 hf_oversample weights). 1 = decode GT clump at startup, "
                     "bucket train patches into 4 clump QUARTILES, sample BALANCED across buckets (REPLACES E1's sampling weights). "
                     "A coherence proxy down-weights TANGLED patches so we don't oversample mess. Buckets by clump LEVEL, NOT coat type.")
ap.add_argument("--clump_bucket_tangle_deg", type=float, default=30.0,
                help="V16 ADD 2: deviation threshold (deg) for the per-patch tangled-rate coherence proxy.")
a = ap.parse_args()
if a.b3_pre_iters < 1:
    raise ValueError("--b3_pre_iters must be >= 1; use the old script/version for raw masked-mean conditioning ablations.")
# V14: the legacy V11 --mode {direct,residual} guard stays for anyone forcing --mode; V14's primary knob is --diff_mode.
if a.cfg != 1.0 and a.mode != "direct":
    raise ValueError("--cfg is only supported in --mode direct (legacy V11 path; V14 uses --diff_mode)")
if not (0.0 <= a.cond_drop < 1.0):
    raise ValueError("--cond_drop must be in [0,1)")
if not (0.0 <= a.decode_thr < 1.0):
    raise ValueError("--decode_thr must be in [0,1)")
if not (0.0 <= a.ema_decay < 1.0):
    raise ValueError("--ema_decay must be in [0,1)")
# V16 ADD 1: clump_cw must be non-negative (0=off). The per-channel CW is capped at 4 where it is built.
if a.clump_cw < 0.0:
    raise ValueError("--clump_cw must be >= 0 (0 = off)")
# V14: full mode diffuses z64 directly (no reg base) -> --no_diffusion (reg-only) is meaningless. Use --diff_mode residual for the reg-base path.
if a.diff_mode == "full" and a.no_diffusion:
    raise ValueError("--diff_mode full diffuses z64 directly (no reg base); --no_diffusion (reg-only) is incompatible. Use --diff_mode residual for the V13-2 reg-base path.")
PARAM_KEYS = ["arc", "straight", "lift_arc", "curl", "tip_spread", "clump"]
PARAM_WTS = {k: 1.0 for k in PARAM_KEYS}
if a.param_weights.strip():
    for item in a.param_weights.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"bad --param_weights item {item!r}; expected key=value")
        k, v = item.split("=", 1); k = k.strip()
        if k not in PARAM_WTS:
            raise ValueError(f"unknown --param_weights key {k!r}; allowed {PARAM_KEYS}")
        PARAM_WTS[k] = float(v)
# V13: REG-base decoded-param weights (separate knob from the V11 diffused param_weights)
REG_PARAM_WTS = {k: 1.0 for k in PARAM_KEYS}
if a.reg_param_weights.strip():
    for item in a.reg_param_weights.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"bad --reg_param_weights item {item!r}; expected key=value")
        k, v = item.split("=", 1); k = k.strip()
        if k not in REG_PARAM_WTS:
            raise ValueError(f"unknown --reg_param_weights key {k!r}; allowed {PARAM_KEYS}")
        REG_PARAM_WTS[k] = float(v)
dev = "cuda"; H = a.res; S = a.shells; C = 1024; os.makedirs(a.out, exist_ok=True)
store_dev = torch.device(a.cache_device)
idx_dev = torch.device("cpu" if a.cache_device == "cpu" else dev)
print(f"cache_device={a.cache_device} | eval_batch={a.eval_batch}", flush=True)

dirs = sorted(d for d in glob.glob(a.renders + "/*") if os.path.exists(f"{d}/{a.latent_name}") and os.path.exists(f"{d}/field_gt_{H}.npz") and os.path.exists(f"{d}/cameras.json") and os.path.exists(f"{d}/{a.pose_name}"))
if not dirs:
    raise FileNotFoundError(f"no render dirs with {a.latent_name} + field_gt_{H}.npz + cameras.json + {a.pose_name} under {a.renders}")
if getattr(a, "max_dirs", 0) and len(dirs) > a.max_dirs:
    keep = np.linspace(0, len(dirs) - 1, a.max_dirs).round().astype(int)   # coat-balanced stride (dirs sorted by rd_NNNN = coat-blocked)
    dirs = [dirs[i] for i in sorted(set(keep.tolist()))]
    print(f"--max_dirs {a.max_dirs}: subsampled to {len(dirs)} dirs to bound the in-RAM feature cache", flush=True)
K = len(json.load(open(f"{dirs[0]}/cameras.json"))["views"]); N = len(dirs)
KUSE = 1 if a.single_view else K                                                            # V13: views actually fed to the fuser (single-view baseline vs multi-view)
POSE_DIM = 12                                                                               # V13 pose_view token width (matches add_v13_pose_features.POSE_NAMES)
print(f"{N} patches, K={K}, KUSE={KUSE} ({'SINGLE-VIEW baseline' if a.single_view else 'MULTI-VIEW'}), H={H} | no_diffusion={a.no_diffusion} view_dropout={a.view_dropout}", flush=True)
print(
    f"latent_name={a.latent_name} | codec_path={a.codec_path} | "
    f"len_w={a.len_w} | elev_w={a.elev_w} | param_w={a.param_w} | "
    f"hf_w={a.hf_w} | clump_ch_w={a.clump_ch_w} | len_k={a.len_k}",
    flush=True,
)
if a.param_w > 0:
    print("param_weights=" + ",".join(f"{k}:{PARAM_WTS[k]:g}" for k in PARAM_KEYS), flush=True)
ii, jj = np.meshgrid(np.arange(H), np.arange(H)); ug = (ii + 0.5) / H; vg = (jj + 0.5) / H

def shell_grids(meta, hmax):
    size = meta["size"]; hs = np.linspace(0, 1, S) * hmax; G = np.zeros((K, S, H, H, 2), np.float32); I = np.zeros((K, S, H, H), np.float32)
    for k in range(K):
        cam = meta["views"][k]; res = cam["res"]; Minv = np.linalg.inv(np.array(cam["matrix_world"], np.float64)); fpx = cam["lens"] / cam["sensor_width"] * res
        for si, h in enumerate(hs):
            Xw = np.stack([(ug - .5) * size, (vg - .5) * size, np.full_like(ug, h), np.ones_like(ug)], -1)
            Xc = (Xw.reshape(-1, 4) @ Minv.T)[:, :3].reshape(H, H, 3); z = Xc[..., 2]; fr = z < 0; zz = np.where(fr, z, -1e-6)
            px = res / 2 + fpx * (Xc[..., 0] / -zz); py = res / 2 - fpx * (Xc[..., 1] / -zz)
            G[k, si, ..., 0] = (px / res) * 2 - 1; G[k, si, ..., 1] = (py / res) * 2 - 1; I[k, si] = (fr & (px >= 0) & (px < res) & (py >= 0) & (py < res)).astype(np.float32)
    return G, I

import cv2
def bake(uv, val):                                                                  # count-weighted mean per-strand val -> (H,H,C) + mask
    ti = np.clip((uv[:, 0] * H).astype(int), 0, H - 1); tj = np.clip((uv[:, 1] * H).astype(int), 0, H - 1)
    cnt = np.zeros((H, H), np.float32); np.add.at(cnt, (tj, ti), 1.0)
    acc = np.zeros((H, H, val.shape[1]), np.float32); np.add.at(acc, (tj, ti), val)
    return acc / np.maximum(cnt[..., None], 1e-6), (cnt > 0).astype(np.float32)
def fill_missing(x, mask):                                                          # propagate nearest into empty texels (so bilinear render is defined)
    m = (mask > 0.5).astype(np.float32)
    if m.max() < 0.5: return x.astype(np.float32)
    cur = x.astype(np.float32).copy(); cur *= m[..., None]; known = m.copy(); ker = np.ones((3, 3), np.float32); gk = np.ones((3, 3), np.uint8)
    for _ in range(H * 2):
        if known.min() > 0.5: break
        grow = (cv2.dilate(known.astype(np.uint8), gk) > 0) & (known < 0.5)
        if not grow.any(): break
        den = cv2.filter2D(known, -1, ker, borderType=cv2.BORDER_REPLICATE)
        for c in range(cur.shape[2]):
            num = cv2.filter2D(cur[..., c] * known, -1, ker, borderType=cv2.BORDER_REPLICATE); cur[..., c][grow] = num[grow] / np.maximum(den[grow], 1e-6)
        known[grow] = 1.0
    if known.min() < 0.5: cur[known < 0.5] = x[m > 0.5].mean(0)
    return cur.astype(np.float32)

print("DINOv2 + precompute B3 inputs + targets ...", flush=True)
dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').to(dev).eval()
for p in dino.parameters(): p.requires_grad = False
mn = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]; st = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]; Hf = a.dino_res // 14
GLOBC = 2 * C
FMAP = torch.zeros(N, K, C, Hf, Hf, dtype=torch.float16, device=store_dev)
GRID = torch.zeros(N, K, S, H, H, 2, dtype=torch.float16, device=store_dev); INF = torch.zeros(N, K, S, H, H, dtype=torch.float16, device=store_dev)
CAM = torch.zeros(N, K, 6, device=store_dev); GLOB = torch.zeros(N, GLOBC, device=store_dev)
TZ = torch.zeros(N, 64, H, H, device=store_dev); RHO = torch.zeros(N, 1, H, H, device=store_dev); MASK = torch.zeros(N, 1, H, H, device=store_dev)
LAM = torch.zeros(N, 1, H, H, device=store_dev); DIR = torch.zeros(N, 3, H, H, device=store_dev)
LEN = torch.zeros(N, 1, H, H, device=store_dev); CRL = torch.zeros(N, 1, H, H, device=store_dev); VAL = torch.zeros(N, 1, H, H, device=store_dev)
RGBPROJ = torch.zeros(N, 3, H, H, device=store_dev)                                  # v10: raw-RGB back-projection (pixel detail + color DINO washes out)
COL = torch.zeros(N, 6, H, H, device=store_dev); CMASK = torch.zeros(N, 1, H, H, device=store_dev)   # v10: per-texel root+tip color target
# ===== V13 per-view pose conditioning (from v13_pose_64.npz) =====
RAYDIR = torch.zeros(N, K, 3, H, H, device=store_dev)                                        # per-texel per-view ray surf->cam in TBN
GRAZE = torch.zeros(N, K, 3, H, H, device=store_dev)                                         # [ray.n, |ray.n|, asin(ray.n)] per texel
POSE = torch.zeros(N, K, POSE_DIM, device=store_dev)                                         # per-view pose token (count-free, replaces vemb)
VISF = torch.zeros(N, K, 1, H, H, device=store_dev)                                          # per-view per-texel visibility (frustum_only today; depth-occlusion after re-render)
# ===== V14 (spec #1): raydir-in-TBN as EXPLICIT per-texel UNet-stem condition channels. =====
# RAYC = view-reduced [raydir_tbn(3) + grazing(3) + vis(1)] = 7ch. Built from the SAME RAYDIR/GRAZE/VISF the
# fuser consumes and persists past the per-view-tensor `del` so it can be a clean stem condition (droppable by CFG).
RAYC_CH = 7
RAYC = torch.zeros(N, RAYC_CH, H, H, device=store_dev)

def _idx_for(x, idx):
    return idx.detach().cpu() if x.device.type == "cpu" else idx.to(x.device)

def raw_take(x, idx):
    return x[_idx_for(x, idx)]

def fetch(x, idx, dtype=None):
    y = raw_take(x, idx)
    if y.device.type != "cuda":
        y = y.to(dev, non_blocking=True)
    return y.to(dtype=dtype) if dtype is not None else y
with torch.no_grad():
    for i, d in enumerate(dirs):
        meta = json.load(open(f"{d}/cameras.json"))
        ims = [__import__("cv2").resize(__import__("cv2").cvtColor(__import__("cv2").imread(f"{d}/rgb_{k}.png", -1), 4), (a.dino_res, a.dino_res)).astype(np.float32) / 255 for k in range(K)]
        im = torch.tensor(np.stack(ims)).permute(0, 3, 1, 2).to(dev)
        feat = dino.forward_features((im - mn) / st)
        tok = feat["x_norm_patchtokens"]                                                   # (K,P,C)
        cls = feat["x_norm_clstoken"]                                                       # (K,C)
        fmap = tok.transpose(1, 2).reshape(K, C, Hf, Hf)
        FMAP[i] = fmap.half().to(store_dev)
        GLOB[i] = torch.cat([tok.mean((0, 1)), cls.mean(0)], 0).to(store_dev)                 # DiffLocks-style mean local + CLS global
        p = np.load(f"{d}/patch.npz", allow_pickle=True); hmax = float(max(p["strands"][..., 2].max(), 1e-3))
        Gg, I = shell_grids(meta, hmax)
        Gg_t = torch.tensor(Gg, device=dev).half(); I_t = torch.tensor(I, device=dev).half()
        GRID[i] = Gg_t.to(store_dev); INF[i] = I_t.to(store_dev)
        g0 = Gg_t[:, 0].float(); samp = F.grid_sample(im.float(), g0, align_corners=False); inf0 = I_t[:, 0].float()[:, None]   # raw RGB at scalp shell
        RGBPROJ[i] = ((samp * inf0).sum(0) / (inf0.sum(0) + 1e-6)).to(store_dev)
        # v10 LIT-COLOR target (well-posed): the color target is the LIT color projected from the SAME input image,
        # NOT canonical albedo (which the model can't de-light from a lit input -> didn't generalize, val 0.071 >
        # mean-baseline 0.026). root = lit color at the scalp texel (= RGBPROJ); tip = input image sampled at the
        # GT strand-tip projection, averaged over views where the tip is visible. Color-valid = root exists AND visible.
        _uv = p["root_uv"].astype(np.float32); _, _am = bake(_uv, np.ones((len(_uv), 1), np.float32))
        vis0 = (inf0.sum(0)[0] > 0).float().cpu().numpy()                                    # texel visible in >=1 view
        tips = p["strands"].astype(np.float32)[:, -1, :]                                     # (M,3) GT strand tips (patch-local)
        tcol = np.zeros((len(tips), 3), np.float32); tw = np.zeros((len(tips), 1), np.float32)
        for k, cam in enumerate(meta["views"]):
            res = cam["res"]; Minv = np.linalg.inv(np.array(cam["matrix_world"], np.float64)); fpx = cam["lens"] / cam["sensor_width"] * res
            Xc = (np.concatenate([tips, np.ones((len(tips), 1))], 1) @ Minv.T)[:, :3]
            zc = Xc[:, 2]; fr = zc < 0; zz = np.where(fr, zc, -1e-6)
            px = res / 2 + fpx * (Xc[:, 0] / -zz); py = res / 2 - fpx * (Xc[:, 1] / -zz)
            inb = fr & (px >= 0) & (px < res) & (py >= 0) & (py < res)
            grid = torch.tensor(np.stack([px / res * 2 - 1, py / res * 2 - 1], 1)[None, None], device=dev).float()
            ssamp = F.grid_sample(im[k:k + 1].float(), grid, align_corners=False)[0, :, 0].T.cpu().numpy()   # (M,3)
            tcol += ssamp * inb[:, None]; tw += inb[:, None]
        _ct, _ = bake(_uv, (tcol / (tw + 1e-6)).astype(np.float32))
        rgbp_np = RGBPROJ[i].permute(1, 2, 0).cpu().numpy()                                   # root lit color per texel (= RGBPROJ)
        COL[i] = torch.tensor(np.concatenate([rgbp_np, fill_missing(_ct, _am)], -1), device=store_dev).permute(2, 0, 1)
        CMASK[i, 0] = torch.tensor(_am * vis0, device=store_dev)                              # color valid = root exists AND texel visible
        loc = [np.array(v["location"], np.float32) for v in meta["views"]]; vd = [-np.array(v["matrix_world"], np.float32)[:3, 2] for v in meta["views"]]
        CAM[i] = torch.tensor(np.concatenate([np.stack(loc), np.stack(vd)], 1), device=store_dev)
        # ---- V13 pose sidecar: raydir-in-TBN + grazing + pose token + per-view vis (recipe §1a/§5) ----
        pz = np.load(f"{d}/{a.pose_name}", allow_pickle=True)
        rd = np.asarray(pz["raydir_tbn"], np.float32)                                         # (K,Hp,Hp,3)
        if rd.shape[1] != H:                                                                  # pose baked at a different res than --res
            raise ValueError(f"{d}/{a.pose_name} raydir_tbn H={rd.shape[1]} != --res {H}")
        RAYDIR[i] = torch.tensor(rd, device=store_dev).permute(0, 3, 1, 2)                    # (K,3,H,H)
        rn = np.clip(rd[..., 2], -1.0, 1.0)                                                   # grazing = ray.n
        gr = np.stack([rn, np.abs(rn), np.arcsin(rn)], -1).astype(np.float32)                 # (K,H,H,3)
        GRAZE[i] = torch.tensor(gr, device=store_dev).permute(0, 3, 1, 2)
        POSE[i] = torch.tensor(np.asarray(pz["pose_view"], np.float32), device=store_dev)     # (K,12)
        vf = np.asarray(pz["vis_frustum"], np.float32)[:, None]                               # (K,1,H,H)
        VISF[i] = torch.tensor(vf, device=store_dev)
        # ---- V14: view-reduce raydir/grazing/vis -> RAYC (7ch), CONSISTENT with the MVFuser view use ----
        # Use the SAME KUSE views the fuser consumes (single_view -> view 0 only). raydir(3)+grazing(3) =
        # visibility-WEIGHTED mean over the used views (matches the fuser's vis-masked softmax intent);
        # vis(1) = max over used views (texel seen by ANY used view). Falls back to a plain mean where no
        # used view sees the texel (wsum~0) -> still a defined, finite condition.
        vfr = np.asarray(pz["vis_frustum"], np.float32)                                       # (K,H,H)
        rd6 = np.concatenate([rd, gr], -1).astype(np.float32)                                 # (K,H,H,6) raydir(3)+grazing(3)
        wv = vfr[:KUSE]                                                                       # (KUSE,H,H) per-view vis weight
        wsum = wv.sum(0)                                                                       # (H,H)
        seen = wsum > 1e-6
        red6 = np.where(seen[..., None],
                        (rd6[:KUSE] * wv[..., None]).sum(0) / np.maximum(wsum[..., None], 1e-6),
                        rd6[:KUSE].mean(0))                                                    # vis-weighted mean where seen, else plain mean
        visred = np.clip(wv.max(0), 0.0, 1.0)                                                  # (H,H) seen by any used view
        rayc = np.concatenate([red6, visred[..., None]], -1).astype(np.float32)               # (H,H,7)
        RAYC[i] = torch.tensor(rayc, device=store_dev).permute(2, 0, 1)                        # (7,H,H)
        tl = np.load(f"{d}/{a.latent_name}")
        if tl["target"].shape[:2] != (H, H):
            raise ValueError(f"{d}/{a.latent_name} has target shape {tl['target'].shape[:2]}, but --res is {H}. Re-bake targets or use matching --res.")
        TZ[i] = torch.tensor(tl["target"], device=store_dev).permute(2, 0, 1)
        rho = tl["density"]; rho = rho[..., 0] if rho.ndim == 3 else rho
        msk = tl["mask"]; msk = msk[..., 0] if msk.ndim == 3 else msk
        RHO[i, 0] = torch.tensor(np.asarray(rho, np.float32), device=store_dev)
        MASK[i, 0] = torch.tensor(np.asarray(msk, np.float32), device=store_dev)
        fg = np.load(f"{d}/field_gt_{H}.npz")
        LAM[i, 0] = torch.tensor(fg["lambda_map"], device=store_dev); DIR[i] = torch.tensor(fg["direction_map"], device=store_dev).permute(2, 0, 1)
        LEN[i, 0] = torch.tensor(fg["length_mu"][..., 0], device=store_dev); CRL[i, 0] = torch.tensor(fg["curl_mu"][..., 0], device=store_dev)
        VAL[i, 0] = torch.tensor(fg["valid_mask"], device=store_dev)
        if (i + 1) % 100 == 0: print(f"  {i+1}/{N}", flush=True)
del dino; torch.cuda.empty_cache()

g_ = torch.Generator().manual_seed(0)
# PATCH-LEVEL split (small data): hold out WHOLE patches (all 4 views) -> no view leakage, honest overfit measure.
_pid = [os.path.basename(d).rsplit("_v", 1)[0] for d in dirs]                                # rd_XXXX_vK -> rd_XXXX
_uniq = sorted(set(_pid)); _pp = torch.randperm(len(_uniq), generator=g_).tolist()
_valset = set(_uniq[_pp[i]] for i in range(max(1, int(len(_uniq) * 0.15))))
_isval = torch.tensor([p in _valset for p in _pid])
if a.overfit:
    perm = torch.randperm(N, generator=g_); ti = perm[:a.overfit].to(idx_dev); vi = ti.clone()
else:
    vi = torch.nonzero(_isval, as_tuple=True)[0].to(idx_dev); ti = torch.nonzero(~_isval, as_tuple=True)[0].to(idx_dev)
print(f"PATCH-LEVEL split: {len(_uniq)} patches -> {len(_valset)} val patches ({len(vi)} dirs) / {len(ti)} train dirs (no view leakage)", flush=True)
HF_ARCH = (
    "wiry_terrier", "clumped_wet", "dog_short_double",                                      # old v5 names
    "stiff_double", "curly_teddy", "wavy_long", "long_straight", "wiry",                   # v6 redesigned hard coats
)
_names = [os.path.basename(d) for d in dirs]
_wfull = torch.tensor([a.hf_oversample if any(h in _names[i] for h in HF_ARCH) else 1.0 for i in range(N)], device=idx_dev)
w_ti = _wfull[ti]                                                                            # batch-sampling weights aligned to train indices
print(f"hf_oversample={a.hf_oversample} -> {int((w_ti > 1).sum())}/{len(ti)} train patches up-weighted | hf_w={a.hf_w} spec_w={a.spec_w}", flush=True)

def fstat(x):
    m = raw_take(VAL, ti) > 0.5; v = raw_take(x, ti)[m]
    return v.float().mean().to(dev), v.float().std(unbiased=False).clamp_min(1e-6).to(dev)
lam_m, lam_s = fstat(LAM); len_m, len_s = fstat(LEN); crl_m, crl_s = fstat(CRL)

class MVFuser(nn.Module):
    """V13 multi-view fuser (replaces B3Fuser). Closes the TBN coordinate-frame loop (§1b):

    Per (view k, shell s, texel) token =
        proj(backproj DINO feature)                         # appearance evidence
      + raydir_proj([raydir_local(3), grazing(3)])          # observation direction IN TBN (the new condition §1a)
      + pose_emb(pose_view[k] -> MLP)                        # count-free per-view pose token (REPLACES vemb Parameter §5)
      + semb[s]                                             # canonical shell index (kept)
    reduced over the V*S tokens via a VISIBILITY-MASKED softmax (-1e4 on vis<0.5) + a learned `unseen`
    token used where no view sees a texel. Accepts VARIABLE V (the `views` arg) so train view-dropout
    V in {1..K} and deploy V>K both work -- no Parameter pins the view count.
    """
    def __init__(s, dim, nsh, pose_dim, nb=6):
        super().__init__()
        s.nsh = nsh; s.proj = nn.Conv2d(C, dim, 1)
        s.raydir_proj = nn.Conv2d(6, dim, 1)                                                 # raydir_local(3)+grazing(3) -> dim, per texel
        s.semb = nn.Parameter(torch.randn(nsh, dim) * 0.02)
        s.pose = nn.Sequential(nn.Linear(pose_dim, dim), nn.GELU(), nn.Linear(dim, dim))     # pose token (count-free) replaces vemb
        s.unseen = nn.Parameter(torch.randn(dim) * 0.02)                                     # learned token for texels seen by NO view
        s.attn = nn.Conv2d(dim, 1, 1)
        s.body = nn.ModuleList([nn.Sequential(nn.GroupNorm(8, dim), nn.SiLU(), nn.Conv2d(dim, dim, 3, 1, 1)) for _ in range(nb)])
    def forward(s, bi, views=None):
        B = len(bi)
        V = K if views is None else int(views)                                              # number of views to fuse (<=K at train, can exceed K at deploy)
        vsel = torch.arange(V, device=idx_dev) % K                                           # view indices (wraps if V>K)
        fmap = fetch(FMAP, bi)[:, vsel].reshape(B * V, C, Hf, Hf).float()
        grid = fetch(GRID, bi)[:, vsel][:, :, :s.nsh].reshape(B * V, s.nsh * H, H, 2).float()
        samp = F.grid_sample(fmap, grid, mode="bilinear", align_corners=False).reshape(B, V, C, s.nsh, H, H)
        x = s.proj(samp.permute(0, 1, 3, 2, 4, 5).reshape(B * V * s.nsh, C, H, H)).reshape(B, V, s.nsh, -1, H, H)
        # per-texel raydir-in-TBN + grazing (shell-independent -> broadcast over S)
        rd6 = torch.cat([fetch(RAYDIR, bi)[:, vsel], fetch(GRAZE, bi)[:, vsel]], 2)          # (B,V,6,H,H)
        re = s.raydir_proj(rd6.reshape(B * V, 6, H, H)).reshape(B, V, 1, -1, H, H)
        pe = s.pose(fetch(POSE, bi)[:, vsel].reshape(B * V, -1)).reshape(B, V, 1, -1, 1, 1)  # per-view pose token
        x = x + re + pe + s.semb[None, None, :, :, None, None]
        # visibility mask = (frustum/occlusion vis) per view, broadcast over shells; texel is "seen" if ANY view*shell sees it
        vis = fetch(VISF, bi)[:, vsel]                                                       # (B,V,1,H,H)
        inf = fetch(INF, bi)[:, vsel][:, :, :s.nsh, None]                                    # (B,V,S,1,H,H) frustum (shell ray in-frame)
        m = (vis[:, :, None] > 0.5).float() * (inf > 0.5).float()                            # (B,V,S,1,H,H) = vis AND in-frustum
        x = x.reshape(B, V * s.nsh, -1, H, H); m = m.reshape(B, V * s.nsh, 1, H, H)
        logits = s.attn(x.reshape(B * V * s.nsh, -1, H, H)).reshape(B, V * s.nsh, 1, H, H).masked_fill(m < 0.5, -1e4)
        w = torch.softmax(logits, 1)
        seen = (m.sum(1, keepdim=True) > 0).float()                                          # (B,1,1,H,H)
        w = w * seen
        fused = (x * w).sum(1) + (1.0 - seen[:, 0]) * s.unseen[None, :, None, None]          # learned unseen token where nothing sees the texel
        for b in s.body: fused = fused + b(fused)
        return fused

class FieldHead(nn.Module):
    def __init__(s, dim):
        super().__init__(); s.out = nn.Sequential(nn.GroupNorm(8, dim), nn.SiLU(), nn.Conv2d(dim, 6, 1))
    def forward(s, x): return s.out(x)

def split_field(o):
    dv = o[:, 1:4]
    return o[:, :1], dv / dv.norm(dim=1, keepdim=True).clamp_min(1e-6), o[:, 4:5], o[:, 5:6]
def grad(x):
    gx = x[..., :, 1:] - x[..., :, :-1]; gy = x[..., 1:, :] - x[..., :-1, :]
    return F.pad(gx, (0, 1)), F.pad(gy, (0, 0, 1, 0))
def field_loss(o, bi):
    lam, dv, ln, cr = split_field(o); m = fetch(VAL, bi); w = m.sum() + 1e-6
    tl = (fetch(LAM, bi) - lam_m) / lam_s; tn = (fetch(LEN, bi) - len_m) / len_s; tc = (fetch(CRL, bi) - crl_m) / crl_s
    base = (F.smooth_l1_loss(lam * m, tl * m) + F.smooth_l1_loss(ln * m, tn * m) + F.smooth_l1_loss(cr * m, tc * m)
            + 2.0 * ((1 - (dv * fetch(DIR, bi)).sum(1, keepdim=True)) * m).sum() / w)
    gl = 0.0
    for pr, tg in [(lam, tl), (ln, tn), (cr, tc)]:
        pgx, pgy = grad(pr); tgx, tgy = grad(tg)
        gl = gl + F.smooth_l1_loss(pgx * m, tgx * m) + F.smooth_l1_loss(pgy * m, tgy * m)
    return base + 0.5 * gl

print(f"pretrain learned B3 fuser for {a.b3_pre_iters} steps, then freeze G_COND ...", flush=True)
torch.manual_seed(a.seed); b3 = MVFuser(a.dim, S, POSE_DIM).to(dev); b3h = FieldHead(a.dim).to(dev)
def _rand_views():                                                                          # V13 §5 view-dropout: random V in {1..KUSE} during fuser pretrain (1->many robust); else all KUSE
    return int(torch.randint(1, KUSE + 1, (1,)).item()) if (a.view_dropout and KUSE > 1) else KUSE
optb = torch.optim.AdamW(list(b3.parameters()) + list(b3h.parameters()), 3e-4, weight_decay=1e-4)
schb = torch.optim.lr_scheduler.CosineAnnealingLR(optb, max(a.b3_pre_iters, 1))
best, bvl = None, 1e9; every = max(1, min(500, a.b3_pre_iters))
b3_path = f"{a.out}/b3_fuser.pt"
resumed_b3 = False
if a.resume_b3 and os.path.exists(b3_path):
    ck = torch.load(b3_path, map_location=dev)
    if ck.get("dim") == a.dim and ck.get("shells") == S:
        b3.load_state_dict(ck["b3"]); b3h.load_state_dict(ck["field_head"])
        bvl = float(ck.get("best_val", 0.0)); resumed_b3 = True
        print(f"  [B3-fuser] resumed {b3_path} | val {bvl:.3f}", flush=True)
    else:
        print("  [B3-fuser] resume skipped: checkpoint dim/shell mismatch", flush=True)
if not resumed_b3:
    for it in range(1, a.b3_pre_iters + 1):
        bi = ti[torch.randint(0, len(ti), (a.bs,), device=idx_dev)]
        loss = field_loss(b3h(b3(bi, _rand_views())), bi); optb.zero_grad(); loss.backward(); optb.step(); schb.step()
        if it % every == 0:
            with torch.no_grad():
                vals = [field_loss(b3h(b3(vi[s:s+32], KUSE)), vi[s:s+32]).item() for s in range(0, len(vi), 32)]
                vl = sum(vals) / max(len(vals), 1)
            if vl < bvl:
                bvl = vl; best = {"b3": {k: v.detach().clone() for k, v in b3.state_dict().items()}, "h": {k: v.detach().clone() for k, v in b3h.state_dict().items()}}
            print(f"  [B3-fuser] it {it:4d} | train {loss.item():.3f} | val {vl:.3f}", flush=True)
    if best is not None:
        b3.load_state_dict(best["b3"]); b3h.load_state_dict(best["h"])
        torch.save({"b3": best["b3"], "field_head": best["h"], "dim": a.dim, "shells": S, "best_val": bvl}, b3_path)
b3.eval()
gbs = max(1, int(a.gcond_batch))
print(f"precompute frozen G_COND with batch {gbs} ...", flush=True)
with torch.inference_mode():
    # Build G_COND on CPU so its ~10GB (H64) never coincides with the 42GB FMAP on GPU -> avoids OOM at H64.
    G_COND = torch.cat([b3(torch.arange(s, min(s + gbs, N), device=idx_dev), KUSE).half().cpu() for s in range(0, N, gbs)], 0)
del b3, b3h, optb, schb, FMAP, GRID, INF, CAM, LAM, DIR, LEN, CRL, VAL, RAYDIR, GRAZE, POSE, VISF; torch.cuda.empty_cache()
if a.cache_device == "cuda":
    G_COND = G_COND.to(dev)                                                                  # FMAP etc freed -> safe to put G_COND on GPU now
    RAYC = RAYC.to(dev)                                                                      # V14: keep the explicit geom condition on GPU alongside the other conditions

# Normalize conditioning on the train split. Raw DINO channel scale can otherwise dominate or vanish
# relative to the noised texture input, weakening the effective condition.
gm_raw = raw_take(G_COND, ti).float().mean((0, 2, 3), keepdim=True); gs_raw = raw_take(G_COND, ti).float().std((0, 2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)
gm, gs = gm_raw.detach().to(dev), gs_raw.detach().to(dev)
G_COND = ((G_COND.float() - gm_raw) / gs_raw).half()
glm_raw = raw_take(GLOB, ti).float().mean(0, keepdim=True); gls_raw = raw_take(GLOB, ti).float().std(0, keepdim=True, unbiased=False).clamp_min(1e-4)
glm, gls = glm_raw.detach().to(dev), gls_raw.detach().to(dev)
GLOB = (GLOB - glm_raw) / gls_raw
# standardize target channels (per-channel) over train; sigma_data=1 after
# v6: z64-ONLY diffusion. rho is NOT diffused; it is a deterministic CONDITION (log1p, fed clean to the UNet)
# and drives root sampling at decode. Target X = z64 (64ch).
# v10: JOINT target X = [z64 geometry (64) + root/tip color (6)] = 70ch. Diffusion conditioned on RAW PIXELS
# (RGBPROJ, the detail+color source) + DINO (semantic assist) -> responsibility for detail AND color back on the diffusion.
XD = 64 if a.no_color else 70                                                               # --no_color: z64-only ablation
X = TZ if a.no_color else torch.cat([TZ, COL], 1)                                            # (N,XD,H,W)
m = raw_take(MASK, ti) > 0.5; flat = raw_take(X, ti).permute(1, 0, 2, 3)[:, m[:, 0]]         # (XD, Mpix)
xm_raw = flat.float().mean(1)[None, :, None, None]; xs_raw = flat.float().std(1, unbiased=False).clamp_min(1e-4)[None, :, None, None]
xm, xs = xm_raw.detach().to(dev), xs_raw.detach().to(dev)
Xn = (X - xm_raw) / xs_raw                                                                  # standardized z64+color target
RGBPROJ = (RGBPROJ - 0.5) / 0.25                                                             # center the raw-pixel condition (~unit)
RHOC = torch.log1p(RHO.clamp_min(0))                                                         # rho CONDITION (clean, not noised)
# V14: RAYC (raydir 3 + grazing[dot,|dot|,asin] 3 + vis 1) is already physically ~unit-scaled
# (unit ray dir, dot in [-1,1], asin in [-pi/2,pi/2], vis in [0,1]) -> fed CLEAN, no standardization.
# EDM sigma_data. SD=1 is correct for the V11/full-z target; for the V13-2 RESIDUAL it is RE-ESTIMATED
# after reg warm-up (residual std << x0 std -> SD=1 makes the EDM weighting useless, spec §5/point2).
# res_std = per-channel std of r = x0 - z_base.detach(); the residual target is standardized by it
# (-> ~unit per channel) and the sampler rescales back: z = z_base + denoised_residual * res_std.
SD = 1.0; Gdim = 256
res_std = torch.ones(1, XD, 1, 1, device=dev)                                                # per-channel residual std (XD-vector), set in estimate_res_std()
print(f"train {len(ti)} val {len(vi)} | v10 PIXEL+DINO diffusion: {'z64-ONLY(64ch) [--no_color]' if a.no_color else 'z64(64)+COLOR(6)=70ch'} | std~{xs[0].mean():.3f}", flush=True)

class Res(nn.Module):
    def __init__(s, ci, co, ed):
        super().__init__(); s.n1 = nn.GroupNorm(8, ci); s.c1 = nn.Conv2d(ci, co, 3, 1, 1); s.n2 = nn.GroupNorm(8, co); s.c2 = nn.Conv2d(co, co, 3, 1, 1)
        s.emb = nn.Linear(ed, 2 * co); s.sk = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
    def forward(s, x, e):
        h = s.c1(F.silu(s.n1(x))); sh = s.emb(e)[:, :, None, None]; sc, bi = sh.chunk(2, 1); h = s.c2(F.silu(s.n2(h) * (1 + sc) + bi)); return s.sk(x) + h

class BottleneckAttn(nn.Module):
    """Patch-level self-attention at the lowest UNet resolution.

    Long/curly/fluffy fur needs patch-wide consistency (parallel flow, coherent
    swatches).  The old pure-conv bottleneck can regress these structures into
    local felt.  Attention is zero-output initialized, so the network starts as
    the clean V10 UNet and only uses attention if training finds it useful.
    """
    def __init__(s, ch, heads=4):
        super().__init__()
        s.n = nn.GroupNorm(8, ch)
        s.a = nn.MultiheadAttention(ch, heads, batch_first=True)
        nn.init.zeros_(s.a.out_proj.weight)
        nn.init.zeros_(s.a.out_proj.bias)
    def forward(s, x):
        b, c, h, w = x.shape
        q = s.n(x).flatten(2).transpose(1, 2)
        y = s.a(q, q, q, need_weights=False)[0].transpose(1, 2).reshape(b, c, h, w)
        return x + y

class UNet(nn.Module):
    def __init__(s, dim=160, ed=256, cond_ch=None, use_attn=True, cond_zbase=False, cond_raydir=True):
        super().__init__(); cond_ch = cond_ch or C
        s.cond_zbase = bool(cond_zbase)                                                      # V13-2: widen stem to condition the residual UNet on z_base.detach()
        s.cond_raydir = bool(cond_raydir)                                                    # V14 (spec #1): widen stem with explicit raydir-in-TBN + grazing + vis (7ch)
        s.gp = nn.Conv2d(cond_ch, Gdim, 1); s.gp2 = nn.Conv2d(cond_ch, dim, 1)               # gp2: G injected at mid scale
        uv = torch.tensor(np.stack([ug * 2 - 1, vg * 2 - 1], 0), dtype=torch.float32)
        s.register_buffer("uv", uv[None])
        # V13-2: +XD channels when cond_zbase -> noised RESIDUAL is conditioned on the deterministic base (§5 widen s.inp)
        # V14: +RAYC_CH(7) when cond_raydir -> raydir_tbn(3)+grazing(3)+vis(1) explicit stem condition (spec #1). Droppable by CFG.
        s.inp = nn.Conv2d(XD + (XD if cond_zbase else 0) + 1 + 3 + Gdim + 2 + (RAYC_CH if cond_raydir else 0), dim, 3, 1, 1)  # noisy[z64(+color)](XD)[+z_base(XD)] + rho(1) + RGBPROJ(3) + G + uv [+ raydir/grazing/vis(7)]
        s.semb = nn.Sequential(nn.Linear(1, ed), nn.SiLU(), nn.Linear(ed, ed)); s.gemb = nn.Linear(GLOBC, ed)
        s.d1 = Res(dim, dim, ed); s.d2 = Res(dim, 2 * dim, ed); s.mid = Res(2 * dim + dim, 2 * dim, ed)
        s.attn = BottleneckAttn(2 * dim) if use_attn else nn.Identity()
        s.u2 = Res(4 * dim, dim, ed); s.u1 = Res(2 * dim, dim, ed); s.out = nn.Conv2d(dim, XD, 3, 1, 1)
        s.dn = nn.AvgPool2d(2); s.up = nn.Upsample(scale_factor=2, mode="nearest")
    def forward(s, x, cnoise, gc, gl, rhoc, rgbp, zbase=None, rayc=None):
        gcf = gc.float(); e = s.semb(cnoise[:, None]) + s.gemb(gl.float())
        feats = [x]
        if s.cond_zbase:                                                                     # condition residual denoiser on the (detached) base
            feats.append((torch.zeros_like(x) if zbase is None else zbase).float())
        feats += [rhoc, rgbp.float(), s.gp(gcf), s.uv.expand(len(x), -1, -1, -1)]            # RAW PIXELS + rho + G as condition channels
        if s.cond_raydir:                                                                    # V14: explicit per-texel view geometry (raydir/grazing/vis). None -> zeros (CFG/uncond null).
            rc = torch.zeros(len(x), RAYC_CH, x.shape[-2], x.shape[-1], device=x.device, dtype=torch.float32) if rayc is None else rayc.float()
            feats.append(rc)
        h0 = s.inp(torch.cat(feats, 1))
        a1 = s.d1(h0, e); a2 = s.d2(s.dn(a1), e)
        gmid = s.gp2(F.avg_pool2d(gcf, 4))                                                  # G re-injected at mid scale (multi-scale condition)
        mm = s.mid(torch.cat([s.dn(a2), gmid], 1), e)
        mm = s.attn(mm)
        b2 = s.u2(torch.cat([s.up(mm), a2], 1), e); b1 = s.u1(torch.cat([s.up(b2), a1], 1), e); return s.out(b1)

def precond(net, x, sig, gc, gl, rhoc, rgbp, zbase=None, rayc=None):
    sig = sig.reshape(-1, 1, 1, 1); cs = SD**2 / (sig**2 + SD**2); co = sig * SD / (sig**2 + SD**2).sqrt(); ci = 1 / (sig**2 + SD**2).sqrt(); cn = sig.flatten().log() / 4
    return cs * x + co * net(ci * x, cn, gc, gl, rhoc, rgbp, zbase, rayc)

use_attn = not a.no_attn
a.attn = int(use_attn)  # ckpt metadata consumed by infer_v10_surfpack_proposals.py
# ===== V14 Part E: diffusion mode. FULL = diffuse z64 directly (no reg base, no z_base concat). RESIDUAL = V13-2 path. =====
FULL = (a.diff_mode == "full")
# The denoiser stem condition on z_base is ON only in RESIDUAL mode (FULL has no base to condition on).
NET_COND_ZBASE = (not FULL) and (not a.no_diffusion)
print(f"V14 denoiser: diff_mode={a.diff_mode} unet_dim={a.unet_dim} bottleneck_attn={use_attn} | cond_zbase={NET_COND_ZBASE} cond_raydir=True (+{RAYC_CH}ch) | "
      f"p_mean={a.p_mean} p_std={a.p_std} sigma_max={a.sigma_max} cond_drop={a.cond_drop} drop_group={a.drop_group} cfg={a.cfg} ema_decay={a.ema_decay}", flush=True)
print(f"V16 Clump-Lite levers: clump_cw={a.clump_cw} (ADD1 z64-MSE channel weighting, 0=off) | "
      f"clump_bucket={a.clump_bucket} (ADD2 clump-quartile balanced sampler, 0=off) | "
      f"ADD3 coherence+strong-attr eval=ON. E1 recipe otherwise INTACT.", flush=True)
# V14: cond_raydir=True on BOTH nets -> raydir/grazing/vis is an explicit stem condition (spec #1).
net = UNet(a.unet_dim, cond_ch=a.dim, use_attn=use_attn, cond_zbase=NET_COND_ZBASE, cond_raydir=True).to(dev)
reg = UNet(a.unet_dim, cond_ch=a.dim, use_attn=False, cond_raydir=True).to(dev)             # reg = conditional-mean baseline (RESIDUAL mode only); no attention to save compute
BASE_LR = 2e-4
opt = torch.optim.AdamW(net.parameters(), BASE_LR, weight_decay=1e-4)
optr = torch.optim.AdamW(reg.parameters(), BASE_LR, weight_decay=1e-4)
start_it = 0
resumed_reg_only = False                                                                     # set when a reg-only base is loaded for fresh residual training
if a.resume_stage3:
    _resume = torch.load(a.resume_stage3, map_location=dev, weights_only=False)
    try:                                                                                     # V14: in FULL mode a ckpt may have no usable reg; tolerate it
        reg.load_state_dict(_resume["reg"])
    except (RuntimeError, KeyError):
        print("  [stage3] reg weights not loaded (FULL-mode ckpt or shape mismatch) -- ok", flush=True)
    try:
        net.load_state_dict(_resume["net"])
        start_it = int(_resume.get("iter", 0))
        print(f"  [stage3] resumed net+reg {a.resume_stage3} @ it {start_it}", flush=True)
    except RuntimeError:
        # Stem mismatch = resuming a REG-ONLY base (no_diffusion=True ckpt) into a residual run (cond_zbase stem +XD).
        # Keep the converged reg, train the residual net from scratch ON TOP of it (§1 item1: don't re-warm reg).
        resumed_reg_only = True
        print(f"  [stage3] resumed REG base only from {a.resume_stage3} (net stem differs); "
              f"residual net trains fresh on top, reg warm-up SKIPPED", flush=True)
# ===== V14 Part D: EMA shadow of the diffusion net (eval + save the EMA weights when on). =====
ema = None
if a.ema_decay > 0:
    import copy as _copy
    ema = _copy.deepcopy(net).eval()
    for _p in ema.parameters(): _p.requires_grad_(False)
    print(f"  [ema] shadow net created, decay={a.ema_decay} (eval + save use EMA weights)", flush=True)
@torch.no_grad()
def ema_update():
    if ema is None: return
    d = a.ema_decay
    for es, ps in zip(ema.state_dict().values(), net.state_dict().values()):
        if es.dtype.is_floating_point: es.mul_(d).add_(ps.detach(), alpha=1 - d)
        else: es.copy_(ps)
def eval_net():                                                                              # V14: eval/sample/save use EMA weights when EMA is on, else the live net
    return ema if ema is not None else net
def set_cos_lr(abs_step):
    # Absolute-step cosine, robust for stage3 resume. Iteration `it` uses the LR
    # after `it-1` completed steps, so resuming at it=20000 continues near 5e-5
    # instead of restarting AdamW at 2e-4.
    t = min(max(int(abs_step), 0), max(int(a.iters), 1))
    lr = BASE_LR * 0.5 * (1.0 + np.cos(np.pi * t / max(float(a.iters), 1.0)))
    for _opt in (opt, optr):
        for _pg in _opt.param_groups:
            _pg["lr"] = float(lr)
    return float(lr)
resume_lr = set_cos_lr(start_it)
if start_it:
    print(f"  [stage3] resume_lr {resume_lr:.6g}", flush=True)
zc = lambda B: torch.zeros(B, device=dev)
# V13: reg warm-up length. --no_diffusion => reg is the whole run; else warm reg before residual target switches on (§8 risk6).
# When a converged reg base is resumed (resumed_reg_only), skip the warm-up entirely -> residual trains from step 1.
# V14 FULL mode: no reg base at all -> REG_ITERS=0 (diffuse z64 from step 1, reg machinery bypassed).
REG_ITERS = 0 if (FULL or resumed_reg_only) else (a.reg_iters if a.reg_iters > 0 else (a.iters if a.no_diffusion else min(a.iters, 8000)))
print(f"V14 staging: diff_mode={a.diff_mode} no_diffusion={a.no_diffusion} | REG_ITERS={REG_ITERS} | reg_param_w={a.reg_param_w} reg_len_w={a.reg_len_w} | "
      f"reg_param_weights=" + ",".join(f"{k}:{REG_PARAM_WTS[k]:g}" for k in PARAM_KEYS), flush=True)
def save_stage3(path, step):
    tmp = f"{path}.tmp"
    # V14: when EMA is on, the saved diffusion net IS the EMA shadow (deploy/eval target, Part D).
    net_sd = (eval_net()).state_dict()
    torch.save({
        "net": net_sd, "reg": reg.state_dict(), "args": vars(a), "iter": int(step),
        "unet_dim": int(a.unet_dim), "attn": int(use_attn),
        "xm": xm.detach().cpu(), "xs": xs.detach().cpu(), "xd": XD, "color_dim": 0 if a.no_color else 6,   # target = z64(64)[+color(6)]
        "cond_mean": gm.detach().cpu(), "cond_std": gs.detach().cpu(),
        "glob_mean": glm.detach().cpu(), "glob_std": gls.detach().cpu(),
        # ---- V13 ckpt contract additions (consumed by deploy; reject V10/vemb ckpts) ----
        "v13": True, "kuse": int(KUSE), "single_view": bool(a.single_view),
        "no_diffusion": bool(a.no_diffusion), "pose_dim": int(POSE_DIM),
        # V13-2 residual contract: deploy denoises the residual conditioned on z_base and rescales by res_std.
        "cond_zbase": bool(NET_COND_ZBASE), "res_std_ready": bool(res_std_ready),
        "edm_sd": float(SD), "res_std": res_std.detach().cpu(),
        # ---- V14 ckpt contract additions ----
        "v14": True, "diff_mode": a.diff_mode, "cond_raydir": True, "rayc_ch": int(RAYC_CH),
        "sigma_max": float(a.sigma_max), "sigma_min": float(a.sigma_min),
        "p_mean": float(a.p_mean), "p_std": float(a.p_std), "ema_decay": float(a.ema_decay),
        "ema_saved": bool(ema is not None),
    }, tmp)
    os.replace(tmp, path)
def masked_tex_mse(pred, target, mask):                                                     # 70ch (z64+color), masked
    return (((pred - target) ** 2) * mask).sum() / (mask.sum() * XD + 1e-6)

# differentiable decode for length/height/elevation + V11.1 param-matching losses. Must use the same codec that baked --latent_name.
# V14: ALWAYS load the codec -- the composite-score held-out eval (Part F) SAMPLES + decodes every eval, regardless of the reg losses.
_NEED_DECODE = True
if _NEED_DECODE:
    _lck = torch.load(a.codec_path, weights_only=False)
    _lcn = {k: torch.tensor(v, device=dev, dtype=torch.float32) for k, v in _lck["fur_norm"].items()}
    _lcodec = StrandCodec(do_vae=False, decode_type="dir", scale_init=30.0, nr_verts_per_strand=256, nr_values_to_decode=255, dim_per_value_decoded=3).cuda().eval()
    _lcodec.load_state_dict(_lck["state_dict"])
    for p in _lcodec.parameters(): p.requires_grad = False
def decode_lenheight(z_std, idxb, idxhw):                                                    # z_std (B,64,H,W) standardized; sample texels -> arc-length + max-height + rise/arc
    z = (z_std[:, :64] * xs[:, :64] + xm[:, :64]).permute(0, 2, 3, 1).reshape(len(z_std), H * H, 64)[idxb, idxhw]   # (M,64) z64 slice of the 70ch
    pr = _lcodec.decoder(z, None, _lcn); rec = pr["strand_positions"] if "strand_positions" in pr else pr[[k for k in pr if "pos" in k.lower()][0]]
    rec = rec - rec[:, :1]; seg = (rec[:, 1:] - rec[:, :-1]).norm(dim=2); arc = seg.sum(1)    # (M,)
    h = rec[..., 2].clamp_min(0).max(1).values
    rise_arc = rec[:, -1, 2].clamp_min(0) / arc.clamp_min(1e-6)
    return arc, h, rise_arc

def decode_params(z_std, idxb, idxhw, lk):                                                    # V11.1: per-patch deployment params, differentiable (matches extract_fur_character)
    z = (z_std[:, :64] * xs[:, :64] + xm[:, :64]).permute(0, 2, 3, 1).reshape(len(z_std), H * H, 64)[idxb, idxhw]
    pr = _lcodec.decoder(z, None, _lcn); rec = pr["strand_positions"] if "strand_positions" in pr else pr[[k for k in pr if "pos" in k.lower()][0]]
    rec = rec - rec[:, :1]                                                                     # (M,P,3) root-centered
    seg = (rec[:, 1:] - rec[:, :-1]).norm(dim=2); arc = seg.sum(1).clamp_min(1e-6)
    tip = rec[:, -1, :2]; chord2 = tip.norm(dim=1); chord3 = rec[:, -1].norm(dim=1); straight = chord3 / arc
    lift_arc = rec[..., 2].abs().max(1).values / arc
    ang = torch.atan2(tip[:, 1], tip[:, 0])                                                    # per-strand tip direction (pre-canon)
    yc = torch.sin(-ang)[:, None] * rec[..., 0] + torch.cos(-ang)[:, None] * rec[..., 1]       # canonical lateral (rotate tip->+x)
    curl = yc.abs().mean(1) / arc
    nv = len(idxb) // lk; R = lambda x: x.reshape(nv, lk)                                       # patch-grouped (lk texels/patch)
    cm = R(torch.cos(ang)).mean(1); sm = R(torch.sin(ang)).mean(1)
    med = lambda x: R(x).median(1).values
    return dict(
        arc=med(arc),
        straight=med(straight),
        lift_arc=med(lift_arc),
        curl=med(curl),
        tip_spread=R(chord2).std(1, unbiased=False) / med(arc).clamp_min(1e-6),
        clump=(cm ** 2 + sm ** 2).clamp_min(1e-9).sqrt(),
    )

# clump-structure-bearing z64 channels (from v5_lowpass_diag test 1; ch37 strongest). Up-weight them so the
# denoiser spends capacity getting the束心/收口 channels right -> better-structured high-freq at sampling.
_CLUMP_CH = [37, 29, 38, 52, 48, 7, 54, 16, 13, 28, 24, 12]
CW = torch.ones(1, XD, 1, 1, device=dev)                                                     # 70ch: z64(64) + color(6); color weight 1
if a.clump_ch_w != 1.0:
    for _c in _CLUMP_CH: CW[0, _c, 0, 0] = a.clump_ch_w
    CW[0, 37, 0, 0] = a.clump_ch_w * 1.4                                                     # ch37 dominant
# ===== V16 ADD 1: clump-sensitive z64 CHANNEL WEIGHTING (CW), behind --clump_cw (0=off, default 0). =====
# EXTENDS the per-channel CW tensor above with the diag_clump_channels.json 'suggested' tiers. This is a
# WEIGHT on ||z_pred[c]-z_gt[c]||^2 in the PURE EDM-weighted latent MSE (CW multiplies the per-channel
# squared error at the diffusion loss line, CW[:, :64]). It is NOT a decoded-param loss on the diffused
# output D (the V11/V12 teacher-forced trap is forbidden). tier-2.0 -> clump_cw; tier-3.0 -> clump_cw *
# clump_cw_t3_mul; capped at 4. Channels read from JSON 'suggested'.'tier_channel_sets'.
CW_TIER2 = [7, 16, 24, 29, 37, 38, 52, 54]                                                   # fallback if JSON missing
CW_TIER3 = [24, 29, 38, 54]
if a.clump_cw > 0:
    try:
        _cwj = json.load(open(a.clump_cw_json)); _sets = _cwj["suggested"]["tier_channel_sets"]
        CW_TIER2 = [int(c) for c in _sets["2.0"]]; CW_TIER3 = [int(c) for c in _sets["3.0"]]
        print(f"V16 ADD1 clump_cw={a.clump_cw}: loaded tiers from {a.clump_cw_json} | tier2={CW_TIER2} tier3={CW_TIER3}", flush=True)
    except (OSError, KeyError, ValueError) as _e:
        print(f"V16 ADD1 clump_cw={a.clump_cw}: JSON read failed ({_e}); using built-in tier2={CW_TIER2} tier3={CW_TIER3}", flush=True)
    _CW_CAP = 4.0
    for _c in CW_TIER2: CW[0, _c, 0, 0] = min(a.clump_cw, _CW_CAP)                            # tier-2.0 set at clump_cw
    for _c in CW_TIER3: CW[0, _c, 0, 0] = min(a.clump_cw * a.clump_cw_t3_mul, _CW_CAP)        # tier-3.0 higher sub-weight, capped at 4
    _wt = {int(c): float(CW[0, c, 0, 0]) for c in sorted(set(CW_TIER2) | set(CW_TIER3))}
    print(f"V16 ADD1 CW (z64 latent-MSE weights, !=1 only): {_wt} | max {float(CW[0,:64].max()):.2f} (cap 4)", flush=True)
def masked_weighted_tex_mse(pred, target, mask, weight):                                    # 70ch (z64+color), clump-channel weighted
    err = (pred - target) ** 2; wb = weight.reshape(-1, 1, 1, 1)
    return (wb * err * CW * mask).sum() / (mask.sum() * XD + 1e-6)

def _grad2(x):
    gx = x[..., :, 1:] - x[..., :, :-1]; gy = x[..., 1:, :] - x[..., :-1, :]
    return F.pad(gx, (0, 1)), F.pad(gy, (0, 0, 1, 0))
def hf_resid_loss(pred, target, mask, weight):                                              # gradient-RECONSTRUCTION on z64 (align ∇D to ∇x0 at low σ)
    pgx, pgy = _grad2(pred); tgx, tgy = _grad2(target); e = (pgx - tgx) ** 2 + (pgy - tgy) ** 2; wb = weight.reshape(-1, 1, 1, 1)
    nc = pred.shape[1]
    return (wb * e * CW[:, :nc] * mask).sum() / (mask.sum() * nc + 1e-6)

# ===== V13 §6: decoded-character/param losses on the ZERO-INPUT REG BASE ONLY (no teacher-forcing) =====
def _sample_texels(m, B):                                                                    # pick len_k masked texels per patch for the differentiable decode
    mf = (m[:, 0] > 0.5).reshape(B, H * H); ib = []; ihw = []
    for bb in range(B):
        nz = mf[bb].nonzero(as_tuple=True)[0]
        if len(nz) == 0: continue
        ib.append(torch.full((a.len_k,), bb, device=dev)); ihw.append(nz[torch.randint(len(nz), (a.len_k,), device=dev)])
    if not ib: return None, None
    return torch.cat(ib), torch.cat(ihw)

# ===== V16 ADD 2: clump-LEVEL BUCKET sampler, behind --clump_bucket (0/1, default 0). =====
# At startup decode GT z64 -> per-patch clump (REUSE decode_params) + a per-patch COHERENCE proxy
# (per-texel tip-azimuth tangled-rate). Bucket the TRAIN patches into 4 clump QUARTILES (low/mid/high/
# extreme) and build sampling weights that are BALANCED across buckets (each bucket gets equal total
# mass), REPLACING E1's hf_oversample weights w_ti. The coherence proxy DOWN-WEIGHTS tangled patches
# inside their bucket so we do NOT oversample mess (avoid teaching 'more mess = more clump'). Buckets by
# clump LEVEL, NOT coat type (coat-balanced = the V15 failure). Existing-data bucketing only (no new data).
@torch.no_grad()
def _gt_clump_and_tangle(jj):
    """Decode GT z64 for patches jj at len_k masked texels -> (clump scalar, tangled-rate) per kept patch.
    clump = decode_params['clump']; tangled-rate = frac of per-texel tip azimuths deviating > thr from the
    patch circular-mean direction (the diag_weak_features coherence proxy)."""
    mm = fetch(MASK, jj); ib, ihw = _sample_texels(mm, len(jj))
    if ib is None: return None, None, None
    gz = fetch(Xn, jj)
    pg = decode_params(gz, ib, ihw, a.len_k)                                                  # per-patch medians incl clump
    # per-texel tip azimuths for the tangled-rate: re-decode the SAME sampled texels (mirror decode_params)
    z = (gz[:, :64] * xs[:, :64] + xm[:, :64]).permute(0, 2, 3, 1).reshape(len(gz), H * H, 64)[ib, ihw]
    pr = _lcodec.decoder(z, None, _lcn); rec = pr["strand_positions"] if "strand_positions" in pr else pr[[k for k in pr if "pos" in k.lower()][0]]
    rec = rec - rec[:, :1]; tip = rec[:, -1, :2]; ang = torch.atan2(tip[:, 1], tip[:, 0])
    nv = len(ib) // a.len_k; A = ang.reshape(nv, a.len_k)
    cbar = torch.cos(A).mean(1); sbar = torch.sin(A).mean(1); mean_ang = torch.atan2(sbar, cbar)
    dev_ang = torch.atan2(torch.sin(A - mean_ang[:, None]), torch.cos(A - mean_ang[:, None])).abs()
    thr = np.deg2rad(a.clump_bucket_tangle_deg)
    tangled = (dev_ang > thr).float().mean(1)                                                 # (nv,) per-patch tangled-rate
    kept = (mm[:, 0] > 0.5).reshape(len(jj), -1).any(1)
    return pg["clump"].detach(), tangled.detach(), kept

w_clump_bucket = None                                                                         # set below when --clump_bucket=1; else stays None (E1 w_ti used)
if a.clump_bucket:
    print("V16 ADD2 clump_bucket=1: decoding GT clump + coherence proxy over TRAIN patches for quartile bucketing ...", flush=True)
    _cl = torch.full((len(ti),), float("nan"), device=dev); _tg = torch.full((len(ti),), float("nan"), device=dev)
    for s in range(0, len(ti), a.eval_batch):
        jj = ti[s:s + a.eval_batch]
        cl, tg, kept = _gt_clump_and_tangle(jj)
        if cl is None: continue
        pos = torch.arange(s, s + len(jj), device=dev)[kept]                                  # positions into ti for KEPT patches
        _cl[pos] = cl.to(dev); _tg[pos] = tg.to(dev)
    valid = torch.isfinite(_cl)
    cl_ok = _cl[valid]
    # 4 clump QUARTILES (low/mid/high/extreme) by GT clump level
    qs = torch.quantile(cl_ok, torch.tensor([0.25, 0.5, 0.75], device=dev))
    bucket = torch.bucketize(_cl, qs)                                                         # 0..3; NaN -> 3 but masked out below
    bucket[~valid] = -1
    # coherence proxy -> per-patch keep weight: down-weight TANGLED patches (don't oversample mess).
    # weight in [0.25,1] linear in (1 - tangled_rate); clamp so even messy patches keep some mass.
    coh_w = (1.0 - _tg.nan_to_num(1.0)).clamp(0.0, 1.0) * 0.75 + 0.25
    coh_w[~valid] = 0.0
    w_clump_bucket = torch.zeros(len(ti), device=idx_dev)
    counts = []
    for b in range(4):
        sel = (bucket == b)
        n_b = int(sel.sum()); counts.append(n_b)
        if n_b == 0: continue
        wb = coh_w[sel]
        wb = wb / (wb.sum() + 1e-9)                                                           # within-bucket coherence-weighted, sums to 1
        w_clump_bucket[sel.cpu() if idx_dev.type == "cpu" else sel] = (wb / 1.0).to(idx_dev)  # each bucket -> equal total mass (BALANCED)
    w_clump_bucket = w_clump_bucket / (w_clump_bucket.sum() + 1e-9)
    print(f"V16 ADD2 clump quartiles q=[{qs[0]:.3f},{qs[1]:.3f},{qs[2]:.3f}] | bucket counts (low..extreme)={counts} | "
          f"valid {int(valid.sum())}/{len(ti)} | mean tangled {float(_tg.nan_to_num(0).mean()):.3f} "
          f"(tangled patches down-weighted, NOT oversampled)", flush=True)
    w_ti = w_clump_bucket                                                                     # REPLACE E1's hf_oversample sampling weights with balanced-quartile weights

def reg_decoded_loss(mu, target, m):                                                        # mu = reg base output (grad on), target = GT std latent (no grad)
    ib, ihw = _sample_texels(m, len(mu))
    items = {}; total = 0.0
    if ib is None: return torch.zeros((), device=dev), items
    if a.reg_len_w > 0:
        pa, pha, _ = decode_lenheight(mu, ib, ihw)
        with torch.no_grad(): ga, gha, _ = decode_lenheight(target, ib, ihw)
        llen = (pa - ga).abs().mean() + (pha - gha).abs().mean()
        total = total + a.reg_len_w * llen; items["len"] = float(llen.detach())
    if a.reg_param_w > 0:
        pp_ = decode_params(mu, ib, ihw, a.len_k)
        with torch.no_grad(): gp_ = decode_params(target, ib, ihw, a.len_k)
        lparts = {k: (pp_[k] - gp_[k]).abs().mean() / gp_[k].abs().mean().clamp_min(1e-3) for k in pp_}
        wsum = sum(REG_PARAM_WTS[k] for k in lparts)
        lpar = sum(float(REG_PARAM_WTS[k]) * lparts[k] for k in lparts) / max(float(wsum), 1e-6)
        total = total + a.reg_param_w * lpar
        for k, v in lparts.items(): items[k] = float(v.detach())
    return total, items

# ---- V13 §6: held-out 7-param decode eval (the SAVE-BEST / EARLY-STOP signal). Forward-declared; full impl below. ----
EVAL_PARAM_KEYS = ["arc", "straight", "curl", "clump", "lift_arc", "tip_spread"]            # density/radius reported separately in the final decode eval
def is_short(name):                                                                          # short vs long split by coat name (build_full_dataset coats)
    n = name.lower()
    return any(t in n for t in ("ultra_short", "short", "buzz", "stubble", "velvet"))
def reg_param_eval():
    """Held-out: decode reg base, report 7-param relerr (nMAE) split short/long. Returns headline scalar (lower=better)."""
    reg.eval(); tot = {k: [] for k in EVAL_PARAM_KEYS}; tot_s = {k: [] for k in EVAL_PARAM_KEYS}; tot_l = {k: [] for k in EVAL_PARAM_KEYS}
    with torch.no_grad():
        for s in range(0, len(vi), a.eval_batch):
            jj = vi[s:s + a.eval_batch]
            mu = reg(torch.zeros(len(jj), XD, H, H, device=dev), zc(len(jj)), fetch(G_COND, jj), fetch(GLOB, jj), fetch(RHOC, jj), fetch(RGBPROJ, jj), rayc=fetch(RAYC, jj))
            mm = fetch(MASK, jj)
            ib, ihw = _sample_texels(mm, len(jj))
            if ib is None: continue
            pp_ = decode_params(mu, ib, ihw, a.len_k); gp_ = decode_params(fetch(Xn, jj), ib, ihw, a.len_k)
            # _sample_texels skips empty-mask patches -> decode_params returns one row per KEPT patch, in batch order.
            # Build shortmask over the SAME kept rows so it aligns with pp_/gp_ (else size mismatch on an empty mask).
            kept = (mm[:, 0] > 0.5).reshape(len(jj), -1).any(1)
            jj_kept = jj[kept]
            shortmask = torch.tensor([is_short(os.path.basename(dirs[int(j)])) for j in jj_kept.tolist()], device=dev)
            for k in EVAL_PARAM_KEYS:
                rel = (pp_[k] - gp_[k]).abs() / gp_[k].abs().clamp_min(1e-3)
                tot[k].append(rel); tot_s[k].append(rel[shortmask]); tot_l[k].append(rel[~shortmask])
    def agg(d): return {k: (torch.cat(d[k]).mean().item() if d[k] and len(torch.cat(d[k])) else float('nan')) for k in EVAL_PARAM_KEYS}
    A, Sm, Lm = agg(tot), agg(tot_s), agg(tot_l)
    headline = float(np.nanmean([A[k] for k in EVAL_PARAM_KEYS]))
    reg.train()
    return headline, A, Sm, Lm

# ===== V13-2: re-estimate EDM SD + per-channel res_std on the residual r = x0 - z_base.detach() (§5/point2) =====
res_std_ready = False
def estimate_res_std(nbatch=24):
    """After reg warm-up (or after loading a converged reg), measure the residual r = x0 - z_base.detach()
    over the TRAIN split (masked texels) -> per-channel res_std (XD-vector) + global SD = std of the
    res_std-standardized residual (~1). Residual std << x0 std, so this fixes the EDM weighting (else SD=1
    treats the tiny residual as if it had x0's scale -> useless precond)."""
    global SD, res_std, res_std_ready
    reg.eval(); acc = []
    with torch.no_grad():
        for _ in range(nbatch):
            bj = ti[torch.randint(0, len(ti), (a.bs,), device=idx_dev)]
            x0b = fetch(Xn, bj); mb = fetch(MASK, bj)
            mub = reg(torch.zeros_like(x0b), zc(len(bj)), fetch(G_COND, bj), fetch(GLOB, bj), fetch(RHOC, bj), fetch(RGBPROJ, bj), rayc=fetch(RAYC, bj))
            r = (x0b - mub)                                                                   # (B,XD,H,W)
            mm = (mb[:, 0] > 0.5)                                                             # (B,H,W) geometry mask
            acc.append(r.permute(1, 0, 2, 3)[:, mm])                                          # (XD, Mpix)
    rf = torch.cat(acc, 1).float()                                                            # (XD, totalMpix)
    rs = rf.std(1, unbiased=False).clamp_min(1e-4)                                            # per-channel residual std
    res_std = rs.reshape(1, XD, 1, 1).to(dev)
    r_std_norm = rf / rs[:, None]                                                             # standardize -> SD = global std of the unit-ish residual (~1)
    SD = float(r_std_norm.std(unbiased=False).clamp_min(1e-4).item())
    res_std_ready = True
    reg.train()
    print(f"  [res_std] estimated over {rf.shape[1]} masked texels | global SD {SD:.4f} | "
          f"res_std mean {res_std.mean().item():.4f} (z64 {res_std[0,:64].mean().item():.4f}) | "
          f"x0 std was 1.0 -> residual {res_std.mean().item():.3f}x smaller", flush=True)

# ===== V14 unified Heun sampler (§A: ramp at a.sigma_max; §B: CFG drops ALL groups together; §D: uses EMA via eval_net()).
#   FULL mode  : zbase=None, rayc the explicit geom condition; denoise standardized x0 directly -> return standardized z.
#   RESIDUAL   : zbase=z_base (computed once outside the loop); denoise standardized residual -> return z_base + x*res_std.
#   True ALL-condition CFG: the null branch zeros gc, gl, rhoc, rgbp AND rayc (matches the trained all-group drop). =====
@torch.no_grad()
def sample_z(gc, gl, rhoc, rgbp, rayc, zbase=None, steps=None):
    steps = a.steps if steps is None else steps
    snet = eval_net()                                                                        # EMA weights when EMA is on
    sigs = torch.linspace(0, 1, steps + 1, device=dev)
    sigs = (a.sigma_max**(1 / 7) + sigs * (a.sigma_min**(1 / 7) - a.sigma_max**(1 / 7)))**7  # Karras ramp at a.sigma_max
    sigs = torch.cat([sigs, torch.zeros(1, device=dev)])
    x = torch.randn(len(gc), XD, H, H, device=dev) * sigs[0] * a.sample_temp                 # noised standardized [x0 | residual]
    zb = None if zbase is None else zbase.detach()
    null = lambda t: (None if t is None else torch.zeros_like(t))
    def denoise(xx, ss):
        dc = precond(snet, xx, ss.expand(len(gc)), gc, gl, rhoc, rgbp, zb, rayc)
        if a.cfg != 1.0:                                                                     # ALL-condition CFG: zero EVERY condition group for the uncond branch
            du = precond(snet, xx, ss.expand(len(gc)), null(gc), null(gl), null(rhoc), null(rgbp), zb, null(rayc))
            dc = du + a.cfg * (dc - du)
        return dc
    for j in range(steps):
        s0, s1 = sigs[j], sigs[j + 1]; d = (x - denoise(x, s0)) / s0; xn = x + (s1 - s0) * d
        if s1 > 0: d2 = (xn - denoise(xn, s1)) / s1; xn = x + (s1 - s0) * 0.5 * (d + d2)
        x = xn
    return x if zb is None else (zb + x * res_std)                                           # FULL: standardized x0 | RESIDUAL: z_base + residual*res_std

# Back-compat alias for the V13-2 residual call sites (B-vs-F eval, final decode eval).
@torch.no_grad()
def sample_residual(gc, gl, rhoc, rgbp, zbase, steps=None, rayc=None):
    return sample_z(gc, gl, rhoc, rgbp, rayc, zbase=zbase, steps=steps)

# ===== V13-2 §7: B-vs-F decomposition. B = 7-param nMAE of decode(z_base); F = decode(z_base + z_res). =====
def reg_BvsF_eval():
    """At each eval, report BOTH B (base only) and F (base+residual sampled), per-param, split short/long.
    F<B on clump/curl => residual contributes; F~=B => residual is dead weight (ship base-only, §7)."""
    net.eval(); reg.eval()
    def acc_dict(): return {k: {"B": [], "F": [], "Bs": [], "Fs": [], "Bl": [], "Fl": []} for k in EVAL_PARAM_KEYS}
    tot = acc_dict()
    with torch.no_grad():
        for s in range(0, len(vi), a.eval_batch):
            jj = vi[s:s + a.eval_batch]
            gcb = fetch(G_COND, jj); glb = fetch(GLOB, jj); rcb = fetch(RHOC, jj); rpb = fetch(RGBPROJ, jj); ryb = fetch(RAYC, jj)
            zb = reg(torch.zeros(len(jj), XD, H, H, device=dev), zc(len(jj)), gcb, glb, rcb, rpb, rayc=ryb)   # z_base
            zr = sample_residual(gcb, glb, rcb, rpb, zb, rayc=ryb)                             # z_base + denoised residual*res_std
            mm = fetch(MASK, jj)
            ib, ihw = _sample_texels(mm, len(jj))
            if ib is None: continue
            pB = decode_params(zb, ib, ihw, a.len_k); pF = decode_params(zr, ib, ihw, a.len_k)
            gp_ = decode_params(fetch(Xn, jj), ib, ihw, a.len_k)
            kept = (mm[:, 0] > 0.5).reshape(len(jj), -1).any(1); jj_kept = jj[kept]
            shortmask = torch.tensor([is_short(os.path.basename(dirs[int(j)])) for j in jj_kept.tolist()], device=dev)
            for k in EVAL_PARAM_KEYS:
                rb = (pB[k] - gp_[k]).abs() / gp_[k].abs().clamp_min(1e-3)
                rf = (pF[k] - gp_[k]).abs() / gp_[k].abs().clamp_min(1e-3)
                tot[k]["B"].append(rb); tot[k]["F"].append(rf)
                tot[k]["Bs"].append(rb[shortmask]); tot[k]["Fs"].append(rf[shortmask])
                tot[k]["Bl"].append(rb[~shortmask]); tot[k]["Fl"].append(rf[~shortmask])
    def m(lst): c = torch.cat(lst) if lst else None; return (c.mean().item() if (c is not None and len(c)) else float('nan'))
    out = {k: {kk: m(tot[k][kk]) for kk in tot[k]} for k in EVAL_PARAM_KEYS}
    B = float(np.nanmean([out[k]["B"] for k in EVAL_PARAM_KEYS])); F = float(np.nanmean([out[k]["F"] for k in EVAL_PARAM_KEYS]))
    net.train(); reg.train()
    return B, F, out

# ===== V14 Part F: held-out SAMPLED z64 helper (full vs residual), composite score, cond-usage delta. =====
def _pcorr_t(u, v):                                                                          # Pearson corr of two 1-D tensors (mean-agnostic; 0 if degenerate)
    u = u.float(); v = v.float()
    if u.numel() < 3 or u.std(unbiased=False) < 1e-6 or v.std(unbiased=False) < 1e-6: return 0.0
    um, vm = u - u.mean(), v - v.mean()
    return float((um * vm).mean() / (um.std(unbiased=False) * vm.std(unbiased=False) + 1e-12))

@torch.no_grad()
def _sample_z_for(jj):
    """Sample the standardized z64(+color) for held-out patches jj via the V14 Heun sampler (EMA when on).
    FULL mode -> diffuse z64 directly (zbase=None). RESIDUAL -> reg base + residual."""
    gcb = fetch(G_COND, jj); glb = fetch(GLOB, jj); rcb = fetch(RHOC, jj); rpb = fetch(RGBPROJ, jj); ryb = fetch(RAYC, jj)
    if FULL:
        return sample_z(gcb, glb, rcb, rpb, ryb, zbase=None)
    zb = reg(torch.zeros(len(jj), XD, H, H, device=dev), zc(len(jj)), gcb, glb, rcb, rpb, rayc=ryb)
    return sample_z(gcb, glb, rcb, rpb, ryb, zbase=zb)

def composite_score_eval():
    """V14 Part F headline: SAMPLE (Heun @ a.sigma_max) ~score_eval_n held-out patches, decode via the codec,
    and compute per-param {length_corr, curl_corr, straight_corr, clump_std_ratio} vs GT. Returns (score, components).
    score = 0.35*length_corr + 0.20*curl_corr + 0.20*straight_corr + 0.15*(1-|1-std(pred_clump)/std(GT_clump)|) + 0.10*0(visual)."""
    net.eval(); reg.eval()
    sub = vi[:min(int(a.score_eval_n), len(vi))]
    pL = []; pC = []; pS = []; pK = []; gL = []; gC = []; gS = []; gK = []                    # per-patch arc/curl/straight/clump for pred & GT
    pLi = []; gLi = []                                                                        # V16 ADD3: lift_arc (strong attr) pred & GT
    pPar = []; gPar = []; pTan = []; gTan = []                                                # V16 ADD3: per-patch parallelism + tangled-rate (coherence proxy)
    _thr = np.deg2rad(a.clump_bucket_tangle_deg)
    def _coh(z_std, ib, ihw):                                                                 # per-patch parallelism (resultant len) + tangled-rate from per-texel tip azimuths
        zt = (z_std[:, :64] * xs[:, :64] + xm[:, :64]).permute(0, 2, 3, 1).reshape(len(z_std), H * H, 64)[ib, ihw]
        pr = _lcodec.decoder(zt, None, _lcn); rec = pr["strand_positions"] if "strand_positions" in pr else pr[[k for k in pr if "pos" in k.lower()][0]]
        rec = rec - rec[:, :1]; tip = rec[:, -1, :2]; ang = torch.atan2(tip[:, 1], tip[:, 0])
        nv = len(ib) // a.len_k; Aa = ang.reshape(nv, a.len_k)
        cbar = torch.cos(Aa).mean(1); sbar = torch.sin(Aa).mean(1)
        par = torch.hypot(cbar, sbar)                                                         # resultant length = parallelism in [0,1]
        mean_ang = torch.atan2(sbar, cbar)
        dev_ang = torch.atan2(torch.sin(Aa - mean_ang[:, None]), torch.cos(Aa - mean_ang[:, None])).abs()
        tan = (dev_ang > _thr).float().mean(1)
        return par, tan
    for s in range(0, len(sub), a.eval_batch):
        jj = sub[s:s + a.eval_batch]
        z = _sample_z_for(jj)                                                                # standardized z64(+color)
        mm = fetch(MASK, jj)
        ib, ihw = _sample_texels(mm, len(jj))
        if ib is None: continue
        gz = fetch(Xn, jj)
        pp_ = decode_params(z, ib, ihw, a.len_k); gp_ = decode_params(gz, ib, ihw, a.len_k)
        pL.append(pp_["arc"]); pC.append(pp_["curl"]); pS.append(pp_["straight"]); pK.append(pp_["clump"]); pLi.append(pp_["lift_arc"])
        gL.append(gp_["arc"]); gC.append(gp_["curl"]); gS.append(gp_["straight"]); gK.append(gp_["clump"]); gLi.append(gp_["lift_arc"])
        ppar, ptan = _coh(z, ib, ihw); gpar, gtan = _coh(gz, ib, ihw)                         # V16 ADD3: coherence on SAMPLED output vs GT
        pPar.append(ppar); pTan.append(ptan); gPar.append(gpar); gTan.append(gtan)
    net.train(); reg.train()
    if not pL:
        return float("nan"), dict(length_corr=float("nan"), curl_corr=float("nan"), straight_corr=float("nan"), clump_stdratio=float("nan"),
                                  clump_corr=float("nan"), lift_corr=float("nan"),
                                  coh_par_pred=float("nan"), coh_par_gt=float("nan"), tangled_pred=float("nan"), tangled_gt=float("nan"))
    pL = torch.cat(pL); pC = torch.cat(pC); pS = torch.cat(pS); pK = torch.cat(pK)
    gL = torch.cat(gL); gC = torch.cat(gC); gS = torch.cat(gS); gK = torch.cat(gK)
    pLi = torch.cat(pLi); gLi = torch.cat(gLi)
    pPar = torch.cat(pPar); gPar = torch.cat(gPar); pTan = torch.cat(pTan); gTan = torch.cat(gTan)
    length_corr = _pcorr_t(pL, gL); curl_corr = _pcorr_t(pC, gC); straight_corr = _pcorr_t(pS, gS)
    gk_std = float(gK.std(unbiased=False)); pk_std = float(pK.std(unbiased=False))
    stdratio = pk_std / (gk_std + 1e-9)
    clump_closeness = 1.0 - abs(1.0 - stdratio)                                              # 1 at ratio==1, falls off either side
    # E1 score formula UNCHANGED (V16 metrics are diagnostic-only; they do NOT enter the save-best/early-stop signal).
    score = 0.35 * length_corr + 0.20 * curl_corr + 0.20 * straight_corr + 0.15 * clump_closeness + 0.10 * 0.0
    comps = dict(length_corr=length_corr, curl_corr=curl_corr, straight_corr=straight_corr,
                 clump_stdratio=stdratio, clump_closeness=clump_closeness,
                 # ---- V16 ADD3 clump-coherence + strong-attr diagnostics (SAMPLED output vs GT) ----
                 clump_corr=_pcorr_t(pK, gK), lift_corr=_pcorr_t(pLi, gLi),
                 coh_par_pred=float(pPar.mean()), coh_par_gt=float(gPar.mean()),
                 tangled_pred=float(pTan.mean()), tangled_gt=float(gTan.mean()))
    return float(score), comps

@torch.no_grad()
def cond_usage_delta(nb=None):
    """V14 Part F diagnostic: mean ||pred_cond - pred_uncond|| (cond vs the trained all-zero null) on a small held-out
    batch, at a fixed mid sigma. Large delta = the model USES the condition; ~0 = it ignores it (injection too weak)."""
    snet = eval_net(); snet_was = snet.training; snet.eval()
    nb = a.eval_batch if nb is None else nb
    jj = vi[:min(nb, len(vi))]
    gcb = fetch(G_COND, jj); glb = fetch(GLOB, jj); rcb = fetch(RHOC, jj); rpb = fetch(RGBPROJ, jj); ryb = fetch(RAYC, jj)
    zb = None
    if not FULL:
        zb = reg(torch.zeros(len(jj), XD, H, H, device=dev), zc(len(jj)), gcb, glb, rcb, rpb, rayc=ryb)
    sig = torch.full((len(jj),), float(np.sqrt(max(a.sigma_min, 1e-3) * a.sigma_max)), device=dev)   # geometric-mid sigma
    x = torch.randn(len(jj), XD, H, H, device=dev) * sig.reshape(-1, 1, 1, 1)
    pc = precond(snet, x, sig, gcb, glb, rcb, rpb, zb, ryb)
    z = lambda t: (None if t is None else torch.zeros_like(t))
    pu = precond(snet, x, sig, z(gcb), z(glb), z(rcb), z(rpb), zb, z(ryb))                    # ALL-condition null
    if snet_was: snet.train()
    return float((pc - pu).pow(2).sum(1).sqrt().mean())                                       # mean L2 over channels, averaged over texels/batch

best_eval = 1e9; best_it = 0; no_improve = 0; monitoring_F = False
best_score = -1e9; best_score_it = 0; no_improve_s = 0                                        # V14: save-best/early-stop on the COMPOSITE SCORE (higher=better)
def save_reg_best(step):
    save_stage3(f"{a.out}/v13_reg_best.pt", step)
def save_v14_best(step):
    save_stage3(f"{a.out}/v14_best.pt", step)

# ===== V14 Part B: condition dropping. =====
#  - --cond_drop p (CFG): per-element, with prob p, zero ALL condition groups TOGETHER (trains the true uncond branch).
#  - --drop_group g (ablation): zero ONLY one named group for the WHOLE batch every step.
# Groups: appearance={RGBPROJ, G_COND, GLOB}, geometry={raydir/grazing/vis = RAYC}, density={rho=RHOC}, global={GLOB}.
def apply_cond_drops(gc, gl, rc, rp, ry):
    gc, gl, rc, rp, ry = gc.clone(), gl.clone(), rc.clone(), rp.clone(), ry.clone()
    # (1) CFG all-group random drop (per batch element)
    if a.cond_drop > 0:
        drop = torch.rand(len(gc), device=dev) < a.cond_drop
        if drop.any():
            gc[drop] = 0; gl[drop] = 0; rc[drop] = 0; rp[drop] = 0; ry[drop] = 0             # ALL groups together (fixes V13's partial drop)
    # (2) named-group ablation (whole batch, every step)
    g = a.drop_group
    if g == "appearance":   rp[:] = 0; gc[:] = 0; gl[:] = 0
    elif g == "geometry":   ry[:] = 0
    elif g == "density":    rc[:] = 0
    elif g == "global":     gl[:] = 0
    return gc, gl, rc, rp, ry

for it in range(start_it + 1, a.iters + 1):
    cur_lr = set_cos_lr(it - 1)
    bi = ti[torch.multinomial(w_ti, a.bs, replacement=True)]
    x0 = fetch(Xn, bi); gc = fetch(G_COND, bi); gl = fetch(GLOB, bi); m = fetch(MASK, bi); rc = fetch(RHOC, bi); rp = fetch(RGBPROJ, bi)
    mc = fetch(CMASK, bi); ry = fetch(RAYC, bi)
    # ===== REG BASE (V13 §6): deterministic zero-input regressor. Decoded-character/length losses live HERE ONLY (no teacher-forcing). =====
    # V14 FULL mode bypasses the reg base entirely (no reg loss/step). RESIDUAL mode keeps the V13 reg machinery.
    reg_dec_v = {}; reg_loss = torch.zeros((), device=dev); lr = reg_loss
    if not FULL:
        if resumed_reg_only:                                                                 # converged reg resumed for residual training -> FREEZE it
            with torch.no_grad():
                mu = reg(torch.zeros_like(x0), zc(len(bi)), gc, gl, rc, rp, rayc=ry)
        else:
            mu = reg(torch.zeros_like(x0), zc(len(bi)), gc, gl, rc, rp, rayc=ry)
            lr = masked_tex_mse(mu, x0, m)
            lrcol = 0.0 if a.no_color else (((mu[:, 64:] - x0[:, 64:]) ** 2) * mc).sum() / (mc.sum() * 6 + 1e-6)
            reg_loss = lr + a.color_w * lrcol
            if a.reg_param_w > 0 or a.reg_len_w > 0:
                rdec, reg_dec_v = reg_decoded_loss(mu, x0, m)
                reg_loss = reg_loss + rdec
            optr.zero_grad(); reg_loss.backward(); optr.step()

    # ===== DIFFUSION (V14 Part E). FULL: diffuse z64 DIRECTLY (target=standardized x0, sigma_data=1, NO z_base). =====
    #       RESIDUAL: V13-2 path (reg base + (x0 - z_base.detach())/res_std). HARD RULE §4/§6: pure EDM-weighted MSE on D.
    loss_v = 0.0; z64_v = 0.0; colr_v = colt_v = 0.0
    diff_live = FULL or (not a.no_diffusion and it > REG_ITERS)
    if diff_live:
        if FULL:
            zb = None                                                                        # NO reg base, NO z_base concat (spec Part E)
            target = x0                                                                       # diffuse the standardized x0 DIRECTLY (sigma_data=1)
        else:
            if not res_std_ready:                                                             # §5/risk6: re-estimate SD/res_std ONCE when the residual head turns on
                estimate_res_std()
            zb = mu.detach()                                                                  # the (frozen-for-this-step) deterministic base, conditions the denoiser
            target = (x0 - zb) / res_std                                                      # STANDARDIZED residual (~unit per channel) -> proper EDM scale
        gc_d, gl_d, rc_d, rp_d, ry_d = apply_cond_drops(gc, gl, rc, rp, ry)                   # V14 Part B: ALL-group CFG drop + named-group ablation
        sig = (torch.randn(len(bi), device=dev) * a.p_std + a.p_mean).exp()                   # V14 Part A: log-sigma draw (p_mean/p_std)
        n = torch.randn_like(target) * sig.reshape(-1, 1, 1, 1)
        D = precond(net, target + n, sig, gc_d, gl_d, rc_d, rp_d, zb, ry_d)
        w = (sig**2 + SD**2) / (sig * SD).clamp_min(1e-4)**2; wb = w.reshape(-1, 1, 1, 1)
        # HARD RULE §4/§6: PURE EDM-weighted MSE on D (+ optional hf). NO param_w/len_w on the diffused output.
        lz = (((D[:, :64] - target[:, :64]) ** 2) * CW[:, :64] * wb * m).sum() / (m.sum() * 64 + 1e-6)
        lcol = 0.0 if a.no_color else (((D[:, 64:] - target[:, 64:]) ** 2) * wb * mc).sum() / (mc.sum() * 6 + 1e-6)
        loss = lz + a.color_w * lcol + (a.hf_w * hf_resid_loss(D[:, :64], target[:, :64], m, w) if a.hf_w > 0 else 0.0)
        opt.zero_grad(); loss.backward(); opt.step()
        ema_update()                                                                          # V14 Part D: update EMA shadow after the diffusion step
        loss_v = float(loss.detach())
        with torch.no_grad():
            z64_v = float((((D[:, :64] - target[:, :64]) ** 2) * m).sum() / (m.sum() * 64 + 1e-6))
            if not a.no_color:
                colr_v = float((((D[:, 64:67] - target[:, 64:67]) ** 2) * mc).sum() / (mc.sum() * 3 + 1e-6))
                colt_v = float((((D[:, 67:70] - target[:, 67:70]) ** 2) * mc).sum() / (mc.sum() * 3 + 1e-6))

    if it % 1000 == 0:
        ptxt = (" | reg_dec[" + " ".join(f"{k}:{v:.2f}" for k, v in reg_dec_v.items()) + "]") if reg_dec_v else ""
        difftxt = f" | DIFF loss {loss_v:.3f} z64 {z64_v:.3f} col {colr_v:.4f}/{colt_v:.4f}" if diff_live else " | DIFF off"
        print(f"  it {it} | reg {float(reg_loss.detach()):.3f} (z {float(lr.detach()):.3f}){difftxt} | lr {cur_lr:.2e}{ptxt}", flush=True)

    # ---- V14 Part F: held-out eval. Composite SCORE drives save-best + early-stop (NOT nMAE). cond-usage delta every eval.
    #      RESIDUAL mode also prints the V13 7-param nMAE / B-vs-F diagnostics. ----
    if a.eval_every > 0 and it % a.eval_every == 0:
        delta = cond_usage_delta()
        print(f"  [cond-usage] delta={delta:.4f}  (cond vs all-zero null; ~0 => condition ignored)", flush=True)
        if not FULL:                                                                          # residual diagnostics (reg base 7-param nMAE + B-vs-F)
            headline, A, Sm, Lm = reg_param_eval()
            print(f"  [eval it {it}] 7-param nMAE(B) {headline:.4f} | "
                  + " ".join(f"{k}:{A[k]:.3f}(S{Sm[k]:.3f}/L{Lm[k]:.3f})" for k in EVAL_PARAM_KEYS), flush=True)
            residual_live = (not a.no_diffusion and it > REG_ITERS and res_std_ready)
            if residual_live:
                Bh, Fh, BF = reg_BvsF_eval()
                print(f"  [B-vs-F] B={Bh:.4f} F={Fh:.4f} | "
                      + " ".join(f"{k} B/F {BF[k]['B']:.3f}/{BF[k]['F']:.3f}" for k in ("clump", "straight", "curl")), flush=True)
        # V14 headline: composite score (sampled + decoded). save-best + early-stop on SCORE (higher=better).
        score, comps = composite_score_eval()
        print(f"  [eval] score={score:.4f} len_corr={comps['length_corr']:.3f} curl_corr={comps['curl_corr']:.3f} "
              f"straight_corr={comps['straight_corr']:.3f} clump_stdratio={comps['clump_stdratio']:.3f}", flush=True)
        # V16 ADD3: clump-coherence + strong-attr diagnostics on the SAMPLED output vs GT. SUCCESS = clump_corr/stdratio
        # UP and coherence (parallelism) NOT down and strong attrs (lift/straight/curl) NOT down. FAILURE = clump up but
        # tangled_pred up (mess), or strong attrs drop. These are diagnostics-only; the E1 score/early-stop is unchanged.
        print(f"  [eval-v16] clump_corr={comps['clump_corr']:.3f} clump_stdratio={comps['clump_stdratio']:.3f} | "
              f"lift_corr={comps['lift_corr']:.3f} straight_corr={comps['straight_corr']:.3f} curl_corr={comps['curl_corr']:.3f} | "
              f"parallelism pred/gt={comps['coh_par_pred']:.3f}/{comps['coh_par_gt']:.3f} "
              f"tangled-rate pred/gt={comps['tangled_pred']:.3f}/{comps['tangled_gt']:.3f}", flush=True)
        if not np.isnan(score) and score > best_score + 1e-4:
            best_score = score; best_score_it = it; no_improve_s = 0; save_v14_best(it)
            print(f"  [eval] NEW BEST score {best_score:.4f} @ it {it} -> {a.out}/v14_best.pt", flush=True)
        else:
            no_improve_s += 1
            print(f"  [eval] no improve ({no_improve_s}/{a.early_stop}); best score {best_score:.4f} @ it {best_score_it}", flush=True)
            if a.early_stop > 0 and no_improve_s >= a.early_stop:
                print(f"  [eval] EARLY STOP at it {it} (best score {best_score:.4f} @ it {best_score_it})", flush=True); break

    if a.save_every > 0 and it % a.save_every == 0:
        save_stage3(f"{a.out}/stage3_model_latest.pt", it)
        print(f"  saved latest checkpoint @ it {it} -> {a.out}/stage3_model_latest.pt", flush=True)

save_stage3(f"{a.out}/stage3_model.pt", a.iters)
print(f"V14 done. best composite score {best_score:.4f} @ it {best_score_it} (saved {a.out}/v14_best.pt). final ckpt {a.out}/stage3_model.pt", flush=True)
# V14: load the BEST checkpoint (its `net` is the EMA shadow when EMA was on) for the final decode eval.
if os.path.exists(f"{a.out}/v14_best.pt"):
    _rb = torch.load(f"{a.out}/v14_best.pt", map_location=dev, weights_only=False)
    try: net.load_state_dict(_rb["net"]); print(f"  loaded v14_best.pt net (it {_rb.get('iter')}) for final eval", flush=True)
    except (RuntimeError, KeyError): print("  v14_best.pt net not loaded (shape mismatch) -- using live net", flush=True)
    if not FULL:
        try: reg.load_state_dict(_rb["reg"]); print("  loaded v14_best.pt reg base", flush=True)
        except (RuntimeError, KeyError): pass
# After this point EMA is no longer updated; force eval_net() to return the (best-loaded) live net for the final eval.
ema = None

@torch.no_grad()
def sample(gc, gl, rhoc, rgbp, rayc, steps=a.steps):                                        # V14 FULL: diffuse z64(+color) directly (no zbase). Uses the unified sampler.
    return sample_z(gc, gl, rhoc, rgbp, rayc, zbase=None, steps=steps)

# ---- evaluate by DECODE: diffusion vs regression vs GT-latent vs procedural ----
torch.set_grad_enabled(False); net.eval(); reg.eval()
ck = torch.load(a.codec_path, weights_only=False)
cnorm = {k: torch.tensor(v, device=dev, dtype=torch.float32) for k, v in ck["fur_norm"].items()}
codec = StrandCodec(do_vae=False, decode_type="dir", scale_init=30.0, nr_verts_per_strand=256, nr_values_to_decode=255, dim_per_value_decoded=3).cuda().eval(); codec.load_state_dict(ck["state_dict"])
def decode_tex(Tz, rho, mask, size, seed=0):
    rho = rho[..., 0] if np.ndim(rho) == 3 else rho; mask = mask[..., 0] if np.ndim(mask) == 3 else mask
    lam = np.asarray(rho).copy(); rng = np.random.default_rng(seed); lam2 = np.clip(lam, 0, None) * (mask > 0.5)
    base = np.floor(lam2).astype(int); ex = (rng.uniform(0, 1, lam2.shape) < (lam2 - base)).astype(int); nc = (base + ex).ravel()
    nz = np.nonzero(nc)[0]; cells = np.repeat(nz, nc[nz])
    if len(cells) == 0: return np.zeros((1, 32, 3), np.float32), np.zeros((1, 2), np.float32)
    cj, ci = np.divmod(cells, H); u = (ci + rng.uniform(0, 1, len(cells))) / H; v = (cj + rng.uniform(0, 1, len(cells))) / H; uv = np.stack([u, v], 1).astype(np.float32)
    z = torch.tensor(_bilinear(Tz.transpose(1, 2, 0), uv), device=dev, dtype=torch.float32)
    pr = codec.decoder(z, None, cnorm); rec = (pr["strand_positions"] if "strand_positions" in pr else pr[[k for k in pr if "pos" in k.lower()][0]]).cpu().numpy()
    # resample 32 + place
    out = np.zeros((len(rec), 32, 3), np.float32)
    for n2, s in enumerate(rec):
        seg = np.linalg.norm(np.diff(s, axis=0), axis=1); cl = np.concatenate([[0], np.cumsum(seg)]); tt = np.linspace(0, cl[-1] + 1e-9, 32)
        out[n2] = np.stack([np.interp(tt, cl, s[:, k]) for k in range(3)], 1)
    root = np.stack([(u - .5) * size, (v - .5) * size, np.zeros(len(u))], 1).astype(np.float32)
    out = out - out[:, :1] + root[:, None]; out[..., 2] = np.maximum(out[..., 2], 0); out[:, 0] = root
    return out, uv
def _cov(p, g=24):
    ti2 = np.clip(((p[:, 0] + .6) / 1.2 * g).astype(int), 0, g - 1); tj2 = np.clip(((p[:, 1] + .6) / 1.2 * g).astype(int), 0, g - 1)
    h = np.zeros((g, g)); np.add.at(h, (tj2, ti2), 1.0); o = h[h > 0]; return o.std() / (o.mean() + 1e-9)
def clump(S):  # tip-concentration / root-concentration ratio (>1 = clumped); robust to root count
    return _cov(S[:, -1, :2]) / (_cov(S[:, 0, :2]) + 1e-9)
def curlf(S): seg = np.linalg.norm(S[:, 1:] - S[:, :-1], axis=2); arc = seg.sum(1); ch = np.linalg.norm(S[:, -1] - S[:, 0], axis=1); return float((1 - ch / np.maximum(arc, 1e-9)).mean())


# ===== DISTRIBUTIONAL evaluation: z64 is stochastic given G => evaluate p([z64,rho]|G), not specific-GT alignment =====
def gt_stats(S):
    seg = np.linalg.norm(S[:, 1:] - S[:, :-1], axis=2); arc = seg.sum(1); ch = np.linalg.norm(S[:, -1] - S[:, 0], axis=1)
    rise = np.clip(S[:, -1, 2] - S[:, 0, 2], 0, None) / np.maximum(arc, 1e-9)
    return dict(clump=clump(S), curl=float((1 - ch / np.maximum(arc, 1e-9)).mean()), length=float(arc.mean()), rise=float(rise.mean()), dens=float(len(S)))
def dec_stats_z(z64, rho_map, size, seed=0):                                                  # decode z64 at roots sampled from rho_map
    rho = np.clip(rho_map, 0, None); thr = 0.0 if a.decode_thr <= 0 else a.decode_thr * float(rho.max())
    S, _ = decode_tex(z64, rho, (rho > thr).astype(np.float32), size, seed=seed)
    if len(S) < 8: return None
    st = gt_stats(S); st["S"] = S; return st
def pcorr(u, v):
    u = np.asarray(u, float); v = np.asarray(v, float)
    return 0.0 if (u.std() < 1e-6 or v.std() < 1e-6) else float(np.corrcoef(u, v)[0, 1])

# ---- density head (rho_hat for tier B). Reconstruct RAW G/GLOB from this script's standardization, restandardize by the head's. ----
class DensityHead(nn.Module):
    def __init__(s, dim, w=128, ed=256):
        super().__init__(); uv = torch.tensor(np.stack([ug * 2 - 1, vg * 2 - 1], 0), dtype=torch.float32); s.register_buffer("uv", uv[None])
        s.gemb = nn.Sequential(nn.Linear(GLOBC, ed), nn.SiLU(), nn.Linear(ed, ed))
        s.inp = nn.Conv2d(dim + 2, w, 3, 1, 1); s.d1 = Res(w, w, ed); s.d2 = Res(w, 2 * w, ed); s.mid = Res(2 * w, 2 * w, ed)
        s.u2 = Res(4 * w, w, ed); s.u1 = Res(2 * w, w, ed); s.out = nn.Conv2d(w, 1, 3, 1, 1)
        s.dn = nn.AvgPool2d(2); s.up = nn.Upsample(scale_factor=2, mode="nearest")
    def forward(s, gc, gl):
        e = s.gemb(gl); h0 = s.inp(torch.cat([gc, s.uv.expand(len(gc), -1, -1, -1)], 1))
        a1 = s.d1(h0, e); a2 = s.d2(s.dn(a1), e); mm = s.mid(s.dn(a2), e)
        b2 = s.u2(torch.cat([s.up(mm), a2], 1), e); b1 = s.u1(torch.cat([s.up(b2), a1], 1), e); return s.out(b1)
_dhp = f"{a.out}/density_head.pt"; HAS_DH = os.path.exists(_dhp)
if HAS_DH:
    DH = torch.load(_dhp); dh = DensityHead(a.dim, DH["args"].get("width", 128)).to(dev).eval(); dh.load_state_dict(DH["net"])
    dgm = DH["gm"].to(dev); dgs = DH["gs"].to(dev); dglm = DH["glm"].to(dev); dgls = DH["gls"].to(dev)
    def rho_hat(idxt):
        gc_std = fetch(G_COND, idxt).float(); gl_std = fetch(GLOB, idxt).float()
        gc = ((gc_std * gs + gm - dgm) / dgs).half(); gl = (gl_std * gls + glm - dglm) / dgls   # undo this script's standardization -> head standardization
        return torch.expm1(dh(gc, gl)).clamp_min(0)
else:
    print("WARN: no density_head.pt -> tier B uses GT rho", flush=True)

valI = vi.tolist(); shuf = torch.roll(vi, 5)
rgt = raw_take(RHO, vi).float().cpu()                                                           # rho_gt maps (Nval,1,H,W), kept CPU for chunked eval
if HAS_DH:
    rh = torch.cat([rho_hat(vi[s:s + a.eval_batch]).cpu() for s in range(0, len(vi), a.eval_batch)], 0)
else:
    rh = rgt

def sample70_for(cond_idx, rho_maps):
    # V14 FULL: diffuse z64(+color) directly (no z_base). RESIDUAL: decode z = z_base + residual*res_std (V13-2 contract).
    use_residual = (not FULL) and (not a.no_diffusion and res_std_ready)
    outs = []
    eb = max(1, int(a.eval_batch))
    for s in range(0, len(cond_idx), eb):
        jj = cond_idx[s:s + eb]; rr = rho_maps[s:s + eb].to(dev)
        gcb = fetch(G_COND, jj); glb = fetch(GLOB, jj); rcb = torch.log1p(rr); rpb = fetch(RGBPROJ, jj); ryb = fetch(RAYC, jj)
        if use_residual:
            zb = reg(torch.zeros(len(jj), XD, H, H, device=dev), zc(len(jj)), gcb, glb, rcb, rpb, rayc=ryb)
            z = sample_z(gcb, glb, rcb, rpb, ryb, zbase=zb)
        else:
            z = sample(gcb, glb, rcb, rpb, ryb)
        outs.append((z * xs + xm).cpu())
    return torch.cat(outs, 0)

def reg70_for(cond_idx, rho_maps):
    outs = []
    eb = max(1, int(a.eval_batch))
    for s in range(0, len(cond_idx), eb):
        jj = cond_idx[s:s + eb]; rr = rho_maps[s:s + eb].to(dev)
        outs.append((reg(torch.zeros(len(jj), XD, H, H, device=dev), zc(len(jj)), fetch(G_COND, jj), fetch(GLOB, jj), torch.log1p(rr), fetch(RGBPROJ, jj), rayc=fetch(RAYC, jj)) * xs + xm).cpu())
    return torch.cat(outs, 0)

zA = sample70_for(vi, rgt)[:, :64].numpy()                                                       # tier A: z64 slice (GT-rho + raw pixels)
zB = sample70_for(vi, rh)[:, :64].numpy()                                                        # tier B: rho_hat
zS = sample70_for(shuf, rgt)[:, :64].numpy()                                                     # shuffle G+pixels
rgo = reg70_for(vi, rgt)[:, :64].numpy()
rgt_np = rgt.numpy()[:, 0]; rh_np = rh.numpy()[:, 0]
GT = []; A = []; B = []; Sx = []; Rg = []; sizes = []
for r, idx in enumerate(valI):
    size = float(np.load(f"{dirs[idx]}/patch.npz", allow_pickle=True)["size"]); sizes.append(size)
    Sg = np.load(f"{dirs[idx]}/patch.npz", allow_pickle=True)["strands"].astype(np.float32)
    GT.append(gt_stats(Sg)); A.append(dec_stats_z(zA[r], rgt_np[r], size)); B.append(dec_stats_z(zB[r], rh_np[r], size))
    Sx.append(dec_stats_z(zS[r], rgt_np[r], size)); Rg.append(dec_stats_z(rgo[r], rgt_np[r], size))
keep = [i for i in range(len(valI)) if A[i] and B[i] and Sx[i] and Rg[i]]
col = lambda lst, k: np.array([lst[i][k] for i in keep])
EVAL_KEYS = ["clump", "curl", "length", "rise", "dens"]
gt = {k: col(GT, k) for k in EVAL_KEYS}
A_ = {k: col(A, k) for k in EVAL_KEYS}; B_ = {k: col(B, k) for k in EVAL_KEYS}
S_ = {k: col(Sx, k) for k in EVAL_KEYS}; R_ = {k: col(Rg, k) for k in EVAL_KEYS}
print(f"\n====  v10 PIXEL+DINO DIFFUSION EVAL (z64 geometry, held-out {len(keep)}) | tierA=GT-rho  tierB=rho_hat  ====")
print(f"{'stat':7s}| GT          | tierA(GTrho) | tierB(rho_hat)| REG         | corr(A,GT) | corr(shufG,GT) | corr(REG,GT)")
for k in EVAL_KEYS:
    print(f"{k:7s}| {gt[k].mean():.2f}+-{gt[k].std():.2f} | {A_[k].mean():.2f}+-{A_[k].std():.2f}  | {B_[k].mean():.2f}+-{B_[k].std():.2f}   | {R_[k].mean():.2f}+-{R_[k].std():.2f} | "
          f"{pcorr(A_[k], gt[k]):+.2f}      | {pcorr(S_[k], gt[k]):+.2f}          | {pcorr(R_[k], gt[k]):+.2f}")
print("READ: corr(A,GT)>>corr(shufG,GT) => z64 uses G. clump std(tierA) vs GT = structure ceiling. tierB = real deployment (rho_hat).")

# ---- v10 COLOR eval (#3): predicted lit root/tip color vs GT, in the color-valid region only ----
if not a.no_color:
    p70 = sample70_for(vi, rgt)                                                                 # full 70ch
    pc = p70[:, 64:]; gcol = raw_take(COL, vi).cpu(); mvc = raw_take(CMASK, vi).cpu()
    cr_mse = float((((pc[:, :3] - gcol[:, :3]) ** 2) * mvc).sum() / (mvc.sum() * 3 + 1e-6))
    ct_mse = float((((pc[:, 3:] - gcol[:, 3:]) ** 2) * mvc).sum() / (mvc.sum() * 3 + 1e-6))
    cr_base = float((((gcol[:, :3].mean((2, 3), keepdim=True) - gcol[:, :3]) ** 2) * mvc).sum() / (mvc.sum() * 3 + 1e-6))
    print(f"COLOR EVAL (color-valid texels) | root RGB MSE {cr_mse:.4f} | tip RGB MSE {ct_mse:.4f} | per-patch-mean baseline {cr_base:.4f}  (color in [0,1])")
else:
    print("COLOR EVAL skipped (--no_color: z64-only ablation)")

viz = valI[:6]; fig, ax = plt.subplots(len(viz), 5, figsize=(16, 3.1 * len(viz))); ax = np.atleast_2d(ax)
for r, idx in enumerate(viz):
    name = os.path.basename(dirs[idx]); pos = valI.index(idx)
    Sg = np.load(f"{dirs[idx]}/patch.npz", allow_pickle=True)["strands"].astype(np.float32)
    cols = [(Sg, "GT actual"), ((A[pos] or {}).get("S"), "tierA GT-rho"), ((B[pos] or {}).get("S"), "tierB rho_hat"), ((Sx[pos] or {}).get("S"), "shuffled-G"), ((Rg[pos] or {}).get("S"), "REG mean")]
    for ci, (SS, ttl) in enumerate(cols):
        if SS is None or len(SS) < 2: ax[r, ci].axis("off"); ax[r, ci].set_title(ttl, fontsize=7); continue
        seg = np.stack([SS[:, :-1, :2], SS[:, 1:, :2]], 2).reshape(-1, 2, 2)
        az = (np.arctan2(SS[:, -1, 1] - SS[:, 0, 1], SS[:, -1, 0] - SS[:, 0, 0]) / (2 * np.pi) + 0.5)
        ax[r, ci].add_collection(LineCollection(seg, lw=.2, alpha=.5, cmap="hsv", array=np.repeat(az, SS.shape[1] - 1)))
        ax[r, ci].set_xlim(-.6, .6); ax[r, ci].set_ylim(-.6, .6); ax[r, ci].set_aspect("equal"); ax[r, ci].axis("off"); ax[r, ci].set_title(f"{ttl}\n{name if ci==0 else ''}", fontsize=7)
plt.tight_layout(); plt.savefig(f"{a.out}/z64_dist_topview.png", dpi=110); print(f"saved {a.out}/z64_dist_topview.png")
