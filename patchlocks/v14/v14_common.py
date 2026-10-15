"""V14 CALIBRATION shared machinery (THING-0 helper for train_param_head.py + rerank_eval.py).

Replicates the EXACT condition build + codec decode + full-mode CFG Heun sampler from
patchlocks/v14/train_v14.py, with the diffusion E1 + b3_fuser FROZEN. This module does NOT
retrain or touch E1/V10/V11/V12/V13 -- it loads the frozen weights and rebuilds the per-texel
conditions (G_COND via the frozen MVFuser, RHOC, RGBPROJ, RAYC, uv) and the seed-0 held-out split.

Hard contract (must match train_v14.py so calibration sees the same conditions E1 was trained on):
  - DINOv2 vitl14_reg, dino_res 448, shells 5, res 64, KUSE from the E1 ckpt (4).
  - b3_fuser pretrained MVFuser (dim 192) -> G_COND, standardized on the TRAIN split.
  - RGBPROJ centered (x-0.5)/0.25; RHOC=log1p(rho); RAYC (raydir3+grazing3+vis1) fed clean.
  - PATCH-LEVEL seed-0 split (15% val), identical generator/order to train_v14.py.
  - codec = the codec_path that baked --latent_name (fur_strand_codec_v6upright.pt); decode_params
    is copied verbatim from train_v14.py.

Usage:
    from v14_common import build_env
    env = build_env(renders=..., e1_dir=..., max_dirs=..., ...)
    # env.G_COND, env.RAYC, env.RHOC, env.RGBPROJ, env.GLOB, env.MASK, env.Xn, env.TZ,
    # env.xm, env.xs, env.vi, env.ti, env.dirs, env.decode_params, env.sample_z, env.fetch, ...
"""
import os, sys, glob, json, types
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "patchlocks/data_gen")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from models.strand_codec import StrandCodec

# ---- constants (must match train_v14.py) ----
C = 1024; GLOBC = 2 * C; POSE_DIM = 12; RAYC_CH = 7; Gdim = 256
PARAM_KEYS = ["arc", "straight", "lift_arc", "curl", "tip_spread", "clump"]

# Canonical coat-type classes derived from the dir-name archetype tokens (THING-1 coat logits).
# dataset tokens: ultra_short, short_dense, short_double_standoff, wavy, wiry, curly, fluffy_double,
# long_golden, long_silky  ->  7 task classes:
COAT_CLASSES = ["ultra_short", "short", "wavy", "wiry", "curly", "fluffy", "long"]
COAT_IDX = {c: i for i, c in enumerate(COAT_CLASSES)}
# per-coat recommended CFG-w (from the CFG sweep best-w heuristic in the task spec)
COAT_REC_W = {
    "ultra_short": 1.5, "short": 1.5, "wiry": 1.5,
    "wavy": 4.0, "curly": 4.0, "fluffy": 4.0, "long": 4.0,
}
DEFAULT_REC_W = 3.0


def coat_of(name):
    """Map a render dir basename to one of the 7 canonical coat classes (longest-token first)."""
    n = name.lower()
    if "ultra_short" in n: return "ultra_short"
    if "fluffy" in n: return "fluffy"
    if "curly" in n: return "curly"
    if "wavy" in n: return "wavy"
    if "wiry" in n: return "wiry"
    if "long" in n: return "long"          # long_golden / long_silky
    if "short" in n: return "short"        # short_dense / short_double_standoff
    return "short"


def coat_group_of(name):
    """Coarse coat GROUP for the rerank report (5 groups, task spec)."""
    c = coat_of(name)
    if c in ("ultra_short", "short", "wiry"):
        # wiry reported in its own slot too; keep wiry separate per spec ("wiry" group)
        return {"wiry": "wiry"}.get(c, "short/straight")
    if c == "wavy": return "wavy"
    if c == "curly": return "curly"
    if c in ("fluffy", "long"): return "fluffy/long"
    return "short/straight"


# ===== MVFuser (copied verbatim from train_v14.py; K-agnostic via pose token) =====
class MVFuser(nn.Module):
    def __init__(s, dim, nsh, pose_dim, K, H, idx_dev, nb=6):
        super().__init__()
        s.nsh = nsh; s.K = K; s.H = H; s.idx_dev = idx_dev
        s.proj = nn.Conv2d(C, dim, 1)
        s.raydir_proj = nn.Conv2d(6, dim, 1)
        s.semb = nn.Parameter(torch.randn(nsh, dim) * 0.02)
        s.pose = nn.Sequential(nn.Linear(pose_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        s.unseen = nn.Parameter(torch.randn(dim) * 0.02)
        s.attn = nn.Conv2d(dim, 1, 1)
        s.body = nn.ModuleList([nn.Sequential(nn.GroupNorm(8, dim), nn.SiLU(), nn.Conv2d(dim, dim, 3, 1, 1)) for _ in range(nb)])
        s._fetch = None  # set by build_env: closure over the cached tensors

    def forward(s, bi, views=None):
        K, H = s.K, s.H
        B = len(bi)
        V = K if views is None else int(views)
        vsel = torch.arange(V, device=s.idx_dev) % K
        fetch, FMAP, GRID, INF, RAYDIR, GRAZE, POSE, VISF, Hf = s._fetch
        fmap = fetch(FMAP, bi)[:, vsel].reshape(B * V, C, Hf, Hf).float()
        grid = fetch(GRID, bi)[:, vsel][:, :, :s.nsh].reshape(B * V, s.nsh * H, H, 2).float()
        samp = F.grid_sample(fmap, grid, mode="bilinear", align_corners=False).reshape(B, V, C, s.nsh, H, H)
        x = s.proj(samp.permute(0, 1, 3, 2, 4, 5).reshape(B * V * s.nsh, C, H, H)).reshape(B, V, s.nsh, -1, H, H)
        rd6 = torch.cat([fetch(RAYDIR, bi)[:, vsel], fetch(GRAZE, bi)[:, vsel]], 2)
        re = s.raydir_proj(rd6.reshape(B * V, 6, H, H)).reshape(B, V, 1, -1, H, H)
        pe = s.pose(fetch(POSE, bi)[:, vsel].reshape(B * V, -1)).reshape(B, V, 1, -1, 1, 1)
        x = x + re + pe + s.semb[None, None, :, :, None, None]
        vis = fetch(VISF, bi)[:, vsel]
        inf = fetch(INF, bi)[:, vsel][:, :, :s.nsh, None]
        m = (vis[:, :, None] > 0.5).float() * (inf > 0.5).float()
        x = x.reshape(B, V * s.nsh, -1, H, H); m = m.reshape(B, V * s.nsh, 1, H, H)
        logits = s.attn(x.reshape(B * V * s.nsh, -1, H, H)).reshape(B, V * s.nsh, 1, H, H).masked_fill(m < 0.5, -1e4)
        w = torch.softmax(logits, 1)
        seen = (m.sum(1, keepdim=True) > 0).float()
        w = w * seen
        fused = (x * w).sum(1) + (1.0 - seen[:, 0]) * s.unseen[None, :, None, None]
        for b in s.body: fused = fused + b(fused)
        return fused


# ===== UNet + EDM precond (copied from train_v14.py; FULL mode -> cond_zbase=False) =====
class Res(nn.Module):
    def __init__(s, ci, co, ed):
        super().__init__(); s.n1 = nn.GroupNorm(8, ci); s.c1 = nn.Conv2d(ci, co, 3, 1, 1); s.n2 = nn.GroupNorm(8, co); s.c2 = nn.Conv2d(co, co, 3, 1, 1)
        s.emb = nn.Linear(ed, 2 * co); s.sk = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
    def forward(s, x, e):
        h = s.c1(F.silu(s.n1(x))); sh = s.emb(e)[:, :, None, None]; sc, bi = sh.chunk(2, 1); h = s.c2(F.silu(s.n2(h) * (1 + sc) + bi)); return s.sk(x) + h

class BottleneckAttn(nn.Module):
    def __init__(s, ch, heads=4):
        super().__init__(); s.n = nn.GroupNorm(8, ch); s.a = nn.MultiheadAttention(ch, heads, batch_first=True)
        nn.init.zeros_(s.a.out_proj.weight); nn.init.zeros_(s.a.out_proj.bias)
    def forward(s, x):
        b, c, h, w = x.shape; q = s.n(x).flatten(2).transpose(1, 2)
        y = s.a(q, q, q, need_weights=False)[0].transpose(1, 2).reshape(b, c, h, w); return x + y

class UNet(nn.Module):
    def __init__(s, XD, ug, vg, dim=160, ed=256, cond_ch=None, use_attn=True, cond_zbase=False, cond_raydir=True):
        super().__init__(); cond_ch = cond_ch or C
        s.XD = XD; s.cond_zbase = bool(cond_zbase); s.cond_raydir = bool(cond_raydir)
        s.gp = nn.Conv2d(cond_ch, Gdim, 1); s.gp2 = nn.Conv2d(cond_ch, dim, 1)
        uv = torch.tensor(np.stack([ug * 2 - 1, vg * 2 - 1], 0), dtype=torch.float32); s.register_buffer("uv", uv[None])
        s.inp = nn.Conv2d(XD + (XD if cond_zbase else 0) + 1 + 3 + Gdim + 2 + (RAYC_CH if cond_raydir else 0), dim, 3, 1, 1)
        s.semb = nn.Sequential(nn.Linear(1, ed), nn.SiLU(), nn.Linear(ed, ed)); s.gemb = nn.Linear(GLOBC, ed)
        s.d1 = Res(dim, dim, ed); s.d2 = Res(dim, 2 * dim, ed); s.mid = Res(2 * dim + dim, 2 * dim, ed)
        s.attn = BottleneckAttn(2 * dim) if use_attn else nn.Identity()
        s.u2 = Res(4 * dim, dim, ed); s.u1 = Res(2 * dim, dim, ed); s.out = nn.Conv2d(dim, XD, 3, 1, 1)
        s.dn = nn.AvgPool2d(2); s.up = nn.Upsample(scale_factor=2, mode="nearest")
    def forward(s, x, cnoise, gc, gl, rhoc, rgbp, zbase=None, rayc=None):
        gcf = gc.float(); e = s.semb(cnoise[:, None]) + s.gemb(gl.float()); feats = [x]
        if s.cond_zbase: feats.append((torch.zeros_like(x) if zbase is None else zbase).float())
        feats += [rhoc, rgbp.float(), s.gp(gcf), s.uv.expand(len(x), -1, -1, -1)]
        if s.cond_raydir:
            rc = torch.zeros(len(x), RAYC_CH, x.shape[-2], x.shape[-1], device=x.device, dtype=torch.float32) if rayc is None else rayc.float()
            feats.append(rc)
        h0 = s.inp(torch.cat(feats, 1)); a1 = s.d1(h0, e); a2 = s.d2(s.dn(a1), e)
        gmid = s.gp2(F.avg_pool2d(gcf, 4)); mm = s.mid(torch.cat([s.dn(a2), gmid], 1), e); mm = s.attn(mm)
        b2 = s.u2(torch.cat([s.up(mm), a2], 1), e); b1 = s.u1(torch.cat([s.up(b2), a1], 1), e); return s.out(b1)


class Env(types.SimpleNamespace):
    pass


def build_env(renders, e1_dir, cond_stats=None, views_override=None,
              codec_path="patchlocks/models/fur_strand_codec_v6upright.pt",
              latent_name="target_latent64_v6codec.npz", pose_name="v13_pose_64.npz",
              res=64, dino_res=448, shells=5, fuser_dim=192,
              max_dirs=0, cache_device="cuda", len_k=32, verbose=True):
    """Build the frozen-condition environment matching train_v14.py. Returns an Env namespace.

    views_override: if set (e.g. 1), EVERY per-patch condition is built from only the first
    views_override views — GLOB/RGBPROJ (normally all-K), RAYC reduction and the fused G_COND
    (normally KUSE) — for true test-time K-view ablation. None = training-identical behavior."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    H = res; S = shells
    store_dev = torch.device(cache_device if torch.cuda.is_available() else "cpu")
    idx_dev = torch.device("cpu" if store_dev.type == "cpu" else dev)

    dirs = sorted(d for d in glob.glob(renders + "/*")
                  if os.path.exists(f"{d}/{latent_name}") and os.path.exists(f"{d}/field_gt_{H}.npz")
                  and os.path.exists(f"{d}/cameras.json") and os.path.exists(f"{d}/{pose_name}"))
    if not dirs:
        raise FileNotFoundError(f"no render dirs with {latent_name}+field_gt_{H}+cameras+{pose_name} under {renders}")
    if max_dirs and len(dirs) > max_dirs:
        keep = np.linspace(0, len(dirs) - 1, max_dirs).round().astype(int)
        dirs = [dirs[i] for i in sorted(set(keep.tolist()))]
        if verbose: print(f"--max_dirs {max_dirs}: subsampled to {len(dirs)} dirs", flush=True)
    K = len(json.load(open(f"{dirs[0]}/cameras.json"))["views"]); N = len(dirs)
    # KUSE from the E1 ckpt (the views the fuser/RAYC reduce over)
    e1_ck = torch.load(f"{e1_dir}/v14_best.pt", map_location="cpu", weights_only=False)
    KUSE = int(e1_ck.get("kuse", K))
    XD = int(e1_ck.get("xd", 64)); assert XD == 64, "this calibration assumes z64-only E1 (no_color)"
    # effective view counts: KV for the all-K conditions (GLOB/RGBPROJ), KR for the KUSE-reduced ones (RAYC/G_COND)
    KV = int(views_override) if views_override else K
    KR = int(views_override) if views_override else KUSE
    if verbose: print(f"{N} patches, K={K}, KUSE={KUSE} (KV={KV}, KR={KR}), H={H}, XD={XD}, store={store_dev}", flush=True)

    ii, jj = np.meshgrid(np.arange(H), np.arange(H)); ug = (ii + 0.5) / H; vg = (jj + 0.5) / H

    def shell_grids(meta, hmax):
        size = meta["size"]; hs = np.linspace(0, 1, S) * hmax; G = np.zeros((K, S, H, H, 2), np.float32); I = np.zeros((K, S, H, H), np.float32)
        for k in range(K):
            cam = meta["views"][k]; rr = cam["res"]; Minv = np.linalg.inv(np.array(cam["matrix_world"], np.float64)); fpx = cam["lens"] / cam["sensor_width"] * rr
            for si, h in enumerate(hs):
                Xw = np.stack([(ug - .5) * size, (vg - .5) * size, np.full_like(ug, h), np.ones_like(ug)], -1)
                Xc = (Xw.reshape(-1, 4) @ Minv.T)[:, :3].reshape(H, H, 3); z = Xc[..., 2]; fr = z < 0; zz = np.where(fr, z, -1e-6)
                px = rr / 2 + fpx * (Xc[..., 0] / -zz); py = rr / 2 - fpx * (Xc[..., 1] / -zz)
                G[k, si, ..., 0] = (px / rr) * 2 - 1; G[k, si, ..., 1] = (py / rr) * 2 - 1
                I[k, si] = (fr & (px >= 0) & (px < rr) & (py >= 0) & (py < rr)).astype(np.float32)
        return G, I

    import cv2
    Hf = dino_res // 14
    FMAP = torch.zeros(N, K, C, Hf, Hf, dtype=torch.float16, device=store_dev)
    GRID = torch.zeros(N, K, S, H, H, 2, dtype=torch.float16, device=store_dev); INF = torch.zeros(N, K, S, H, H, dtype=torch.float16, device=store_dev)
    GLOB = torch.zeros(N, GLOBC, device=store_dev)
    TZ = torch.zeros(N, 64, H, H, device=store_dev); RHO = torch.zeros(N, 1, H, H, device=store_dev); MASK = torch.zeros(N, 1, H, H, device=store_dev)
    RGBPROJ = torch.zeros(N, 3, H, H, device=store_dev)
    RAYDIR = torch.zeros(N, K, 3, H, H, device=store_dev); GRAZE = torch.zeros(N, K, 3, H, H, device=store_dev)
    POSE = torch.zeros(N, K, POSE_DIM, device=store_dev); VISF = torch.zeros(N, K, 1, H, H, device=store_dev)
    RAYC = torch.zeros(N, RAYC_CH, H, H, device=store_dev)

    def _idx_for(x, idx): return idx.detach().cpu() if x.device.type == "cpu" else idx.to(x.device)
    def raw_take(x, idx): return x[_idx_for(x, idx)]
    def fetch(x, idx, dtype=None):
        y = raw_take(x, idx)
        if y.device.type != "cuda" and dev == "cuda": y = y.to(dev, non_blocking=True)
        return y.to(dtype=dtype) if dtype is not None else y

    if verbose: print("DINOv2 + precompute conditions ...", flush=True)
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').to(dev).eval()
    for p in dino.parameters(): p.requires_grad = False
    mn = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]; st = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
    with torch.no_grad():
        for i, d in enumerate(dirs):
            meta = json.load(open(f"{d}/cameras.json"))
            ims = [cv2.resize(cv2.cvtColor(cv2.imread(f"{d}/rgb_{k}.png", -1), 4), (dino_res, dino_res)).astype(np.float32) / 255 for k in range(K)]
            im = torch.tensor(np.stack(ims)).permute(0, 3, 1, 2).to(dev)
            feat = dino.forward_features((im - mn) / st); tok = feat["x_norm_patchtokens"]; cls = feat["x_norm_clstoken"]
            FMAP[i] = tok.transpose(1, 2).reshape(K, C, Hf, Hf).half().to(store_dev)
            GLOB[i] = torch.cat([tok[:KV].mean((0, 1)), cls[:KV].mean(0)], 0).to(store_dev)
            p = np.load(f"{d}/patch.npz", allow_pickle=True); hmax = float(max(p["strands"][..., 2].max(), 1e-3))
            Gg, I = shell_grids(meta, hmax)
            Gg_t = torch.tensor(Gg, device=dev).half(); I_t = torch.tensor(I, device=dev).half()
            GRID[i] = Gg_t.to(store_dev); INF[i] = I_t.to(store_dev)
            g0 = Gg_t[:KV, 0].float(); samp = F.grid_sample(im[:KV].float(), g0, align_corners=False); inf0 = I_t[:KV, 0].float()[:, None]
            RGBPROJ[i] = ((samp * inf0).sum(0) / (inf0.sum(0) + 1e-6)).to(store_dev)
            pz = np.load(f"{d}/{pose_name}", allow_pickle=True)
            rd = np.asarray(pz["raydir_tbn"], np.float32)
            if rd.shape[1] != H: raise ValueError(f"{d}/{pose_name} raydir H={rd.shape[1]} != res {H}")
            RAYDIR[i] = torch.tensor(rd, device=store_dev).permute(0, 3, 1, 2)
            rn = np.clip(rd[..., 2], -1.0, 1.0); gr = np.stack([rn, np.abs(rn), np.arcsin(rn)], -1).astype(np.float32)
            GRAZE[i] = torch.tensor(gr, device=store_dev).permute(0, 3, 1, 2)
            POSE[i] = torch.tensor(np.asarray(pz["pose_view"], np.float32), device=store_dev)
            vf = np.asarray(pz["vis_frustum"], np.float32)[:, None]; VISF[i] = torch.tensor(vf, device=store_dev)
            vfr = np.asarray(pz["vis_frustum"], np.float32); rd6 = np.concatenate([rd, gr], -1).astype(np.float32)
            wv = vfr[:KR]; wsum = wv.sum(0); seen = wsum > 1e-6
            red6 = np.where(seen[..., None], (rd6[:KR] * wv[..., None]).sum(0) / np.maximum(wsum[..., None], 1e-6), rd6[:KR].mean(0))
            visred = np.clip(wv.max(0), 0.0, 1.0)
            rayc = np.concatenate([red6, visred[..., None]], -1).astype(np.float32)
            RAYC[i] = torch.tensor(rayc, device=store_dev).permute(2, 0, 1)
            tl = np.load(f"{d}/{latent_name}")
            if tl["target"].shape[:2] != (H, H): raise ValueError(f"{d}/{latent_name} target {tl['target'].shape[:2]} != res {H}")
            TZ[i] = torch.tensor(tl["target"], device=store_dev).permute(2, 0, 1)
            rho = tl["density"]; rho = rho[..., 0] if rho.ndim == 3 else rho
            msk = tl["mask"]; msk = msk[..., 0] if msk.ndim == 3 else msk
            RHO[i, 0] = torch.tensor(np.asarray(rho, np.float32), device=store_dev)
            MASK[i, 0] = torch.tensor(np.asarray(msk, np.float32), device=store_dev)
            if verbose and (i + 1) % 100 == 0: print(f"  {i+1}/{N}", flush=True)
    del dino; torch.cuda.empty_cache()

    # ---- PATCH-LEVEL seed-0 split (identical to train_v14.py) ----
    g_ = torch.Generator().manual_seed(0)
    _pid = [os.path.basename(d).rsplit("_v", 1)[0] for d in dirs]
    _uniq = sorted(set(_pid)); _pp = torch.randperm(len(_uniq), generator=g_).tolist()
    _valset = set(_uniq[_pp[i]] for i in range(max(1, int(len(_uniq) * 0.15))))
    _isval = torch.tensor([p in _valset for p in _pid])
    vi = torch.nonzero(_isval, as_tuple=True)[0].to(idx_dev); ti = torch.nonzero(~_isval, as_tuple=True)[0].to(idx_dev)
    if verbose: print(f"PATCH-LEVEL split: {len(_uniq)} patches -> {len(_valset)} val ({len(vi)} dirs)/{len(ti)} train dirs", flush=True)

    # ---- load FROZEN b3_fuser -> G_COND, then standardize on TRAIN split (exactly as train_v14.py) ----
    b3 = MVFuser(fuser_dim, S, POSE_DIM, K, H, idx_dev).to(dev)
    bck = torch.load(f"{e1_dir}/b3_fuser.pt", map_location=dev)
    assert bck["dim"] == fuser_dim and bck["shells"] == S, "b3_fuser dim/shells mismatch"
    b3.load_state_dict(bck["b3"]); b3.eval()
    for p in b3.parameters(): p.requires_grad = False
    b3._fetch = (fetch, FMAP, GRID, INF, RAYDIR, GRAZE, POSE, VISF, Hf)
    if verbose: print("precompute frozen G_COND ...", flush=True)
    gbs = 8
    with torch.inference_mode():
        G_COND = torch.cat([b3(torch.arange(s, min(s + gbs, N), device=idx_dev), KR).half().cpu() for s in range(0, N, gbs)], 0)
    del b3, FMAP, GRID, INF, RAYDIR, GRAZE, POSE, VISF; torch.cuda.empty_cache()
    if store_dev.type == "cuda":
        G_COND = G_COND.to(dev); RAYC = RAYC.to(dev)

    if cond_stats is not None:
        # override with TRAINING-run statistics (e.g. the arm ckpt's saved cond/glob stats) so a
        # foreign renders root (independent test set) sees training-identical normalization
        gm_raw = cond_stats["cond_mean"].to(G_COND.device).float(); gs_raw = cond_stats["cond_std"].to(G_COND.device).float().clamp_min(1e-4)
        glm_raw = cond_stats["glob_mean"].to(GLOB.device).float(); gls_raw = cond_stats["glob_std"].to(GLOB.device).float().clamp_min(1e-4)
    else:
        gm_raw = raw_take(G_COND, ti).float().mean((0, 2, 3), keepdim=True); gs_raw = raw_take(G_COND, ti).float().std((0, 2, 3), keepdim=True, unbiased=False).clamp_min(1e-4)
        glm_raw = raw_take(GLOB, ti).float().mean(0, keepdim=True); gls_raw = raw_take(GLOB, ti).float().std(0, keepdim=True, unbiased=False).clamp_min(1e-4)
    G_COND = ((G_COND.float() - gm_raw) / gs_raw).half()
    GLOB = (GLOB - glm_raw) / gls_raw
    RGBPROJ = (RGBPROJ - 0.5) / 0.25
    RHOC = torch.log1p(RHO.clamp_min(0))

    # ---- target standardization: use E1's SAVED xm/xs (NOT re-estimated) so sampled z decodes correctly ----
    xm = e1_ck["xm"].to(dev).float(); xs = e1_ck["xs"].to(dev).float()  # (1,64,1,1)
    X = TZ  # z64-only
    Xn = (X - xm) / xs

    # ---- load FROZEN E1 diffusion net (EMA weights saved in v14_best.pt) ----
    use_attn = bool(e1_ck.get("attn", 1)); unet_dim = int(e1_ck.get("unet_dim", 256))
    net = UNet(XD, ug, vg, unet_dim, cond_ch=fuser_dim, use_attn=use_attn, cond_zbase=False, cond_raydir=True).to(dev)
    net.load_state_dict(e1_ck["net"]); net.eval()
    for p in net.parameters(): p.requires_grad = False
    SD = float(e1_ck.get("edm_sd", 1.0))
    sigma_max = float(e1_ck.get("sigma_max", 80.0)); sigma_min = float(e1_ck.get("sigma_min", 0.002))
    steps = 40
    if verbose: print(f"loaded FROZEN E1 net (it {e1_ck.get('iter')}) | sigma_max={sigma_max} SD={SD}", flush=True)

    def precond(x, sig, gc, gl, rhoc, rgbp, rayc):
        sig = sig.reshape(-1, 1, 1, 1); cs = SD**2 / (sig**2 + SD**2); co = sig * SD / (sig**2 + SD**2).sqrt()
        ci = 1 / (sig**2 + SD**2).sqrt(); cn = sig.flatten().log() / 4
        return cs * x + co * net(ci * x, cn, gc, gl, rhoc, rgbp, None, rayc)

    @torch.no_grad()
    def sample_z(gc, gl, rhoc, rgbp, rayc, cfg=1.0, seed=0, steps_=None, sample_temp=1.0):
        """FULL-mode CFG Heun sampler (matches train_v14.sample_z). Returns STANDARDIZED z64.
        ALL-condition CFG: the null branch zeros gc, gl, rhoc, rgbp, rayc together."""
        st_ = steps if steps_ is None else steps_
        gen = torch.Generator(device=dev).manual_seed(int(seed))
        sigs = torch.linspace(0, 1, st_ + 1, device=dev)
        sigs = (sigma_max**(1 / 7) + sigs * (sigma_min**(1 / 7) - sigma_max**(1 / 7)))**7
        sigs = torch.cat([sigs, torch.zeros(1, device=dev)])
        x = torch.randn(len(gc), XD, H, H, device=dev, generator=gen) * sigs[0] * sample_temp
        null = lambda t: (None if t is None else torch.zeros_like(t))
        def denoise(xx, ss):
            dc = precond(xx, ss.expand(len(gc)), gc, gl, rhoc, rgbp, rayc)
            if cfg != 1.0:
                du = precond(xx, ss.expand(len(gc)), null(gc), null(gl), null(rhoc), null(rgbp), null(rayc))
                dc = du + cfg * (dc - du)
            return dc
        for j in range(st_):
            s0, s1 = sigs[j], sigs[j + 1]; dd = (x - denoise(x, s0)) / s0; xn = x + (s1 - s0) * dd
            if s1 > 0: d2 = (xn - denoise(xn, s1)) / s1; xn = x + (s1 - s0) * 0.5 * (dd + d2)
            x = xn
        return x

    # ---- codec + decode_params (copied verbatim from train_v14.py) ----
    _lck = torch.load(codec_path, weights_only=False)
    _lcn = {k: torch.tensor(v, device=dev, dtype=torch.float32) for k, v in _lck["fur_norm"].items()}
    _lcodec = StrandCodec(do_vae=False, decode_type="dir", scale_init=30.0, nr_verts_per_strand=256, nr_values_to_decode=255, dim_per_value_decoded=3).to(dev).eval()
    _lcodec.load_state_dict(_lck["state_dict"])
    for p in _lcodec.parameters(): p.requires_grad = False

    def _decode_rec(z_std, idxb, idxhw):
        """Decode standardized z64 at sampled texels -> root-centered strand positions (M,P,3)."""
        z = (z_std[:, :64] * xs + xm).permute(0, 2, 3, 1).reshape(len(z_std), H * H, 64)[idxb, idxhw]
        pr = _lcodec.decoder(z, None, _lcn)
        rec = pr["strand_positions"] if "strand_positions" in pr else pr[[k for k in pr if "pos" in k.lower()][0]]
        return rec - rec[:, :1]

    def decode_params(z_std, idxb, idxhw, lk):
        rec = _decode_rec(z_std, idxb, idxhw)
        seg = (rec[:, 1:] - rec[:, :-1]).norm(dim=2); arc = seg.sum(1).clamp_min(1e-6)
        tip = rec[:, -1, :2]; chord2 = tip.norm(dim=1); chord3 = rec[:, -1].norm(dim=1); straight = chord3 / arc
        lift_arc = rec[..., 2].abs().max(1).values / arc
        ang = torch.atan2(tip[:, 1], tip[:, 0])
        yc = torch.sin(-ang)[:, None] * rec[..., 0] + torch.cos(-ang)[:, None] * rec[..., 1]
        curl = yc.abs().mean(1) / arc
        nv = len(idxb) // lk; R = lambda x: x.reshape(nv, lk)
        cm = R(torch.cos(ang)).mean(1); sm = R(torch.sin(ang)).mean(1)
        med = lambda x: R(x).median(1).values
        return dict(arc=med(arc), straight=med(straight), lift_arc=med(lift_arc), curl=med(curl),
                    tip_spread=R(chord2).std(1, unbiased=False) / med(arc).clamp_min(1e-6),
                    clump=(cm ** 2 + sm ** 2).clamp_min(1e-9).sqrt())

    def decode_params_full(z_std, idxb, idxhw, lk):
        """Same patch-level medians as decode_params PLUS per-param texel STDs and the raw per-texel
        tip directions (for tangle scoring). Returns (means_dict, stds_dict, per_texel_dict)."""
        rec = _decode_rec(z_std, idxb, idxhw)
        seg = (rec[:, 1:] - rec[:, :-1]).norm(dim=2); arc = seg.sum(1).clamp_min(1e-6)
        tip = rec[:, -1, :2]; chord3 = rec[:, -1].norm(dim=1); straight = chord3 / arc
        lift_arc = rec[..., 2].abs().max(1).values / arc
        ang = torch.atan2(tip[:, 1], tip[:, 0])
        yc = torch.sin(-ang)[:, None] * rec[..., 0] + torch.cos(-ang)[:, None] * rec[..., 1]
        curl = yc.abs().mean(1) / arc
        nv = len(idxb) // lk; R = lambda x: x.reshape(nv, lk)
        med = lambda x: R(x).median(1).values; sd = lambda x: R(x).std(1, unbiased=False)
        cm = R(torch.cos(ang)).mean(1); sm = R(torch.sin(ang)).mean(1)
        means = dict(arc=med(arc), straight=med(straight), lift_arc=med(lift_arc), curl=med(curl),
                     clump=(cm ** 2 + sm ** 2).clamp_min(1e-9).sqrt())
        ang_p = R(ang)  # (nv, lk) per-texel tip azimuth
        stds = dict(curl=sd(curl), straight=sd(straight), arc=sd(arc), lift_arc=sd(lift_arc),
                    clump=ang_p.std(1, unbiased=False))  # clump_std proxy = per-patch tip-angle dispersion
        per = dict(ang=ang_p)  # (nv, lk) per-texel tip azimuth for tangle scoring
        return means, stds, per

    def sample_texels(m, lk, seed=None):
        """Pick lk masked texels per patch (matches train_v14._sample_texels but row-index = KEPT order).
        Returns (ib, ihw, kept_bool); kept_bool aligns to the INPUT batch order. ib indexes into z_std
        rows by ORIGINAL batch position (so decode_params slices the right patch)."""
        B = m.shape[0]; mf = (m[:, 0] > 0.5).reshape(B, H * H)
        gen = None if seed is None else torch.Generator(device=dev).manual_seed(int(seed))
        rnd = lambda hi, n: (torch.randint(hi, (n,), device=dev) if gen is None else torch.randint(hi, (n,), device=dev, generator=gen))
        ib = []; ihw = []; kept = []
        for bb in range(B):
            nz = mf[bb].nonzero(as_tuple=True)[0]
            if len(nz) == 0: kept.append(False); continue
            kept.append(True)
            ib.append(torch.full((lk,), bb, device=dev))           # ORIGINAL batch row (decode slices z_std[bb])
            ihw.append(nz[rnd(len(nz), lk)])
        if not ib: return None, None, torch.zeros(B, dtype=torch.bool, device=dev)
        return torch.cat(ib), torch.cat(ihw), torch.tensor(kept, device=dev)

    env = Env(dev=dev, H=H, S=S, XD=XD, K=K, KUSE=KUSE, N=N, len_k=len_k,
              dirs=dirs, vi=vi, ti=ti, idx_dev=idx_dev, store_dev=store_dev,
              G_COND=G_COND, GLOB=GLOB, RHOC=RHOC, RGBPROJ=RGBPROJ, RAYC=RAYC,
              MASK=MASK, RHO=RHO, TZ=TZ, Xn=Xn, xm=xm, xs=xs,
              fetch=fetch, raw_take=raw_take, net=net, precond=precond, sample_z=sample_z,
              decode_params=decode_params, decode_params_full=decode_params_full,
              sample_texels=sample_texels, sigma_max=sigma_max, sigma_min=sigma_min, steps=steps,
              coat_of=coat_of, coat_group_of=coat_group_of)
    return env
