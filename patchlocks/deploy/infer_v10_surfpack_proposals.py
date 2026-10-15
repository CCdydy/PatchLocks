#!/usr/bin/env python3
"""Run v10 diffusion on a whole surface_patch_dataset and emit per-patch V2 proposals.

This is the first whole-dog V2 adapter. It deliberately stops at proposal/prior
data, not final grow:

  surface_patch_dataset.npz
    -> close-up crops from pack uv/vis
    -> V2 texture-prior + DINO geometry diffusion
    -> patch proposal npz + density table
    -> build_v2_region_prior.py consumes the table with --pack

When baked world-to-clip matrices are available, every shell is projected with
real parallax:

  shell point = surf + h * normal, h in [0, 0.25 * patch_size]
  clip = homo(shell point) @ baked_matrix

If matrices are missing, the script falls back to scalp uv reuse; that mode is
only for plumbing/debug.
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "patchlocks")
sys.path.insert(0, "patchlocks/data_gen")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dir_extractor import extract_dir_field, resolve_axial_sign, fur_texture_highlight  # measured fur direction + texture clarity from the crop
from models.texture_prior import build_texture_prior, aggregate_texture_prior, center_texture_prior, TexStem, TPC


torch.set_grad_enabled(False)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
C = 1024
GLOBC = 2 * C
Gdim = 256


def gate_axial_map(raw_dir, conf, valid, gate_mode="soft", global_prior=0.08, coh_gate=0.65):
    raw_dir = np.asarray(raw_dir, np.float32)
    conf = np.asarray(conf, np.float32)
    valid = np.asarray(valid, bool)
    mag = np.linalg.norm(raw_dir, axis=-1)
    local = raw_dir / (mag[..., None] + 1e-9)
    conf = np.clip(conf, 0, 1) * valid.astype(np.float32) * np.clip(mag, 0, 1)
    gv = (local * conf[..., None]).reshape(-1, 2).sum(0).astype(np.float32)
    v = gv / np.linalg.norm(gv) if np.linalg.norm(gv) > 1e-9 else np.array([1.0, 0.0], np.float32)
    if gate_mode == "hard":
        out = local.copy()
        out[(conf < float(coh_gate)) | (~valid)] = v
        return out.astype(np.float32), conf.astype(np.float32)
    blend = local * conf[..., None] + v[None, None, :] * max(float(global_prior), 0.0)
    bn = np.linalg.norm(blend, axis=-1)
    out = (blend / (bn[..., None] + 1e-9)).astype(np.float32)
    out[bn < 1e-8] = v
    return out, conf.astype(np.float32)


def weighted_axial_mean(axial, conf, valid):
    w = np.asarray(conf, np.float32) * np.asarray(valid, bool).astype(np.float32)
    acc = (np.asarray(axial, np.float32) * w[..., None]).reshape(-1, 2).sum(0)
    nrm = float(np.linalg.norm(acc))
    if nrm <= 1e-9:
        return np.zeros(2, np.float32), 0.0
    return (acc / nrm).astype(np.float32), float(min(nrm / max(float(w.sum()), 1.0), 1.0))


def crop_pad(img, x0, y0, x1, y1):
    h0, w0 = img.shape[:2]
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, w0), min(y1, h0)
    sub = img[cy0:cy1, cx0:cx1]
    top, left, bot, right = cy0 - y0, cx0 - x0, y1 - cy1, x1 - cx1
    if top or left or bot or right:
        sub = cv2.copyMakeBorder(sub, top, bot, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return sub


def texture_hf_score(img_rgb):
    tex = fur_texture_highlight(img_rgb)
    gx = cv2.Sobel(tex, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(tex, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.percentile(mag, 90.0))


def load_matrices(path):
    if not path or not os.path.exists(path):
        return None
    mm = pickle.load(open(path, "rb"))
    return mm["matrices"] if isinstance(mm, dict) and "matrices" in mm else mm


def matrix_for_view(mats, view_id, index):
    if mats is None:
        return None
    keys = [view_id, str(view_id), int(view_id) if str(view_id).isdigit() else None, index]
    for k in keys:
        if k is not None and isinstance(mats, dict) and k in mats:
            return np.asarray(mats[k], np.float64)
    if not isinstance(mats, dict) and index < len(mats):
        return np.asarray(mats[index], np.float64)
    return None


def camera_center_from_clip(M):
    # Row-vector homogeneous camera center: C @ M has x=y=w=0.
    A = M[:, [0, 1, 3]].T
    _u, _s, vh = np.linalg.svd(A)
    c = vh[-1]
    den = c[3] if abs(c[3]) > 1e-8 else (1e-8 if c[3] >= 0 else -1e-8)
    return (c[:3] / den).astype(np.float32)


class B3Fuser(nn.Module):
    def __init__(self, dim, k_views, shells, h, hf, nb=6):
        super().__init__()
        self.k_views = k_views
        self.shells = shells
        self.h = h
        self.hf = hf
        self.proj = nn.Conv2d(C, dim, 1)
        self.vemb = nn.Parameter(torch.randn(k_views, dim) * 0.02)
        self.semb = nn.Parameter(torch.randn(shells, dim) * 0.02)
        self.cam = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))
        self.attn = nn.Conv2d(dim, 1, 1)
        self.body = nn.ModuleList(
            [nn.Sequential(nn.GroupNorm(8, dim), nn.SiLU(), nn.Conv2d(dim, dim, 3, 1, 1)) for _ in range(nb)]
        )

    def forward(self, fmap, grid, inf, cam):
        b = fmap.shape[0]
        h = self.h
        fmap_b = fmap.reshape(b * self.k_views, C, self.hf, self.hf).float()
        grid_b = grid[:, :, : self.shells].reshape(b * self.k_views, self.shells * h, h, 2).float()
        samp = F.grid_sample(fmap_b, grid_b, align_corners=False).reshape(
            b, self.k_views, C, self.shells, h, h
        )
        inf_b = inf[:, :, : self.shells].float()
        x = self.proj(
            samp.permute(0, 1, 3, 2, 4, 5).reshape(b * self.k_views * self.shells, C, h, h)
        ).reshape(b, self.k_views, self.shells, -1, h, h)
        ce = self.cam(cam.reshape(b * self.k_views, 6)).reshape(b, self.k_views, 1, -1, 1, 1)
        x = x + self.vemb[None, :, None, :, None, None] + self.semb[None, None, :, :, None, None] + ce
        x = x.reshape(b, self.k_views * self.shells, -1, h, h)
        mask = inf_b.reshape(b, self.k_views * self.shells, 1, h, h)
        w = torch.softmax(
            self.attn(x.reshape(b * self.k_views * self.shells, -1, h, h)).reshape(
                b, self.k_views * self.shells, 1, h, h
            ).masked_fill(mask < 0.5, -1e4),
            1,
        )
        w = w * (mask.sum(1, keepdim=True) > 0)
        fused = (x * w).sum(1)
        for block in self.body:
            fused = fused + block(fused)
        return fused


class Res(nn.Module):
    def __init__(self, ci, co, ed):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ci)
        self.c1 = nn.Conv2d(ci, co, 3, 1, 1)
        self.n2 = nn.GroupNorm(8, co)
        self.c2 = nn.Conv2d(co, co, 3, 1, 1)
        self.emb = nn.Linear(ed, 2 * co)
        self.sk = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        sh = self.emb(e)[:, :, None, None]
        sc, bi = sh.chunk(2, 1)
        h = self.c2(F.silu(self.n2(h) * (1 + sc) + bi))
        return self.sk(x) + h


class Attn(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.n = nn.GroupNorm(8, ch)
        self.a = nn.MultiheadAttention(ch, heads, batch_first=True)
        nn.init.zeros_(self.a.out_proj.weight)
        nn.init.zeros_(self.a.out_proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.n(x).flatten(2).transpose(1, 2)
        y = self.a(q, q, q, need_weights=False)[0].transpose(1, 2).reshape(b, c, h, w)
        return x + y


class UNet(nn.Module):
    def __init__(self, xd, rpc, dim, h, ed=256, cond_ch=None, attn=False, cond_inject="base",
                 cond_mode="rgbproj", tex_stem_width=16):
        super().__init__()
        cond_ch = cond_ch or C
        self.xd = xd
        self.rpc = rpc
        self.cond_inject = cond_inject
        self.cond_mode = cond_mode
        self.gp = nn.Conv2d(cond_ch, Gdim, 1)
        self.gp2 = nn.Conv2d(cond_ch, dim, 1)
        if cond_mode in ("texture_prior", "both"):
            self.tex_stem = TexStem.build(rpc, tex_stem_width)
            self.cond_in = nn.Conv2d(tex_stem_width, tex_stem_width, 1)
            cond_inp_ch = tex_stem_width
        else:
            self.tex_stem = None
            self.cond_in = None
            cond_inp_ch = rpc
        if self.cond_inject == "strong":
            self.cond_hi = nn.Sequential(nn.Conv2d(rpc, dim, 3, 1, 1), nn.SiLU(), nn.Conv2d(dim, dim, 3, 1, 1))
            self.cond_mid = nn.Conv2d(dim, dim, 1)
            self.g_hi = nn.Conv2d(cond_ch, dim, 1)
        ii, jj = np.meshgrid(np.arange(h), np.arange(h))
        ug = (ii + 0.5) / h
        vg = (jj + 0.5) / h
        uv = torch.tensor(np.stack([ug * 2 - 1, vg * 2 - 1], 0), dtype=torch.float32)
        self.register_buffer("uv", uv[None])
        self.inp = nn.Conv2d(xd + 1 + cond_inp_ch + Gdim + 2, dim, 3, 1, 1)
        self.semb = nn.Sequential(nn.Linear(1, ed), nn.SiLU(), nn.Linear(ed, ed))
        self.gemb = nn.Linear(GLOBC, ed)
        self.d1 = Res(dim, dim, ed)
        self.d2 = Res(dim, 2 * dim, ed)
        self.mid = Res(2 * dim + dim, 2 * dim, ed)
        self.attn = Attn(2 * dim) if attn else None
        self.u2 = Res(4 * dim, dim, ed)
        self.u1 = Res(2 * dim, dim, ed)
        self.out = nn.Conv2d(dim, xd, 3, 1, 1)
        self.dn = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, cnoise, gc, gl, rhoc, condp):
        gcf = gc.float()
        e = self.semb(cnoise[:, None]) + self.gemb(gl.float())
        cf = condp.float()
        cin = self.cond_in(self.tex_stem(cf)) if self.cond_mode in ("texture_prior", "both") else cf
        h0 = self.inp(torch.cat([x, rhoc, cin, self.gp(gcf), self.uv.expand(len(x), -1, -1, -1)], 1))
        if self.cond_inject == "strong":
            rhi = self.cond_hi(cf)
            ghi = self.g_hi(gcf)
            h0 = h0 + rhi + ghi
        else:
            rhi = ghi = None
        a1 = self.d1(h0, e)
        a2 = self.d2(self.dn(a1), e)
        gmid = self.gp2(F.avg_pool2d(gcf, 4))
        if self.cond_inject == "strong":
            gmid = gmid + self.cond_mid(F.avg_pool2d(rhi, 4)) + F.avg_pool2d(ghi, 4)
        mm = self.mid(torch.cat([self.dn(a2), gmid], 1), e)
        if self.attn is not None:
            mm = self.attn(mm)
        b2 = self.u2(torch.cat([self.up(mm), a2], 1), e)
        b1 = self.u1(torch.cat([self.up(b2), a1], 1), e)
        return self.out(b1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="surface_patch_dataset.npz")
    ap.add_argument("--geo_dir", default="patchlocks/datasets/stage3_z64_canon_density")
    ap.add_argument("--model", default="", help="stage3_model*.pt; default <geo_dir>/stage3_model.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--matrices", default="", help="optional baked matrices; default uses pack data_dir/.../30000_matrices.pkl when present")
    ap.add_argument("--max_patches", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dino_res", type=int, default=448)
    ap.add_argument("--crop_margin", type=float, default=1.2)
    ap.add_argument("--crop_mode", choices=["comfort_window", "patch_bbox"], default="comfort_window",
                    help="comfort_window crops a fixed native context window instead of magnifying the projected patch bbox; patch_bbox is the legacy close crop")
    ap.add_argument("--comfort_crop_px", type=float, default=256.0,
                    help="native crop side for --crop_mode comfort_window; chosen to match the successful real-patch SR/input regime")
    ap.add_argument("--length_remap_ref_model_px", type=float, default=0.0,
                    help="model-space patch footprint treated as the training/reference scale for length remap; <=0 uses dino_res/crop_margin")
    ap.add_argument("--length_remap_clip", type=float, default=4.0,
                    help="clip predicted length remap to [1/clip, clip]")
    ap.add_argument("--comfort_patch_px_min", type=float, default=80.0,
                    help="native projected patch size below this is not trusted as a V10 feature anchor")
    ap.add_argument("--comfort_patch_px_good", type=float, default=192.0,
                    help="native/effective projected patch size at which V10 feature confidence saturates")
    ap.add_argument("--comfort_upscale_good", type=float, default=3.0,
                    help="crop upsample factor <= this is considered sharp enough")
    ap.add_argument("--comfort_upscale_bad", type=float, default=5.5,
                    help="crop upsample factor >= this is considered too blurred for diffusion features")
    ap.add_argument("--sr_scale_assumed", type=float, default=1.0,
                    help="metadata only: multiply native patch px for scale confidence when an actual SR preprocessor is used upstream")
    ap.add_argument("--default_density", type=float, default=4.0)
    ap.add_argument("--view_select", choices=["scale", "pack"], default="scale",
                    help="scale: choose the candidate view whose crop is closest to the V10 sharp-patch comfort zone; pack: use pack['sel'] order")
    ap.add_argument("--dir_shell_frac", type=float, default=0.5,
                    help="shell fraction used for measured direction input: 0=scalp, 0.5=middle fur body")
    ap.add_argument("--dir_texture_mode", choices=["gray", "highlight", "line", "gabor"], default="gray",
                    help="texture source for measured direction extraction. gray is conservative; highlight/gabor are experimental texture-enhanced modes.")
    ap.add_argument("--dir_gate_mode", choices=["soft", "hard"], default="soft",
                    help="soft blends local direction with a patch-global prior; hard uses coh gate fallback")
    ap.add_argument("--dir_global_prior", type=float, default=0.08,
                    help="soft-gate patch-global prior strength for measured/head direction maps")
    ap.add_argument("--dir_coh_gate", type=float, default=0.65,
                    help="minimum local direction confidence before falling back in hard gate and grow")
    ap.add_argument("--direction_source", choices=["auto", "measured", "diffusion"], default="auto",
                    help="selected direction written to patch_dir. auto uses diffusion when the model predicts direction, otherwise measured texture direction.")
    ap.add_argument("--save_maps", action="store_true", help="save z64/color/density maps as float16; large on full packs")
    ap.add_argument("--save_z64_map_npy", action="store_true",
                    help="stream patch_z64_map to out/patch_z64_map.npy as float16 memmap and store its path in the proposal; avoids --save_maps OOM")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pack = np.load(args.pack, allow_pickle=True)
    centers = pack["centers"].astype(np.float32)
    Tt = pack["Tt"].astype(np.float32)
    Bb = pack["Bb"].astype(np.float32) if "Bb" in pack.files else np.cross(pack["Nn"], Tt).astype(np.float32)
    Nn = pack["Nn"].astype(np.float32)
    surf = pack["surf"].astype(np.float32)                                       # (P,H,H,3) world surface points -> multi-shell lift along Nn
    psize = float(pack["psize"]) if "psize" in pack.files else 0.04
    shell_hmax = max(0.25 * psize, 1e-3)                                          # match training/infer shell_grids: shells span 0..0.25*patch_size
    valid = pack["valid"].astype(np.float32)
    uvp = pack["uv"].astype(np.float32)
    visp = pack["vis"].astype(np.float32)
    sel = pack["sel"].astype(np.int32)
    cand_names = [str(x) for x in pack["cand_names"]]
    view_ids = [str(x) for x in pack["view_ids"]] if "view_ids" in pack.files else [str(i) for i in range(len(cand_names))]
    H = int(pack["H"])
    P = len(centers) if args.max_patches <= 0 else min(len(centers), args.max_patches)

    model_path = args.model or os.path.join(args.geo_dir, "stage3_model.pt")
    GEO = torch.load(model_path, map_location=DEV)
    B3 = torch.load(os.path.join(args.geo_dir, "b3_fuser.pt"), map_location=DEV)
    GA = GEO["args"]
    dim = int(GA["dim"])
    XD = int(GEO.get("xd", 70))
    color_dim = int(GEO.get("color_dim", 6))
    has_color = color_dim >= 6
    cond_mode = str(GEO.get("cond_mode", GA.get("cond_mode", "rgbproj")))
    cond_image = str(GEO.get("cond_image", "texture_prior_v1" if cond_mode == "texture_prior" else "rgbproj"))
    if cond_image == "texture_prior_v1":
        cond_mode = "texture_prior"
    elif cond_image in ("tex+rgb", "texture_prior_v1+rgbproj"):
        cond_mode = "both"
    rpc = int(GEO.get("cond_image_ch", TPC if cond_mode == "texture_prior" else (TPC + 3 if cond_mode == "both" else 3)))
    if cond_mode not in ("texture_prior", "rgbproj", "both"):
        raise ValueError(f"unsupported cond_mode={cond_mode!r}")
    if cond_mode == "texture_prior" and rpc != TPC:
        raise ValueError(f"texture_prior checkpoint says cond_image_ch={rpc}, expected {TPC}")
    if cond_mode == "both" and rpc != TPC + 3:
        raise ValueError(f"both checkpoint says cond_image_ch={rpc}, expected {TPC + 3}")
    tex_stem_width = int(GEO.get("tex_stem_width", 16))
    unet_dim = int(GEO.get("unet_dim", GA.get("unet_dim", 160)))
    texture_norm = GEO.get("texture_norm", {}) or {}
    texture_hf_xs = float(texture_norm.get("hf_xs", texture_norm.get("xs_hf", 1.0)))
    pred_density = bool(GEO.get("pred_density", False))
    dens_i = int(GEO.get("dens_i", -1))
    pred_length = bool(GEO.get("pred_length", False))
    len_i = int(GEO.get("len_i", -1))
    length_transform = str(GEO.get("length_transform", "raw"))
    pred_direction = bool(GEO.get("pred_direction", False))
    dir_i = int(GEO.get("dir_i", -1))
    pred_radius = bool(GEO.get("pred_radius", False))
    rad_i = int(GEO.get("rad_i", -1))
    radius_transform = str(GEO.get("radius_transform", "log"))
    cond_inject = str(GA.get("cond_inject", "base"))
    direction_source = "diffusion" if args.direction_source == "auto" and pred_direction and dir_i >= 0 else ("measured" if args.direction_source == "auto" else args.direction_source)
    shells = int(B3["b3"]["semb"].shape[0])
    model_k = int(B3["b3"]["vemb"].shape[0])
    if sel.shape[1] < model_k:
        raise ValueError(f"pack has K={sel.shape[1]} selected views but model expects K={model_k}")
    Hf = args.dino_res // 14
    print(
        f"v10-surfpack-proposals | P={P}/{len(centers)} H={H} K={model_k} shells={shells} "
        f"XD={XD} color_dim={color_dim} pred_density={pred_density} pred_length={pred_length} "
        f"pred_direction={pred_direction} pred_radius={pred_radius} cond={cond_image}/{rpc}ch cond_inject={cond_inject} unet_dim={unet_dim} "
        f"direction_source={direction_source} dir_texture={args.dir_texture_mode} model={model_path}",
        flush=True,
    )

    xm = GEO["xm"].to(DEV)
    xs = GEO["xs"].to(DEV)
    gm = GEO["cond_mean"].to(DEV)
    gs = GEO["cond_std"].to(DEV)
    glm = GEO["glob_mean"].to(DEV)
    gls = GEO["glob_std"].to(DEV)
    smin, smax = GA["sigma_min"], GA["sigma_max"]
    SD = 1.0

    b3 = B3Fuser(dim, model_k, shells, H, Hf).to(DEV).eval()
    b3.load_state_dict(B3["b3"])
    net = UNet(
        XD, rpc, unet_dim, H, cond_ch=dim, attn=bool(GA.get("attn", 0)),
        cond_inject=cond_inject, cond_mode=cond_mode, tex_stem_width=tex_stem_width,
    ).to(DEV).eval()
    state = GEO["net"]
    # Back-compat for abandoned strong-condition checkpoints that named the
    # generic condition residuals after RGB/texture specifically.
    if any(k.startswith("rgb_hi.") or k.startswith("rgb_mid.") or k.startswith("tex_hi.") or k.startswith("tex_mid.") for k in state):
        remap = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("rgb_hi."):
                nk = "cond_hi." + nk[len("rgb_hi."):]
            elif nk.startswith("rgb_mid."):
                nk = "cond_mid." + nk[len("rgb_mid."):]
            elif nk.startswith("tex_hi."):
                nk = "cond_hi." + nk[len("tex_hi."):]
            elif nk.startswith("tex_mid."):
                nk = "cond_mid." + nk[len("tex_mid."):]
            remap[nk] = v
        state = remap
    net.load_state_dict(state)

    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg").to(DEV).eval()
    mn = torch.tensor([0.485, 0.456, 0.406], device=DEV)[:, None, None]
    st = torch.tensor([0.229, 0.224, 0.225], device=DEV)[:, None, None]
    frames = [cv2.cvtColor(cv2.imread(n), cv2.COLOR_BGR2RGB) for n in cand_names]
    if any(f is None for f in frames):
        bad = [n for n, f in zip(cand_names, frames) if f is None][:3]
        raise FileNotFoundError(f"missing frame(s): {bad}")

    matrices = args.matrices
    if not matrices and "data_dir" in pack.files:
        cand = os.path.join(str(pack["data_dir"]), "3d_gaussian_splatting", "GS_", "cameras", "30000_matrices.pkl")
        if os.path.exists(cand):
            matrices = cand
    mats = load_matrices(matrices)
    cam_centers = []
    for k in range(len(cand_names)):
        M = matrix_for_view(mats, view_ids[k] if k < len(view_ids) else k, k)
        cam_centers.append(camera_center_from_clip(M) if M is not None else None)
    print(f"matrices={matrices or 'none'} | camera centers available {sum(c is not None for c in cam_centers)}/{len(cam_centers)}", flush=True)
    svals = np.linspace(0, 1, shells, dtype=np.float64) * shell_hmax              # shell heights (world); multi-shell parallax via baked matrices
    print(f"multi-shell projection: {'baked-matrix (real parallax)' if mats is not None else 'scalp-reuse (degraded)'} | psize={psize:.4f} shell_hmax={shell_hmax:.4f}", flush=True)

    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=DEV).manual_seed(args.seed)
    patch_id = np.arange(P).astype(np.int32)
    coverage = np.zeros(P, np.float32)
    density_mean = np.zeros(P, np.float32)
    length_mean = np.zeros(P, np.float32)
    radius_mean = np.zeros(P, np.float32)
    texture_hf = np.zeros(P, np.float32)
    texture_count = np.zeros(P, np.float32)
    patch_native_px = np.zeros(P, np.float32)                                      # best-view projected patch footprint in the original photo
    patch_effective_px = np.zeros(P, np.float32)                                   # native footprint after any real upstream SR scale
    patch_crop_px = np.zeros(P, np.float32)                                        # best-view crop side before resize to dino_res
    patch_model_px = np.zeros(P, np.float32)                                       # patch footprint after resize into model input
    patch_upscale = np.full(P, np.inf, np.float32)                                 # dino_res / crop_px for the best scale-conf view
    patch_scale_conf = np.zeros(P, np.float32)                                     # reliability: does the crop live in V10's sharp-patch comfort zone?
    patch_length_remap = np.ones(P, np.float32)                                    # predicted length scale -> real patch scale
    patch_selected_view = np.full(P, -1, np.int32)
    rgbproj_color = np.zeros((P, 3), np.float32)                                              # REAL per-patch observed albedo (color source; diffusion color demoted to residual)
    color_mean = np.zeros((P, 6), np.float32)
    z64_mean = np.zeros((P, 64), np.float32)
    patch_dir = np.zeros((P, 2), np.float32)                                      # chosen axial [cos2theta,sin2theta] direction (UV frame)
    patch_dir_coh = np.zeros(P, np.float32)                                       # chosen confidence/consistency (0..1)
    patch_dir_measured = np.zeros((P, 2), np.float32)                             # structure-tensor measured axial direction
    patch_dir_measured_coh = np.zeros(P, np.float32)
    patch_dir_diffusion = np.zeros((P, 2), np.float32)                            # diffusion-predicted axial direction
    patch_dir_diffusion_coh = np.zeros(P, np.float32)
    patch_dir_map = np.zeros((P, H, H, 2), np.float16)                             # locally continuous oriented [cos theta,sin theta] direction map
    patch_dir_map_conf = np.zeros((P, H, H), np.float16)
    conf = np.clip(valid[:P].reshape(P, -1).mean(1), 0, 1).astype(np.float32)
    maps_z = [] if args.save_maps else None
    maps_c = [] if args.save_maps else None
    maps_r = [] if args.save_maps and pred_density else None
    z64_map_path = ""
    z64_map_mm = None
    if args.save_z64_map_npy:
        z64_map_path = os.path.join(args.out, "patch_z64_map.npy")
        z64_map_mm = np.lib.format.open_memmap(
            z64_map_path, mode="w+", dtype=np.float16, shape=(P, H, H, 64)
        )
        print(f"  streaming patch_z64_map -> {z64_map_path} shape={(P, H, H, 64)}", flush=True)

    sigs = torch.linspace(0, 1, args.steps + 1, device=DEV)
    sigs = (smax ** (1 / 7) + sigs * (smin ** (1 / 7) - smax ** (1 / 7))) ** 7
    sigs = torch.cat([sigs, torch.zeros(1, device=DEV)])

    def _smoothstep(lo, hi, x):
        if hi <= lo + 1e-6:
            return 1.0 if x >= hi else 0.0
        t = np.clip((float(x) - float(lo)) / (float(hi) - float(lo)), 0.0, 1.0)
        return float(t * t * (3.0 - 2.0 * t))

    def _scale_conf(native_patch_px, crop_side_px):
        eff = float(native_patch_px) * max(float(args.sr_scale_assumed), 1e-6)
        px_conf = _smoothstep(args.comfort_patch_px_min, args.comfort_patch_px_good, eff)
        up = float(args.dino_res) / max(float(crop_side_px), 1e-6)
        if args.comfort_upscale_bad <= args.comfort_upscale_good + 1e-6:
            up_conf = 1.0
        else:
            up_conf = 1.0 - _smoothstep(args.comfort_upscale_good, args.comfort_upscale_bad, up)
        return float(np.clip(px_conf * up_conf, 0.0, 1.0)), eff, up

    length_ref_model_px = float(args.length_remap_ref_model_px)
    if length_ref_model_px <= 0:
        length_ref_model_px = float(args.dino_res) / max(float(args.crop_margin), 1e-6)

    def _crop_half(iw, ih, bbox_side, fallback=False):
        if fallback:
            bbox_side = 0.12 * max(float(iw), float(ih)) / max(float(args.crop_margin), 1e-6)
        legacy_half = max(0.5 * float(args.crop_margin) * float(bbox_side), 8.0)
        if args.crop_mode == "patch_bbox":
            return legacy_half
        return max(legacy_half, 0.5 * float(args.comfort_crop_px))

    def _crop_scale_metrics(p, k):
        img = frames[int(k)]
        ih, iw = img.shape[:2]
        uv = uvp[p, int(k)]
        px = (uv[..., 0] + 1.0) * 0.5 * iw
        py = (uv[..., 1] + 1.0) * 0.5 * ih
        m = (visp[p, int(k)] > 0.5) & (valid[p] > 0.5)
        xs0 = px[m]
        ys0 = py[m]
        if xs0.size >= 4:
            bbox_side = max(float(xs0.max() - xs0.min()), float(ys0.max() - ys0.min()))
            half = _crop_half(iw, ih, bbox_side)
        else:
            bbox_side = 0.0
            half = _crop_half(iw, ih, bbox_side, fallback=True)
        crop_side = 2.0 * float(half)
        sc_conf, eff_patch_px, up_factor = _scale_conf(bbox_side, crop_side)
        return sc_conf, eff_patch_px, up_factor, bbox_side

    def _select_views_for_patch(p):
        cand = [int(x) for x in sel[p] if int(x) >= 0]
        if not cand:
            return [0] * model_k
        if args.view_select == "scale":
            scored = []
            for k in cand:
                sc_conf, eff_patch_px, up_factor, bbox_side = _crop_scale_metrics(p, k)
                scored.append((sc_conf, eff_patch_px, bbox_side, -up_factor, k))
            scored.sort(reverse=True)
            chosen = [int(x[-1]) for x in scored[:model_k]]
        else:
            chosen = cand[:model_k]
        while len(chosen) < model_k:
            chosen.append(chosen[-1])
        return chosen

    def precond(x, sig, gc, gl, rhoc, condp):
        sig = sig.reshape(-1, 1, 1, 1)
        cs = SD**2 / (sig**2 + SD**2)
        co = sig * SD / (sig**2 + SD**2).sqrt()
        ci = 1 / (sig**2 + SD**2).sqrt()
        cn = sig.flatten().log() / 4
        return cs * x + co * net(ci * x, cn, gc, gl, rhoc, condp)

    for s0 in range(0, P, args.chunk):
        s1 = min(P, s0 + args.chunk)
        bsz = s1 - s0
        crops = np.full((bsz * model_k, args.dino_res, args.dino_res, 3), 1.0, np.float32)
        grid_np = np.zeros((bsz, model_k, shells, H, H, 2), np.float32)
        inf_np = np.zeros((bsz, model_k, shells, H, H), np.float32)
        cam_np = np.zeros((bsz, model_k, 6), np.float32)
        view_cos_np = np.ones((bsz, model_k), np.float32)

        selected_views = [_select_views_for_patch(p) for p in range(s0, s1)]
        for bi, p in enumerate(range(s0, s1)):
            patch_selected_view[p] = int(selected_views[bi][0])
            for kk in range(model_k):
                k = int(selected_views[bi][kk])
                img = frames[k]
                ih, iw = img.shape[:2]
                uv = uvp[p, k]
                px = (uv[..., 0] + 1.0) * 0.5 * iw
                py = (uv[..., 1] + 1.0) * 0.5 * ih
                m = (visp[p, k] > 0.5) & (valid[p] > 0.5)
                xs0 = px[m]
                ys0 = py[m]
                if xs0.size >= 4:
                    bbox_side = max(float(xs0.max() - xs0.min()), float(ys0.max() - ys0.min()))
                    cx = 0.5 * (float(xs0.min()) + float(xs0.max()))
                    cy = 0.5 * (float(ys0.min()) + float(ys0.max()))
                    half = _crop_half(iw, ih, bbox_side)
                else:
                    bbox_side = 0.0
                    cx = float(np.clip(px[H // 2, H // 2], 0, iw - 1))
                    cy = float(np.clip(py[H // 2, H // 2], 0, ih - 1))
                    half = _crop_half(iw, ih, bbox_side, fallback=True)
                crop_side = 2.0 * float(half)
                sc_conf, eff_patch_px, up_factor = _scale_conf(bbox_side, crop_side)
                if sc_conf >= patch_scale_conf[p]:
                    model_px = float(bbox_side / max(crop_side, 1e-6) * args.dino_res)
                    remap = length_ref_model_px / max(model_px, 1e-6)
                    remap = float(np.clip(remap, 1.0 / max(float(args.length_remap_clip), 1.0), max(float(args.length_remap_clip), 1.0)))
                    patch_scale_conf[p] = sc_conf
                    patch_native_px[p] = float(bbox_side)
                    patch_effective_px[p] = float(eff_patch_px)
                    patch_crop_px[p] = float(crop_side)
                    patch_model_px[p] = model_px
                    patch_upscale[p] = float(up_factor)
                    patch_length_remap[p] = remap
                x0, x1 = int(round(cx - half)), int(round(cx + half))
                y0, y1 = int(round(cy - half)), int(round(cy + half))
                crops[bi * model_k + kk] = cv2.resize(crop_pad(img, x0, y0, x1, y1), (args.dino_res, args.dino_res)).astype(np.float32) / 255.0
                texture_hf[p] += texture_hf_score(crops[bi * model_k + kk])
                texture_count[p] += 1.0
                gx = (px - (cx - half)) / (2 * half) * 2 - 1
                gy = (py - (cy - half)) / (2 * half) * 2 - 1
                Mk = matrix_for_view(mats, view_ids[k] if k < len(view_ids) else k, k)
                if Mk is not None:                                              # MULTI-SHELL baked-matrix parallax: lift surf along Nn, project world->clip (matches v9lite 0.01px)
                    base = surf[p].reshape(-1, 3).astype(np.float64); nrm = Nn[p].astype(np.float64)
                    for si in range(shells):
                        pts = base + svals[si] * nrm
                        clip = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ Mk
                        w = clip[:, 3]; fr = np.abs(w) > 1e-8
                        ndc = np.zeros((len(pts), 2)); ndc[fr] = clip[fr, :2] / w[fr, None]
                        pxs = ((ndc[:, 0] + 1.0) * 0.5 * iw).reshape(H, H)
                        pys = ((-ndc[:, 1] + 1.0) * 0.5 * ih).reshape(H, H)     # y-flip = uvp convention
                        grid_np[bi, kk, si, ..., 0] = (pxs - (cx - half)) / (2 * half) * 2 - 1
                        grid_np[bi, kk, si, ..., 1] = (pys - (cy - half)) / (2 * half) * 2 - 1
                        inf_np[bi, kk, si] = m & fr.reshape(H, H)
                else:                                                           # no baked matrices -> degraded scalp reuse
                    for si in range(shells):
                        grid_np[bi, kk, si, ..., 0] = gx
                        grid_np[bi, kk, si, ..., 1] = gy
                        inf_np[bi, kk, si] = m
                cc = cam_centers[k]
                if cc is not None:
                    dirw = cc - centers[p]
                    dirw = dirw / (np.linalg.norm(dirw) + 1e-9)
                    dloc = np.array([Tt[p] @ dirw, Bb[p] @ dirw, Nn[p] @ dirw], np.float32)
                    cam_np[bi, kk, :3] = dloc * 2.78
                    cam_np[bi, kk, 3:] = -dloc
                    view_cos_np[bi, kk] = float(np.clip(abs(dloc[2]), 0.0, 1.0))

        im = torch.tensor(crops, device=DEV).permute(0, 3, 1, 2)
        feat = dino.forward_features((im - mn) / st)
        tok = feat["x_norm_patchtokens"].reshape(bsz, model_k, -1, C)
        cls = feat["x_norm_clstoken"].reshape(bsz, model_k, C)
        fmap = tok.permute(0, 1, 3, 2).reshape(bsz, model_k, C, Hf, Hf).half()
        glob = torch.cat([tok.mean((1, 2)), cls.mean(1)], 1)
        grid = torch.tensor(grid_np, device=DEV).half()
        inf = torch.tensor(inf_np, device=DEV).half()
        cam = torch.tensor(cam_np, device=DEV)
        g0 = grid[:, :, 0].reshape(bsz * model_k, H, H, 2).float()
        samp = F.grid_sample(im.float(), g0, align_corners=False).reshape(bsz, model_k, 3, H, H)
        inf0 = inf[:, :, 0].float()[:, :, None]
        rgbproj_raw = (samp * inf0).sum(1) / (inf0.sum(1) + 1e-6)                             # REAL observed albedo (multi-view projected input RGB, [0,1]) -> saved per-patch as the color source (diffusion color is unreliable at dark/saturated extremes)
        rgbproj_np = rgbproj_raw.permute(0, 2, 3, 1).detach().cpu().numpy()                   # (bsz,H,H,3)
        rgbproj = (rgbproj_raw - 0.5) / 0.25
        tex_tensor = None
        if cond_mode in ("texture_prior", "both"):
            tex_np = np.zeros((bsz, TPC, H, H), np.float32)
            for bi in range(bsz):
                texs, infs = [], []
                for kk in range(model_k):
                    inf0_k = inf_np[bi, kk, 0].astype(np.float32)
                    texs.append(
                        build_texture_prior(
                            crops[bi * model_k + kk],
                            grid_np[bi, kk, 0],
                            inf0_k > 0.5,
                            args.dino_res,
                            view_cos=float(view_cos_np[bi, kk]),
                        )
                    )
                    infs.append(inf0_k)
                tex_avg = aggregate_texture_prior(texs, infs)
                tex_avg[:, np.stack(infs, 0).sum(0) <= 0] = 0.0
                tex_np[bi] = tex_avg
            tex_tensor = torch.tensor(center_texture_prior(tex_np, texture_hf_xs), device=DEV)
            condp = tex_tensor if cond_mode == "texture_prior" else torch.cat([tex_tensor, rgbproj], 1)
        else:
            condp = rgbproj
        gcond = (b3(fmap, grid, inf, cam) - gm) / gs
        globn = (glob - glm) / gls
        rhoc = torch.zeros(bsz, 1, H, H, device=DEV) if pred_density else torch.full((bsz, 1, H, H), np.log1p(args.default_density), device=DEV)

        x = torch.randn(bsz, XD, H, H, device=DEV, generator=gen) * sigs[0]
        for j in range(args.steps):
            q0, q1 = sigs[j], sigs[j + 1]
            d0 = (x - precond(x, q0.expand(bsz), gcond, globn, rhoc, condp)) / q0
            xn = x + (q1 - q0) * d0
            if q1 > 0:
                d2 = (xn - precond(xn, q1.expand(bsz), gcond, globn, rhoc, condp)) / q1
                xn = x + (q1 - q0) * 0.5 * (d0 + d2)
            x = xn
        out = (x * xs + xm).permute(0, 2, 3, 1).detach().cpu().numpy()
        for bi, p in enumerate(range(s0, s1)):
            vm = valid[p] > 0.5
            z = out[bi, ..., :64]
            cm = np.clip(out[bi, ..., 64:70], 0, 1) if has_color else np.zeros((H, H, 6), np.float32)
            if pred_density and dens_i >= 0:
                rho = np.clip(np.expm1(out[bi, ..., dens_i]), 0, None).astype(np.float32)
            else:
                rho = np.full((H, H), float(args.default_density), np.float32)
            if pred_length and len_i >= 0:
                lraw = out[bi, ..., len_i]
                if length_transform == "log":
                    lmap = np.exp(lraw).astype(np.float32)
                else:
                    lmap = np.maximum(lraw, 0.0).astype(np.float32)
                lmap = (lmap * float(patch_length_remap[p])).astype(np.float32)
            else:
                lmap = np.zeros((H, H), np.float32)
            if pred_radius and rad_i >= 0:
                rraw = out[bi, ..., rad_i]
                rmap = (np.exp(rraw) if radius_transform == "log" else np.maximum(rraw, 0.0)).astype(np.float32)
            else:
                rmap = np.zeros((H, H), np.float32)
            diff_vec_map = None
            diff_conf_map = None
            if pred_direction and dir_i >= 0:
                da = out[bi, ..., dir_i:dir_i + 2].astype(np.float32)
                mag = np.linalg.norm(da, axis=-1)
                axial = da / (mag[..., None] + 1e-9)
                good = vm & np.isfinite(axial).all(-1) & (mag > 1e-6)
                dconf = np.clip(mag, 0.0, 1.0).astype(np.float32) * good.astype(np.float32)
                accd = (axial * dconf[..., None]).reshape(-1, 2).sum(0)
                nrm = float(np.linalg.norm(accd))
                wsumd = float(dconf.sum())
                if wsumd > 1e-6 and nrm > 1e-9:
                    patch_dir_diffusion[p] = (accd / (nrm + 1e-9)).astype(np.float32)
                    patch_dir_diffusion_coh[p] = float(min(nrm / (wsumd + 1e-9), 1.0))
                diff_vec_map = resolve_axial_sign(axial, dconf, good)
                diff_conf_map = dconf
            vv = vm & np.isfinite(rho)
            if not vv.any():
                vv = np.ones((H, H), bool)
            density_mean[p] = float(rho[vv].mean())
            if pred_length and len_i >= 0:
                lv = vv & np.isfinite(lmap) & (lmap > 0)
                length_mean[p] = float(lmap[lv].mean()) if lv.any() else 0.0
            if pred_radius and rad_i >= 0:
                rv = vv & np.isfinite(rmap) & (rmap > 0)
                radius_mean[p] = float(rmap[rv].mean()) if rv.any() else 0.0
            _cvm = vm & np.isfinite(rgbproj_np[bi]).all(-1) & (rgbproj_np[bi].sum(-1) > 1e-3)  # observed (in-frame, non-black) texels
            rgbproj_color[p] = rgbproj_np[bi][_cvm if _cvm.any() else vv].mean(0)              # real albedo for this patch
            coverage[p] = float((rho[vv] > 0.05).mean() * rho[vv].mean())
            z64_mean[p] = z[vv].mean(0)
            if z64_map_mm is not None:
                z64_map_mm[p] = z.astype(np.float16)
            if cm.shape[-1] == 6:
                color_mean[p] = cm[vv].mean(0)
            acc = np.zeros(2); wsum = 0.0                                     # measured direction: coherence-weighted over views (UV frame)
            acc_map = np.zeros((H, H, 2), np.float32)
            w_map = np.zeros((H, H), np.float32)
            view_count = 0
            dsi = int(np.clip(round(float(args.dir_shell_frac) * (shells - 1)), 0, shells - 1))
            for kk in range(model_k):
                mvk = inf_np[bi, kk, dsi] > 0.5
                if mvk.sum() < 8:
                    continue
                df = extract_dir_field(crops[bi * model_k + kk], grid_np[bi, kk, dsi], mvk, args.dino_res, texture_mode=args.dir_texture_mode)
                wk = df[2][mvk]
                acc += np.array([float((df[0][mvk] * wk).sum()), float((df[1][mvk] * wk).sum())]); wsum += float(wk.sum())
                wf = df[2] * mvk.astype(np.float32)
                acc_map += np.moveaxis(df[:2], 0, -1) * wf[..., None]
                w_map += wf
                view_count += 1
            if wsum > 1e-6:
                nrm = np.linalg.norm(acc)
                patch_dir_measured[p] = acc / (nrm + 1e-9); patch_dir_measured_coh[p] = float(min(nrm / (wsum + 1e-9), 1.0))
            if view_count > 0:
                raw_axial = acc_map / (w_map[..., None] + 1e-9)
                conf_raw = np.clip(w_map / max(float(view_count), 1.0), 0.0, 1.0)
                good_map = (w_map > 1e-6) & np.isfinite(raw_axial).all(-1)
                axial_map, conf_map = gate_axial_map(
                    raw_axial,
                    conf_raw,
                    good_map,
                    args.dir_gate_mode,
                    args.dir_global_prior,
                    args.dir_coh_gate,
                )
                vec_map = resolve_axial_sign(axial_map, conf_map, good_map)
                patch_dir_map[p] = vec_map.astype(np.float16)
                patch_dir_map_conf[p] = conf_map.astype(np.float16)
            patch_dir[p] = patch_dir_measured[p]; patch_dir_coh[p] = patch_dir_measured_coh[p]
            if direction_source == "diffusion" and patch_dir_diffusion_coh[p] > 1e-6:
                patch_dir[p] = patch_dir_diffusion[p]
                patch_dir_coh[p] = patch_dir_diffusion_coh[p]
                if diff_vec_map is not None and diff_conf_map is not None:
                    patch_dir_map[p] = diff_vec_map.astype(np.float16)
                    patch_dir_map_conf[p] = diff_conf_map.astype(np.float16)
            if args.save_maps:
                maps_z.append(z.astype(np.float16))
                maps_c.append(cm.astype(np.float16))
                if pred_density:
                    maps_r.append(rho.astype(np.float16))
        print(f"  patches {s1}/{P} | coverage mean {coverage[:s1].mean():.3f}", flush=True)

    out_npz = os.path.join(args.out, "v10_surfpack_proposals.npz")
    texture_hf_mean = texture_hf / np.maximum(texture_count, 1.0)
    tex_valid = np.isfinite(texture_hf_mean) & (texture_count > 0)
    texture_conf = np.zeros_like(texture_hf_mean, dtype=np.float32)
    if tex_valid.any():
        tex_vals = texture_hf_mean[tex_valid].astype(np.float32)
        qlo, qhi = np.percentile(tex_vals, [20.0, 90.0])
        if qhi <= qlo + 1e-6:
            qlo, qhi = float(tex_vals.min()), float(tex_vals.max())
        texture_conf[tex_valid] = np.clip((tex_vals - qlo) / max(qhi - qlo, 1e-6), 0.0, 1.0)
        # Visibility gates the texture rank, but the rank itself is whole-animal relative.
        texture_conf[tex_valid] *= np.clip(conf[tex_valid], 0.0, 1.0)
        print(
            f"  texture_conf robust rank q20={qlo:.4f} q90={qhi:.4f} "
            f"range [{texture_conf.min():.3f},{texture_conf.max():.3f}] "
            f"above0.35 {int((texture_conf >= 0.35).sum())}/{P}",
            flush=True,
        )
    save = dict(
        patch_id=patch_id,
        patch_coverage_pred=coverage.astype(np.float32),
        patch_density_mean=density_mean.astype(np.float32),
        patch_length_mean=length_mean.astype(np.float32),
        patch_texture_hf=texture_hf_mean.astype(np.float32),
        patch_texture_conf=texture_conf.astype(np.float32),
        patch_native_px=patch_native_px.astype(np.float32),
        patch_effective_px=patch_effective_px.astype(np.float32),
        patch_crop_px=patch_crop_px.astype(np.float32),
        patch_model_px=patch_model_px.astype(np.float32),
        patch_upscale=patch_upscale.astype(np.float32),
        patch_scale_conf=patch_scale_conf.astype(np.float32),
        patch_length_remap=patch_length_remap.astype(np.float32),
        patch_selected_view=patch_selected_view.astype(np.int32),
        patch_radius_mean=radius_mean.astype(np.float32),
        patch_z64_mean=z64_mean.astype(np.float32),
        patch_color_root_tip=color_mean.astype(np.float32),
        patch_rgbproj_color=rgbproj_color.astype(np.float32),
        patch_dir=patch_dir.astype(np.float32),
        patch_dir_coh=patch_dir_coh.astype(np.float32),
        patch_dir_vec_map=patch_dir_map,
        patch_dir_vec_map_conf=patch_dir_map_conf,
        patch_dir_source=np.asarray(direction_source),
        patch_dir_measured=patch_dir_measured.astype(np.float32),
        patch_dir_measured_coh=patch_dir_measured_coh.astype(np.float32),
        patch_dir_diffusion=patch_dir_diffusion.astype(np.float32),
        patch_dir_diffusion_coh=patch_dir_diffusion_coh.astype(np.float32),
        patch_conf=conf.astype(np.float32),
        source_pack=np.asarray(args.pack),
        geo_dir=np.asarray(args.geo_dir),
        model=np.asarray(model_path),
        cond_image=np.asarray(cond_image),
        cond_mode=np.asarray(cond_mode),
        cond_image_ch=np.asarray(rpc, dtype=np.int32),
        shell_projection=np.asarray("baked_matrix_multishell" if mats is not None else "scalp_uv_repeated"),
        view_select=np.asarray(args.view_select),
        crop_mode=np.asarray(args.crop_mode),
        comfort_crop_px=np.float32(args.comfort_crop_px),
        length_remap_ref_model_px=np.float32(length_ref_model_px),
        length_remap_clip=np.float32(args.length_remap_clip),
        comfort_patch_px_min=np.float32(args.comfort_patch_px_min),
        comfort_patch_px_good=np.float32(args.comfort_patch_px_good),
        comfort_upscale_good=np.float32(args.comfort_upscale_good),
        comfort_upscale_bad=np.float32(args.comfort_upscale_bad),
        sr_scale_assumed=np.float32(args.sr_scale_assumed),
    )
    if args.save_maps:
        save["patch_z64_map"] = np.asarray(maps_z, np.float16)
        save["patch_color_map"] = np.asarray(maps_c, np.float16)
        if pred_density:
            save["patch_density_map"] = np.asarray(maps_r, np.float16)
    if z64_map_mm is not None:
        z64_map_mm.flush()
        save["patch_z64_map_path"] = np.asarray(os.path.abspath(z64_map_path))
        save["patch_z64_map_shape"] = np.asarray([P, H, H, 64], dtype=np.int32)
        save["patch_z64_map_dtype"] = np.asarray("float16")
    np.savez_compressed(out_npz, **save)

    table = os.path.join(args.out, "density_table.txt")
    with open(table, "w") as f:
        for p in range(P):
            f.write(f"{p} {coverage[p]:.6f}\n")
    print(f"\nwrote proposals -> {out_npz}", flush=True)
    print(f"wrote density table -> {table}", flush=True)
    print("next:", flush=True)
    subset = f" --max_patches {P}" if P < len(centers) else ""
    print(
        f"  python patchlocks/deploy/build_v2_region_prior.py --pack {args.pack}{subset} "
        f"--proposal_npz {out_npz} --table {table} --out {args.out}/v2_region_prior.npz",
        flush=True,
    )


if __name__ == "__main__":
    main()
