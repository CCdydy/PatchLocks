"""Visual grid (CONTRACT metric level 3): Input | GT | Direct-7 | Det-z64 | Diffusion.

4 coat archetypes from the val80 set, ONE camera (view 0) + neutral material for
every 3D panel so the comparison is geometry/character only. All predicted arms
share the SAME rho-weighted roots (GT-rho conditioning, protocol-consistent) and
the SAME per-root radius map; colors neutral tan everywhere (GT included).

Direct-7 disclosure (CONTRACT clause 4): its strands are synthesized by
_stroke_offsets_from_stats in the canonical +x sweep — direction is external to
all arms' claims; the panel shows what 7 scalars + the fixed synthesizer can
express (that IS the experiment).

Run (AFTER eval_k1 frees the GPU):
  cd /home/rog/Desktop/diffusionF && /home/rog/miniconda3/envs/difflocks/bin/python \
    patchlocks/experiments/rebuttal_2027/grid_dump_3arm.py \
    --out patchlocks/experiments/rebuttal_2027/grid5
"""
import os, sys, json, argparse, subprocess
import numpy as np, torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "patchlocks/v14")
sys.path.insert(0, "patchlocks/v16")
sys.path.insert(0, "patchlocks/data_gen")
from v14_common import build_env, UNet
from infer_v16_patch import (sample_z_field, decode_dense, resample_arc,
                             distribute_roots, NEUTRAL_TAN)
from field_to_strands import _bilinear

ap = argparse.ArgumentParser()
ap.add_argument("--renders", default="patchlocks/datasets/v11_longfur/renders_mv")
ap.add_argument("--e1_dir", default="patchlocks/datasets/v16/e2a_cw15")
ap.add_argument("--det_ckpt", default="patchlocks/experiments/rebuttal_2027/det_z64/det_z64_best.pt")
ap.add_argument("--d7_ckpt", default="patchlocks/experiments/rebuttal_2027/direct7/direct7_best.pt")
ap.add_argument("--codec_path", default="patchlocks/models/fur_strand_codec_v6upright.pt")
ap.add_argument("--val_json", default="patchlocks/experiments/rebuttal_2027/eval3arm_val80.json")
ap.add_argument("--coats", default="ultra_short,wavy,curly,long_silky")
ap.add_argument("--view", type=int, default=0)
ap.add_argument("--samples", type=int, default=64)
ap.add_argument("--cfg", type=float, default=3.0)
ap.add_argument("--blender", default="patchlocks/third_party/blender-4.5.10-linux-x64/blender")
ap.add_argument("--out", required=True)
ap.add_argument("--skip_render", action="store_true", help="dump npz only (render later)")
ap.add_argument("--ckpt_cond_stats", action="store_true",
                help="standardize conditions with the ckpt training stats (use for foreign root B)")
a = ap.parse_args()
dev = "cuda"
ATTRS = ["arc", "curl", "clump", "straight", "lift_arc", "tip_spread"]
RADIUS_RATIO_FIXED = 0.017
os.makedirs(a.out, exist_ok=True)

# pick: first val80 patch of each requested coat (deterministic)
val = json.load(open(a.val_json))
want = [c.strip() for c in a.coats.split(",") if c.strip()]
picks = {}
for e in val["per_patch"]:
    if e["coat"] in want and e["coat"] not in picks:
        picks[e["coat"]] = e
missing = [c for c in want if c not in picks]
if missing:
    raise SystemExit(f"coats not in val80: {missing}")
print("picked:", {c: picks[c]["dir"] for c in want}, flush=True)

cond_stats = None
if a.ckpt_cond_stats:
    _ck = torch.load(f"{a.e1_dir}/v14_best.pt", map_location="cpu", weights_only=False)
    cond_stats = {k: _ck[k] for k in ["cond_mean", "cond_std", "glob_mean", "glob_std"]}
    print("using ckpt training cond stats (foreign root)", flush=True)
env = build_env(renders=a.renders, e1_dir=a.e1_dir, cond_stats=cond_stats,
                codec_path=a.codec_path, verbose=True)
H = env.H

# Det-z64 arm
ii, jj = np.meshgrid(np.arange(H), np.arange(H))
ug, vg = (ii + 0.5) / H, (jj + 0.5) / H
dck = torch.load(a.det_ckpt, map_location="cpu", weights_only=False)
det = UNet(env.XD, ug, vg, int(dck["unet_dim"]), cond_ch=env.G_COND.shape[1],
           use_attn=True, cond_zbase=False, cond_raydir=True).to(dev)
det.load_state_dict(dck["net"]); det.eval()

# Direct-7 arm (head class copied from train_direct7 — import would execute training)
import torch.nn as nn
class Direct7Head(nn.Module):
    def __init__(s, cond_ch, glob_ch, w=128, n_out=6):
        super().__init__()
        s.cnn = nn.Sequential(
            nn.Conv2d(cond_ch, w, 3, 1, 1), nn.GroupNorm(8, w), nn.SiLU(),
            nn.Conv2d(w, w, 3, 2, 1), nn.GroupNorm(8, w), nn.SiLU(),
            nn.Conv2d(w, 2 * w, 3, 2, 1), nn.GroupNorm(8, 2 * w), nn.SiLU(),
            nn.Conv2d(2 * w, 2 * w, 3, 1, 1), nn.GroupNorm(8, 2 * w), nn.SiLU())
        s.gproj = nn.Sequential(nn.Linear(glob_ch, w), nn.SiLU())
        s.mlp = nn.Sequential(nn.Linear(2 * w + w, 2 * w), nn.SiLU(),
                              nn.Linear(2 * w, 2 * w), nn.SiLU())
        s.head = nn.Linear(2 * w, n_out)
    def forward(s, gc, gl):
        f = s.cnn(gc.float()).mean((2, 3))
        return s.head(s.mlp(torch.cat([f, s.gproj(gl.float())], 1)))
hck = torch.load(a.d7_ckpt, map_location="cpu", weights_only=False)
d7 = Direct7Head(hck["cond_ch"], hck["glob_ch"], hck["head_dim"], n_out=len(hck["attrs"])).to(dev)
d7.load_state_dict(hck["head"]); d7.eval()
d7_smean = hck["smean"].to(dev); d7_sstd = hck["sstd"].to(dev)

def _stroke_offsets_from_stats(n, points, arc_med, curl_frac, lift_frac, radius_ratio, rng):
    n = int(n); points = int(points)
    t = np.linspace(0.0, 1.0, points, dtype=np.float32)[None, :, None]
    arc_med = max(float(arc_med), 1e-6)
    curl_frac = max(float(curl_frac), 0.0)
    lift_frac = max(float(lift_frac), 0.0)
    radius_ratio = max(float(radius_ratio), 1e-6)
    arc_i = arc_med * np.exp(rng.normal(0.0, 0.055, size=n)).astype(np.float32)
    arc_i = np.clip(arc_i, 0.78 * arc_med, 1.22 * arc_med)
    curl_i = curl_frac * np.exp(rng.normal(0.0, 0.18, size=n)).astype(np.float32)
    lift_i = lift_frac * np.exp(rng.normal(0.0, 0.16, size=n)).astype(np.float32)
    y_amp = (curl_i * arc_i / (2.0 / np.pi + 1e-6)).astype(np.float32)
    z_amp = (lift_i * arc_i).astype(np.float32)
    sign = rng.choice(np.array([-1.0, 1.0], np.float32), size=n).astype(np.float32)
    phase = rng.uniform(-0.18, 0.18, size=n).astype(np.float32)
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

def write_npz(path, offs, roots, uv, radius, size, L):
    """root-centered offsets (M,P,3) -> arc-resampled L, placed at roots, NEUTRAL color npz.
    resample_arc is numpy-only -> force CPU numpy first."""
    if torch.is_tensor(offs):
        offs = offs.detach().float().cpu().numpy()
    outp = np.asarray(resample_arc(np.asarray(offs, np.float32), L), np.float32)
    strands = (outp - outp[:, :1] + roots[:, None]).astype(np.float32)
    strands[..., 2] = np.maximum(strands[..., 2], 0.0)
    strands[:, 0] = roots
    M = len(strands)
    color = np.tile(NEUTRAL_TAN, (M, 1)).astype(np.float32)
    np.savez(path, strands=strands, root_uv=uv.astype(np.float32), color=color,
             color_tip=np.clip(color * 0.78 + 0.22, 0, 1).astype(np.float32),
             radius=radius.astype(np.float32), size=np.float32(size))
    return M

rows = []
with torch.no_grad():
    for coat in want:
        e = picks[coat]; gidx = e["gidx"]; d = env.dirs[gidx]
        od = os.path.join(a.out, coat); os.makedirs(od, exist_ok=True)
        gt = np.load(f"{d}/patch.npz", allow_pickle=True)
        L = int(gt["strands"].shape[1]); size = float(gt["size"])
        fg = np.load(f"{d}/field_gt_{H}.npz")
        rho_map = fg["lambda_map"].astype(np.float32)
        rm = fg["radius_mu"].astype(np.float32) if "radius_mu" in fg.files else None
        radius_map = (rm[..., 0] if rm is not None and rm.ndim == 3 else rm) if rm is not None \
            else np.full((H, H), 0.0007359216, np.float32)
        uv, roots = distribute_roots(rho_map, size, H, seed=gidx)
        radius = _bilinear(radius_map[..., None], uv)[:, 0].astype(np.float32)
        M = len(uv)

        j = torch.tensor([gidx], device=env.idx_dev)
        gc, gl = env.fetch(env.G_COND, j), env.fetch(env.GLOB, j)
        rc, rp, ry = env.fetch(env.RHOC, j).float(), env.fetch(env.RGBPROJ, j).float(), env.fetch(env.RAYC, j).float()

        # v16
        z_real = sample_z_field(env, gidx, a.cfg, gidx)
        rec = decode_dense(env, z_real, uv, a.codec_path)
        write_npz(f"{od}/v16.npz", rec, roots, uv, radius, size, L)
        # det
        zdet = det(torch.zeros(1, env.XD, H, H, device=dev), torch.zeros(1, device=dev),
                   gc.float(), gl.float(), rc, rp, rayc=ry)
        zdet_real = (zdet[:, :64] * env.xs + env.xm)[0].permute(1, 2, 0).contiguous().cpu().numpy()
        rec = decode_dense(env, zdet_real, uv, a.codec_path)
        write_npz(f"{od}/det.npz", rec, roots, uv, radius, size, L)
        # d7s (canonical +x sweep, disclosed)
        preg = (d7(gc, gl) * d7_sstd + d7_smean)[0]
        pr = {k: float(preg[c]) for c, k in enumerate(ATTRS)}
        rng = np.random.RandomState(90000 + gidx)
        off, _ = _stroke_offsets_from_stats(M, 256, pr["arc"], pr["curl"], pr["lift_arc"],
                                            RADIUS_RATIO_FIXED, rng)
        write_npz(f"{od}/d7s.npz", off, roots, uv, radius, size, L)
        # GT: neutralized copy
        gs = gt["strands"].astype(np.float32); gn = len(gs)
        gcol = np.tile(NEUTRAL_TAN, (gn, 1)).astype(np.float32)
        grad = gt["radius"].astype(np.float32).reshape(-1)
        if len(grad) == 1: grad = np.full(gn, float(grad[0]), np.float32)
        np.savez(f"{od}/gt.npz", strands=gs, root_uv=gt["root_uv"].astype(np.float32),
                 color=gcol, color_tip=np.clip(gcol * 0.78 + 0.22, 0, 1).astype(np.float32),
                 radius=grad, size=np.float32(size))
        rows.append({"coat": coat, "dir": d, "gidx": gidx, "M": M, "d7_params": pr})
        print(f"[{coat}] {os.path.basename(d)}: M={M} dumped (v16/det/d7s/gt)", flush=True)

del env, det, d7
torch.cuda.empty_cache()
json.dump(rows, open(f"{a.out}/picks.json", "w"), indent=1)

if a.skip_render:
    print("npz dumped; --skip_render set. done."); sys.exit(0)

# Blender renders: one call per npz, same camera/view/lighting
for r in rows:
    od = os.path.join(a.out, r["coat"])
    for arm in ["gt", "d7s", "det", "v16"]:
        cmd = [a.blender, "-b", "--factory-startup",
               "--python", "patchlocks/qa/render_patch_samecam.py", "--",
               "--patch", f"{od}/{arm}.npz", "--cameras", f"{r['dir']}/cameras.json",
               "--views", str(a.view), "--out", f"{od}/render_{arm}",
               "--samples", str(a.samples), "--light_scale", "0.6", "--view_transform", "AgX"]
        print("RENDER", r["coat"], arm, flush=True)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# compose grid: rows = coats, cols = Input | GT | Direct-7 | Det-z64 | Diffusion
from PIL import Image, ImageDraw, ImageFont
def cell(p, s=384):
    im = Image.open(p).convert("RGB")
    return im.resize((s, s), Image.LANCZOS)
S = 384; PAD = 6; LBL = 34
cols = ["Input", "GT", "Direct-7", "Det-z64", "Diffusion (ours)"]
W = 5 * S + 6 * PAD; Hh = len(rows) * S + (len(rows) + 1) * PAD + LBL
grid = Image.new("RGB", (W, Hh), (255, 255, 255))
dr = ImageDraw.Draw(grid)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 22)
except Exception:
    font = ImageFont.load_default()
for ci, name in enumerate(cols):
    x = PAD + ci * (S + PAD)
    tw = dr.textlength(name, font=font)
    dr.text((x + (S - tw) / 2, 6), name, fill=(20, 20, 20), font=font)
for ri, r in enumerate(rows):
    od = os.path.join(a.out, r["coat"])
    y = LBL + PAD + ri * (S + PAD)
    vfile = f"rgb_{a.view}.png"
    panels = [f"{r['dir']}/{vfile}"]
    for arm in ["gt", "d7s", "det", "v16"]:
        rd = f"{od}/render_{arm}"
        pngs = sorted([f for f in os.listdir(rd) if f.endswith(".png")])
        panels.append(os.path.join(rd, pngs[0]))
    for ci, p in enumerate(panels):
        grid.paste(cell(p, S), (PAD + ci * (S + PAD), y))
grid.save(f"{a.out}/grid5.png")
print(f"saved -> {a.out}/grid5.png")
