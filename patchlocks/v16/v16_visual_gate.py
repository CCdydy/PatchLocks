"""Build a BLENDER-rendered visual gate sheet for the v16 (V14-family) paper diffusion.

Rows are held-out single-view samples (seed-0 15% split, IDENTICAL to v14_common.build_env's
PATCH-LEVEL split). Columns are:

  input RGB | GT 3D render | v16 predicted 3D render

GT and v16 are rendered with the SAME camera + deterministic lighting via the existing
patchlocks/qa/render_patch_samecam.py Blender renderer, so geometry/character differences are
directly visible. This mirrors patchlocks/qa/v10_visual_gate.py but swaps the inference call to
patchlocks/v16/infer_v16_patch.py (the v16 full-mode CFG diffusion, w=3).

HONEST CAVEAT (rendered fur DIRECTION): the v16 diffusion does NOT set ABSOLUTE strand direction
(z64 carries the codec-CANONICAL in-plane direction). The predicted column therefore shows
GEOMETRY / CHARACTER (length, lift, curl, clump, density, color from the back-projected photo),
NOT image-accurate absolute direction. The GT column is the real ground-truth groom.

Run (repo root, difflocks env):
  /home/rog/miniconda3/envs/difflocks/bin/python patchlocks/v16/v16_visual_gate.py \
    --model patchlocks/datasets/v16/e2a_cw15/v14_best.pt \
    --renders patchlocks/datasets/v11_longfur/renders_mv \
    --out patchlocks/datasets/v16/v16_visual_gate \
    --count 8
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw
import torch

# build the v16 conditioning env ONCE and reuse it across ALL picked patches (the per-patch
# subprocess env rebuild -- DINO + 1600-patch condition precompute -- was the ~3min/patch bottleneck).
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "patchlocks/v16")
from infer_v16_patch import build_v16_env, infer_patch_with_env  # noqa: E402


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="patchlocks/datasets/v16/e2a_cw15/v14_best.pt",
                help="v16 ckpt (v14_best.pt) or its dir; b3_fuser.pt must sit alongside")
ap.add_argument("--renders", default="patchlocks/datasets/v11_longfur/renders_mv")
ap.add_argument("--out", default="patchlocks/datasets/v16/v16_visual_gate")
ap.add_argument("--count", type=int, default=8)
ap.add_argument("--view", type=int, default=0)
ap.add_argument("--cfg", type=float, default=3.0)
ap.add_argument("--seed", type=int, default=-1, help="-1 => per-patch global index (matches eval_e1)")
ap.add_argument("--render_samples", type=int, default=48)
ap.add_argument("--blender", default="patchlocks/third_party/blender-4.5.10-linux-x64/blender")
ap.add_argument("--python", default="")
ap.add_argument("--latent_name", default="target_latent64_v6codec.npz")
ap.add_argument("--pose_name", default="v13_pose_64.npz")
ap.add_argument("--field_name", default="field_gt_64.npz")
a = ap.parse_args()

# resolve model dir + ckpt presence
model_arg = a.model
model_dir = Path(model_arg).parent if model_arg.endswith(".pt") else Path(model_arg)
ckpt = Path(model_arg) if model_arg.endswith(".pt") else (model_dir / "v14_best.pt")
if not ckpt.exists():
    raise SystemExit(f"missing v16 ckpt {ckpt}")
if not (model_dir / "b3_fuser.pt").exists():
    raise SystemExit(f"missing {model_dir/'b3_fuser.pt'} (build_env needs the fuser alongside the ckpt)")

py = a.python or sys.executable
blender = a.blender
if not Path(blender).exists():
    blender = shutil.which("blender") or blender

# held-out selection: replicate v14_common.build_env's PATCH-LEVEL seed-0 15% split EXACTLY
# (dirs filtered the same way: latent + field + cameras + pose present), so the sheet shows the
# real v16 held-out set. NB: build_env filters on field_gt_{H}.npz; H=64 here.
dirs = sorted(
    d for d in Path(a.renders).iterdir()
    if d.is_dir()
    and (d / "cameras.json").exists()
    and (d / "patch.npz").exists()
    and (d / a.latent_name).exists()
    and (d / a.field_name).exists()
    and (d / a.pose_name).exists()
    and (d / f"rgb_{a.view}.png").exists()
)
if not dirs:
    raise SystemExit(f"no render samples found under {a.renders} (need latent+field+pose+cameras+rgb)")

# PATCH-LEVEL split identical to v14_common.build_env (group by basename before _v<k>, seed-0)
g = torch.Generator().manual_seed(0)
pid = [d.name.rsplit("_v", 1)[0] for d in dirs]
uniq = sorted(set(pid))
pp = torch.randperm(len(uniq), generator=g).tolist()
valset = set(uniq[pp[i]] for i in range(max(1, int(len(uniq) * 0.15))))
val_dirs = [d for d, p in zip(dirs, pid) if p in valset]

# pick a DIVERSE subset by coat token (one per coat first, then top up)
def coat_of(name):
    n = name.lower()
    for c in ("ultra_short", "fluffy", "curly", "wavy", "wiry", "long", "short"):
        if c in n:
            return c
    return "short"

by_coat = {}
for d in val_dirs:
    by_coat.setdefault(coat_of(d.name), []).append(d)
picks = []
for c in ["ultra_short", "short", "wavy", "wiry", "curly", "fluffy", "long"]:
    if by_coat.get(c):
        picks.append(by_coat[c][0])
if len(picks) < a.count:
    pool = [d for c in by_coat for d in by_coat[c][1:]]
    for d in pool:
        if len(picks) >= a.count:
            break
        if d not in picks:
            picks.append(d)
val = picks[:a.count]
if not val:
    raise SystemExit("no held-out patches selected")
print(f"held-out picks ({len(val)}):", [d.name for d in val], flush=True)

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
rows = []

# ---- BUILD THE V16 ENV ONCE (shared across all picked patches) ----
t_env0 = time.time()
ENV = build_v16_env(model=str(ckpt), renders=a.renders, latent_name=a.latent_name, verbose=True)
print(f"[gate] shared v16 env built once in {time.time()-t_env0:.1f}s; reusing across {len(val)} patches", flush=True)


def run(cmd):
    print(" ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def load_tile(path, label, size=320):
    im = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    dr.rectangle((0, 0, size, 25), fill=(0, 0, 0))
    dr.text((6, 6), label, fill=(255, 255, 255))
    return im


for ri, d in enumerate(val):
    name = d.name
    od = out / name
    pred_dir = od / "v16"
    gt_dir = od / "gt_render"
    pr_dir = od / "v16_render"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    pr_dir.mkdir(parents=True, exist_ok=True)

    # per-patch fast path: sample + write the DENSE photo-colored patch.npz on the SHARED env (no rebuild)
    t_p0 = time.time()
    _, n_pred = infer_patch_with_env(ENV, str(d), str(pred_dir),
                                     cfg=a.cfg, seed=a.seed, color_source="photo",
                                     rho_source="field_gt")
    print(f"[gate] {name}: dense infer {n_pred} strands in {time.time()-t_p0:.1f}s (env reused)", flush=True)

    render_common = [
        blender, "-b", "--factory-startup",
        "--python", "patchlocks/qa/render_patch_samecam.py",
        "--",
        "--cameras", str(d / "cameras.json"),
        "--views", str(a.view),
        "--samples", str(a.render_samples),
        "--light_scale", "0.6",
        "--view_transform", "AgX",
    ]
    run(render_common + ["--patch", str(d / "patch.npz"), "--out", str(gt_dir)])
    run(render_common + ["--patch", str(pred_dir / "patch.npz"), "--out", str(pr_dir)])

    row = Image.new("RGB", (320 * 3, 320), "white")
    tiles = [
        load_tile(d / f"rgb_{a.view}.png", f"input {name}"),
        load_tile(gt_dir / f"rgb_{a.view}.png", "GT 3D"),
        load_tile(pr_dir / f"rgb_{a.view}.png", "v16 generated 3D"),
    ]
    for ci, tile in enumerate(tiles):
        row.paste(tile, (ci * 320, 0))
    rows.append(row)

sheet = Image.new("RGB", (320 * 3, 320 * len(rows)), "white")
for i, row in enumerate(rows):
    sheet.paste(row, (0, i * 320))
sheet_path = out / "v16_visual_gate.png"
sheet.save(sheet_path)
print(f"wrote {sheet_path}", flush=True)
