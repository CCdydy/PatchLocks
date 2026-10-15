"""3-arm fair evaluation (docs/EVALUATION.md): Diffusion(v16) / Det-z64 / Direct-7.

ONE env, ONE measurement implementation for all arms (decode_params formulas
replicated locally so raw strand arrays are accessible for distribution metrics
and so Direct-7's synthesized primitive is measured by byte-identical code).

Protocols:
  --mode val   : original renders_mv, seed-0 val split, first --nsub (80) dirs,
                 texel seed 1000000+gidx, diffusion seed=gidx cfg=3 steps=40
                 (exact eval_v16_vs_v10 protocol).
  --mode all   : score ALL dirs of --renders (test set B; pass --ckpt_cond_stats).

Per-patch values are dumped to JSON for bootstrap CI (C).

Run (original val):
  python \
    patchlocks/experiments/rebuttal_2027/eval_3arm.py --mode val \
    --out patchlocks/experiments/rebuttal_2027/eval3arm_val80.json
Run (test set B):
  ... --mode all --renders patchlocks/experiments/rebuttal_2027/testB_renders \
      --ckpt_cond_stats --out patchlocks/experiments/rebuttal_2027/eval3arm_testB.json
"""
import os, sys, json, argparse
import numpy as np, torch

sys.path.insert(0, os.path.join(os.getcwd(), "patchlocks", "v14"))
sys.path.insert(0, os.getcwd())
from v14_common import build_env, UNet, GLOBC
from models.strand_codec import StrandCodec

ap = argparse.ArgumentParser()
ap.add_argument("--renders", default="patchlocks/datasets/v11_longfur/renders_mv")
ap.add_argument("--e1_dir", default="patchlocks/datasets/v16/e2a_cw15")
ap.add_argument("--det_ckpt", default="patchlocks/experiments/rebuttal_2027/det_z64/det_z64_best.pt")
ap.add_argument("--d7_ckpt", default="patchlocks/experiments/rebuttal_2027/direct7/direct7_best.pt")
ap.add_argument("--codec_path", default="patchlocks/models/fur_strand_codec_v6upright.pt")
ap.add_argument("--mode", choices=["val", "all"], default="val")
ap.add_argument("--nsub", type=int, default=80)
ap.add_argument("--cfg", type=float, default=3.0)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--lk", type=int, default=32)
ap.add_argument("--ckpt_cond_stats", action="store_true",
                help="standardize conditions with the e1 ckpt's saved training stats (test set B)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda"
ATTRS = ["arc", "curl", "clump", "straight", "lift_arc", "tip_spread"]
RADIUS_RATIO_FIXED = 0.017   # paper radius policy; positions-only metrics unaffected

# ---------------- env (shared conditions for every arm) ----------------
cond_stats = None
if a.ckpt_cond_stats:
    ck = torch.load(f"{a.e1_dir}/v14_best.pt", map_location="cpu", weights_only=False)
    cond_stats = {k: ck[k] for k in ["cond_mean", "cond_std", "glob_mean", "glob_std"]}
    print("using ckpt training cond stats for normalization", flush=True)
env = build_env(renders=a.renders, e1_dir=a.e1_dir, cond_stats=cond_stats,
                codec_path=a.codec_path, len_k=a.lk, verbose=True)
H, XD = env.H, env.XD
fetch = env.fetch

# ---------------- local codec + EXACT decode_params replica ----------------
_ck = torch.load(a.codec_path, weights_only=False)
_cn = {k: torch.tensor(v, device=dev, dtype=torch.float32) for k, v in _ck["fur_norm"].items()}
codec = StrandCodec(do_vae=False, decode_type="dir", scale_init=30.0,
                    nr_verts_per_strand=256, nr_values_to_decode=255,
                    dim_per_value_decoded=3).to(dev).eval()
codec.load_state_dict(_ck["state_dict"])
for p in codec.parameters(): p.requires_grad = False

@torch.no_grad()
def decode_rec(z_std, idxb, idxhw):
    """standardized z64 -> root-centered strands (M,256,3). Replica of _decode_rec."""
    z = (z_std[:, :64] * env.xs + env.xm).permute(0, 2, 3, 1).reshape(len(z_std), H * H, 64)[idxb, idxhw]
    pr = codec.decoder(z, None, _cn)
    rec = pr["strand_positions"]
    return rec - rec[:, :1]

def params_from_rec(rec, lk):
    """EXACT replica of v14_common.decode_params formula body (strands-in)."""
    seg = (rec[:, 1:] - rec[:, :-1]).norm(dim=2); arc = seg.sum(1).clamp_min(1e-6)
    tip = rec[:, -1, :2]; chord2 = tip.norm(dim=1); chord3 = rec[:, -1].norm(dim=1); straight = chord3 / arc
    lift_arc = rec[..., 2].abs().max(1).values / arc
    ang = torch.atan2(tip[:, 1], tip[:, 0])
    yc = torch.sin(-ang)[:, None] * rec[..., 0] + torch.cos(-ang)[:, None] * rec[..., 1]
    curl = yc.abs().mean(1) / arc
    nv = rec.shape[0] // lk; R = lambda x: x.reshape(nv, lk)
    cm = R(torch.cos(ang)).mean(1); sm = R(torch.sin(ang)).mean(1)
    med = lambda x: R(x).median(1).values
    return dict(arc=med(arc), straight=med(straight), lift_arc=med(lift_arc), curl=med(curl),
                tip_spread=R(chord2).std(1, unbiased=False) / med(arc).clamp_min(1e-6),
                clump=(cm ** 2 + sm ** 2).clamp_min(1e-9).sqrt())

def strand_distrib(rec):
    """per-strand arc + curvature (mean turn angle) distributions for W1 metrics."""
    seg = rec[:, 1:] - rec[:, :-1]
    segn = seg.norm(dim=2).clamp_min(1e-9)
    arc = segn.sum(1)
    u = seg / segn[..., None]
    cosd = (u[:, 1:] * u[:, :-1]).sum(-1).clamp(-1, 1)
    curv = torch.acos(cosd).mean(1)          # mean turn angle per strand (rad)
    return arc.float().cpu().numpy(), curv.float().cpu().numpy()

def w1(u, v):
    u, v = np.sort(np.asarray(u)), np.sort(np.asarray(v))
    return float(np.abs(u - v).mean())

# ---------------- stroke synthesis (verbatim copy: build_canon_swatch.py:44) ----------------
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

# ---------------- load arms ----------------
ii, jj = np.meshgrid(np.arange(H), np.arange(H))
ug, vg = (ii + 0.5) / H, (jj + 0.5) / H

dck = torch.load(a.det_ckpt, map_location="cpu", weights_only=False)
det = UNet(XD, ug, vg, int(dck["unet_dim"]), cond_ch=env.G_COND.shape[1],
           use_attn=True, cond_zbase=False, cond_raydir=True).to(dev)
det.load_state_dict(dck["net"]); det.eval()
print(f"Det-z64 loaded (it {dck['iter']}, sel corr {dck['best_corr']:.4f})", flush=True)

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
assert hck["attrs"] == ATTRS
print(f"Direct-7 loaded (it {hck['iter']}, sel corr {hck['best_corr']:.4f})", flush=True)

# ---------------- patch list ----------------
if a.mode == "val":
    idx_list = env.vi[: a.nsub].tolist()
else:
    idx_list = list(range(len(env.dirs)))
print(f"scoring {len(idx_list)} patches (mode={a.mode})", flush=True)

per_patch = []
with torch.no_grad():
    for row, gidx in enumerate(idx_list):
        j = torch.tensor([gidx], device=env.idx_dev)
        mm = fetch(env.MASK, j)
        ib, ihw, kept = env.sample_texels(mm, a.lk, seed=1000000 + gidx)
        if ib is None or not bool(kept[0]):
            continue
        gc, gl = fetch(env.G_COND, j), fetch(env.GLOB, j)
        rc, rp, ry = fetch(env.RHOC, j).float(), fetch(env.RGBPROJ, j).float(), fetch(env.RAYC, j).float()
        rec = {}

        # GT
        zg = fetch(env.Xn, j).float()
        rec["gt"] = decode_rec(zg, ib, ihw)

        # Diffusion (paper protocol)
        zd = env.sample_z(gc, gl, rc, rp, ry, cfg=a.cfg, seed=gidx)
        rec["v16"] = decode_rec(zd.float(), ib, ihw)

        # Det-z64
        zdet = det(torch.zeros(1, XD, H, H, device=dev), torch.zeros(1, device=dev),
                   gc.float(), gl.float(), rc, rp, rayc=ry)
        rec["det"] = decode_rec(zdet, ib, ihw)

        # Direct-7 (regressed params + synthesized primitive)
        preg = (d7(gc, gl) * d7_sstd + d7_smean)[0]
        d7_params = {k: float(preg[c]) for c, k in enumerate(ATTRS)}
        rng = np.random.RandomState(90000 + gidx)
        off, _ = _stroke_offsets_from_stats(a.lk, 256, d7_params["arc"], d7_params["curl"],
                                            d7_params["lift_arc"], RADIUS_RATIO_FIXED, rng)
        rec["d7s"] = torch.tensor(off, device=dev)

        entry = {"dir": os.path.basename(env.dirs[gidx]), "gidx": int(gidx),
                 "coat": env.coat_of(os.path.basename(env.dirs[gidx]))}
        arc_gt, curv_gt = strand_distrib(rec["gt"])
        for arm in ["gt", "v16", "det", "d7s"]:
            pp = params_from_rec(rec[arm], a.lk)
            entry[arm] = {k: float(pp[k][0]) for k in ATTRS}
            if arm != "gt":
                arc_a, curv_a = strand_distrib(rec[arm])
                entry[arm]["w1_arc"] = w1(arc_a, arc_gt)
                entry[arm]["w1_curv"] = w1(curv_a, curv_gt)
        entry["d7_reg"] = d7_params

        # v16s: v16's MEASURED scalars through the SAME synthesizer (deployment-parity arm;
        # same rng convention as the val80 post-hoc addition: RandomState(95000+gidx))
        rng2 = np.random.RandomState(95000 + gidx)
        off2, _ = _stroke_offsets_from_stats(a.lk, 256, entry["v16"]["arc"], entry["v16"]["curl"],
                                             entry["v16"]["lift_arc"], RADIUS_RATIO_FIXED, rng2)
        rec_v16s = torch.tensor(off2, device=dev)
        pp2 = params_from_rec(rec_v16s, a.lk)
        entry["v16s"] = {k: float(pp2[k][0]) for k in ATTRS}
        arc_s, curv_s = strand_distrib(rec_v16s)
        entry["v16s"]["w1_arc"] = w1(arc_s, arc_gt)
        entry["v16s"]["w1_curv"] = w1(curv_s, curv_gt)
        per_patch.append(entry)
        if (row + 1) % 20 == 0:
            print(f"  {row+1}/{len(idx_list)}", flush=True)

# ---------------- aggregates ----------------
def nmae(pred, gt):
    pred, gt = np.asarray(pred), np.asarray(gt)
    s = gt.std()
    return float(np.abs(pred - gt).mean() / (s + 1e-12))

def pcorr(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    if len(u) < 3 or u.std() < 1e-9 or v.std() < 1e-9: return 0.0
    return float(np.corrcoef(u, v)[0, 1])

G = {k: [e["gt"][k] for e in per_patch] for k in ATTRS}
agg = {}
for arm in ["v16", "det", "d7s", "d7_reg", "v16s"]:
    A = {k: [e[arm][k] for e in per_patch] for k in ATTRS}
    agg[arm] = {
        "nmae": {k: nmae(A[k], G[k]) for k in ATTRS},
        "corr": {k: pcorr(A[k], G[k]) for k in ATTRS},
        "std_ratio": {k: float(np.std(A[k]) / (np.std(G[k]) + 1e-12)) for k in ATTRS},
    }
    agg[arm]["mean_nmae"] = float(np.mean(list(agg[arm]["nmae"].values())))
    agg[arm]["mean_corr"] = float(np.mean(list(agg[arm]["corr"].values())))
    if arm != "d7_reg":
        agg[arm]["w1_arc_mean"] = float(np.mean([e[arm]["w1_arc"] for e in per_patch]))
        agg[arm]["w1_curv_mean"] = float(np.mean([e[arm]["w1_curv"] for e in per_patch]))

out = {"meta": {"mode": a.mode, "renders": a.renders, "n": len(per_patch), "lk": a.lk,
                "cfg": a.cfg, "steps": a.steps, "e1_dir": a.e1_dir,
                "det_ckpt": a.det_ckpt, "d7_ckpt": a.d7_ckpt,
                "ckpt_cond_stats": bool(a.ckpt_cond_stats), "attrs": ATTRS},
       "aggregate": agg, "per_patch": per_patch}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({arm: {"mean_nmae": round(agg[arm]["mean_nmae"], 4),
                        "mean_corr": round(agg[arm]["mean_corr"], 4)} for arm in agg}, indent=1))
print(f"saved -> {a.out}")
