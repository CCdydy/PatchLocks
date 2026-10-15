"""E: test-time K=1 vs K=4 for the SAME v16 checkpoint (docs/EVALUATION.md).

K=1 protocol: build_env(views_override=1) -> EVERY condition from view 0 only
(fuser G_COND views=1, RAYC re-reduced over view 0, GLOB/RGBPROJ rebuilt from
view 0 — the all-view leak of --single_view is closed here), standardized with
the ckpt's saved 4-view TRAINING stats (cond_stats). Disclosed caveat: the UNet
and its cond stats were fit on 4-view fused conditions; K=1 is a test-time
distribution shift by construction (the fuser itself was view-dropout-trained).

Same protocol seeds as eval_3arm --mode val: patches env.vi[:80], texel seed
1000000+gidx, sampler seed=gidx, cfg=3.0, steps=40. K=4 numbers are joined
per-patch from the eval_3arm JSON (--k4_json); paired bootstrap on the deltas.

Run:
  python \
    patchlocks/experiments/rebuttal_2027/eval_k1.py \
    --k4_json patchlocks/experiments/rebuttal_2027/eval3arm_val80.json \
    --out patchlocks/experiments/rebuttal_2027/eval_k1_val80.json
"""
import os, sys, json, argparse
import numpy as np, torch

sys.path.insert(0, os.path.join(os.getcwd(), "patchlocks", "v14"))
sys.path.insert(0, os.getcwd())
from v14_common import build_env
from models.strand_codec import StrandCodec

ap = argparse.ArgumentParser()
ap.add_argument("--renders", default="patchlocks/datasets/v11_longfur/renders_mv")
ap.add_argument("--e1_dir", default="patchlocks/datasets/v16/e2a_cw15")
ap.add_argument("--codec_path", default="patchlocks/models/fur_strand_codec_v6upright.pt")
ap.add_argument("--k4_json", required=True, help="eval_3arm val JSON (supplies the K=4 arm per patch)")
ap.add_argument("--nsub", type=int, default=80)
ap.add_argument("--cfg", type=float, default=3.0)
ap.add_argument("--lk", type=int, default=32)
ap.add_argument("--nboot", type=int, default=10000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda"
ATTRS = ["arc", "curl", "clump", "straight", "lift_arc", "tip_spread"]

# K=1 env: view-0-only conditions, ckpt 4-view training stats (REQUIRED for this protocol)
ck = torch.load(f"{a.e1_dir}/v14_best.pt", map_location="cpu", weights_only=False)
cond_stats = {k: ck[k] for k in ["cond_mean", "cond_std", "glob_mean", "glob_std"]}
env = build_env(renders=a.renders, e1_dir=a.e1_dir, cond_stats=cond_stats,
                views_override=1, codec_path=a.codec_path, len_k=a.lk, verbose=True)
H = env.H
fetch = env.fetch

# local codec (strands-in measurement, byte-identical to eval_3arm)
_ck = torch.load(a.codec_path, weights_only=False)
_cn = {k: torch.tensor(v, device=dev, dtype=torch.float32) for k, v in _ck["fur_norm"].items()}
codec = StrandCodec(do_vae=False, decode_type="dir", scale_init=30.0,
                    nr_verts_per_strand=256, nr_values_to_decode=255,
                    dim_per_value_decoded=3).to(dev).eval()
codec.load_state_dict(_ck["state_dict"])
for p in codec.parameters(): p.requires_grad = False

@torch.no_grad()
def decode_rec(z_std, idxb, idxhw):
    z = (z_std[:, :64] * env.xs + env.xm).permute(0, 2, 3, 1).reshape(len(z_std), H * H, 64)[idxb, idxhw]
    pr = codec.decoder(z, None, _cn)
    rec = pr["strand_positions"]
    return rec - rec[:, :1]

def params_from_rec(rec, lk):
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
    seg = rec[:, 1:] - rec[:, :-1]
    segn = seg.norm(dim=2).clamp_min(1e-9)
    arc = segn.sum(1)
    u = seg / segn[..., None]
    cosd = (u[:, 1:] * u[:, :-1]).sum(-1).clamp(-1, 1)
    curv = torch.acos(cosd).mean(1)
    return arc.float().cpu().numpy(), curv.float().cpu().numpy()

def w1(u, v):
    u, v = np.sort(np.asarray(u)), np.sort(np.asarray(v))
    return float(np.abs(u - v).mean())

# K=4 arm joined from the eval_3arm dump
k4 = json.load(open(a.k4_json))
k4_by_dir = {e["dir"]: e for e in k4["per_patch"]}

per_patch = []
idx_list = env.vi[: a.nsub].tolist()
print(f"K=1 scoring {len(idx_list)} patches", flush=True)
with torch.no_grad():
    for row, gidx in enumerate(idx_list):
        dname = os.path.basename(env.dirs[gidx])
        if dname not in k4_by_dir:
            continue                      # keep the K1/K4 patch sets identical
        j = torch.tensor([gidx], device=env.idx_dev)
        mm = fetch(env.MASK, j)
        ib, ihw, kept = env.sample_texels(mm, a.lk, seed=1000000 + gidx)
        if ib is None or not bool(kept[0]):
            continue
        gc, gl = fetch(env.G_COND, j), fetch(env.GLOB, j)
        rc, rp, ry = fetch(env.RHOC, j).float(), fetch(env.RGBPROJ, j).float(), fetch(env.RAYC, j).float()
        rec_gt = decode_rec(fetch(env.Xn, j).float(), ib, ihw)
        zd = env.sample_z(gc, gl, rc, rp, ry, cfg=a.cfg, seed=gidx)
        rec_k1 = decode_rec(zd.float(), ib, ihw)
        e4 = k4_by_dir[dname]
        pg = params_from_rec(rec_gt, a.lk); p1 = params_from_rec(rec_k1, a.lk)
        arc_g, curv_g = strand_distrib(rec_gt); arc_1, curv_1 = strand_distrib(rec_k1)
        entry = {"dir": dname, "gidx": int(gidx),
                 "gt": {k: float(pg[k][0]) for k in ATTRS},
                 "v16k1": {k: float(p1[k][0]) for k in ATTRS},
                 "v16k4": {k: float(e4["v16"][k]) for k in ATTRS},
                 "gt_k4ref": {k: float(e4["gt"][k]) for k in ATTRS}}
        entry["v16k1"]["w1_arc"] = w1(arc_1, arc_g); entry["v16k1"]["w1_curv"] = w1(curv_1, curv_g)
        entry["v16k4"]["w1_arc"] = float(e4["v16"]["w1_arc"]); entry["v16k4"]["w1_curv"] = float(e4["v16"]["w1_curv"])
        per_patch.append(entry)
        if (row + 1) % 20 == 0: print(f"  {row+1}/{len(idx_list)}", flush=True)

# sanity: GT must be identical across the two envs (views do not touch Xn)
gt_drift = max(abs(e["gt"][k] - e["gt_k4ref"][k]) for e in per_patch for k in ATTRS)
print(f"GT drift K1-env vs K4-env: {gt_drift:.2e} (must be ~0)", flush=True)

def nmae(p, g):
    p, g = np.asarray(p), np.asarray(g)
    return float(np.abs(p - g).mean() / (g.std() + 1e-12))

def pcorr(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    if len(u) < 3 or u.std() < 1e-9 or v.std() < 1e-9: return 0.0
    return float(np.corrcoef(u, v)[0, 1])

G = {k: [e["gt"][k] for e in per_patch] for k in ATTRS}
agg = {}
for arm in ["v16k4", "v16k1"]:
    A = {k: [e[arm][k] for e in per_patch] for k in ATTRS}
    agg[arm] = {"nmae": {k: nmae(A[k], G[k]) for k in ATTRS},
                "corr": {k: pcorr(A[k], G[k]) for k in ATTRS},
                "std_ratio": {k: float(np.std(A[k]) / (np.std(G[k]) + 1e-12)) for k in ATTRS},
                "w1_arc_mean": float(np.mean([e[arm]["w1_arc"] for e in per_patch])),
                "w1_curv_mean": float(np.mean([e[arm]["w1_curv"] for e in per_patch]))}
    agg[arm]["mean_nmae"] = float(np.mean(list(agg[arm]["nmae"].values())))
    agg[arm]["mean_corr"] = float(np.mean(list(agg[arm]["corr"].values())))

# paired bootstrap: K1 minus K4 mean nMAE (positive = K1 worse)
n = len(per_patch)
rng = np.random.RandomState(0)
idxb = rng.randint(0, n, size=(a.nboot, n))
Ak = {arm: {k: np.array([e[arm][k] for e in per_patch]) for k in ATTRS} for arm in ["v16k4", "v16k1"]}
Gk = {k: np.array(G[k]) for k in ATTRS}
dm = []
for b in range(a.nboot):
    ii = idxb[b]
    d1 = np.mean([nmae(Ak["v16k1"][k][ii], Gk[k][ii]) for k in ATTRS])
    d4 = np.mean([nmae(Ak["v16k4"][k][ii], Gk[k][ii]) for k in ATTRS])
    dm.append(d1 - d4)
dm = np.asarray(dm)
delta_ci = [float(np.percentile(dm, q)) for q in (2.5, 50, 97.5)]

out = {"meta": {"renders": a.renders, "e1_dir": a.e1_dir, "n": n, "lk": a.lk, "cfg": a.cfg,
                "k4_json": a.k4_json, "views_override": 1, "cond_stats": "ckpt(4-view train)",
                "caveat": "UNet/cond-stats fit on 4-view fused conditions; K=1 is test-time shift",
                "attrs": ATTRS, "gt_drift": gt_drift},
       "aggregate": agg,
       "delta_k1_minus_k4_mean_nmae_ci": delta_ci,
       "per_patch": per_patch}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({arm: {"mean_nmae": round(agg[arm]["mean_nmae"], 4),
                        "mean_corr": round(agg[arm]["mean_corr"], 4),
                        "w1_curv": round(agg[arm]["w1_curv_mean"], 5)} for arm in agg}, indent=1))
print(f"delta (K1-K4) mean nMAE 95% CI: {delta_ci}  (positive = K1 worse)")
print(f"saved -> {a.out}")
