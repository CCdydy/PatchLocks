"""Arm 2 (Det-z64): deterministic z64 regressor for the 3-arm baseline (docs/EVALUATION.md).

Same UNet as the diffusion arm (v14_common.UNet, use_attn=True, dim matched to the
diffusion ckpt), same frozen conditions (build_env on the SAME e1_dir as the paper
v16 => byte-identical fuser + xm/xs), but zero-latent/zero-noise forward (the
train_v14 `reg` calling convention) and a plain masked L2 to the standardized
target latent Xn. Fairness clauses: cond_drop=0, NO decoded-parameter losses,
EMA 0.999 for symmetry with the diffusion arm's EMA checkpoint.

Selection: held-out (seed-0 val split) mean Pearson corr over the 6 eval ATTRS,
measured through the SAME env.decode_params machinery. Test set B is never touched.

Smoke:
  python \
    patchlocks/experiments/rebuttal_2027/train_det_z64.py --max_dirs 80 --iters 60 \
    --eval_every 30 --bs 8 --out /tmp/detz_smoke
Full:
  ... train_det_z64.py --out patchlocks/experiments/rebuttal_2027/det_z64 \
      --iters 20000 --bs 16 --eval_every 1000 --early_stop 6
"""
import os, sys, argparse
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, os.path.join(os.getcwd(), "patchlocks", "v14"))
from v14_common import build_env, UNet

ap = argparse.ArgumentParser()
ap.add_argument("--renders", default="patchlocks/datasets/v11_longfur/renders_mv")
ap.add_argument("--e1_dir", default="patchlocks/datasets/v16/e2a_cw15",
                help="paper v16 dir: supplies the SAME frozen fuser + xm/xs as the diffusion arm")
ap.add_argument("--out", default="patchlocks/experiments/rebuttal_2027/det_z64")
ap.add_argument("--iters", type=int, default=20000)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--ema", type=float, default=0.999)
ap.add_argument("--unet_dim", type=int, default=256)
ap.add_argument("--eval_every", type=int, default=1000)
ap.add_argument("--eval_n", type=int, default=120, help="val patches used for selection")
ap.add_argument("--early_stop", type=int, default=6)
ap.add_argument("--max_dirs", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
ATTRS = ["arc", "curl", "clump", "straight", "lift_arc", "tip_spread"]

env = build_env(renders=a.renders, e1_dir=a.e1_dir, max_dirs=a.max_dirs, len_k=32, verbose=True)
H, XD = env.H, env.XD
fetch, ti, vi = env.fetch, env.ti, env.vi
print(f"train {len(ti)} / val {len(vi)} | H={H} XD={XD}", flush=True)

torch.manual_seed(a.seed)
ii, jj = np.meshgrid(np.arange(H), np.arange(H))
ug, vg = (ii + 0.5) / H, (jj + 0.5) / H
net = UNet(XD, ug, vg, a.unet_dim, cond_ch=env.G_COND.shape[1], use_attn=True,
           cond_zbase=False, cond_raydir=True).to(dev)
print(f"Det-z64 UNet params: {sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)
ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(a.iters, 1))

def batch_conds(bi):
    gc = fetch(env.G_COND, bi).float()
    gl = fetch(env.GLOB, bi).float()
    rc = fetch(env.RHOC, bi).float()
    rp = fetch(env.RGBPROJ, bi).float()
    ry = fetch(env.RAYC, bi).float()
    x0 = fetch(env.Xn, bi).float()
    m = fetch(env.MASK, bi).float()
    if m.dim() == 3: m = m[:, None]
    return gc, gl, rc, rp, ry, x0, m

def forward_det(model, bi):
    gc, gl, rc, rp, ry, x0, m = batch_conds(bi)
    z = torch.zeros(len(bi), XD, H, H, device=dev)
    cn = torch.zeros(len(bi), device=dev)
    pred = model(z, cn, gc, gl, rc, rp, rayc=ry)
    return pred, x0, m

def _pcorr(u, v):
    u, v = u.float(), v.float()
    if u.numel() < 3 or u.std(unbiased=False) < 1e-6 or v.std(unbiased=False) < 1e-6: return 0.0
    um, vm = u - u.mean(), v - v.mean()
    return float((um * vm).mean() / (um.std(unbiased=False) * vm.std(unbiased=False) + 1e-12))

@torch.no_grad()
def evaluate(use_ema=True):
    if use_ema:
        bak = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(ema)
    net.eval()
    sel = vi[: a.eval_n]
    P = {k: [] for k in ATTRS}; G = {k: [] for k in ATTRS}
    for s in range(0, len(sel), 16):
        jj_ = sel[s:s + 16]
        pred, x0, m = forward_det(net, jj_)
        mm = fetch(env.MASK, jj_)
        ib, ihw, kept = env.sample_texels(mm, 32, seed=555000 + s)   # selection-only texel seed
        if ib is None: continue
        dp = env.decode_params(pred, ib, ihw, 32)
        dg = env.decode_params(x0, ib, ihw, 32)
        for k in ATTRS:
            P[k].append(dp[k].float().cpu()); G[k].append(dg[k].float().cpu())
    corrs = {}
    for k in ATTRS:
        if P[k]:
            corrs[k] = _pcorr(torch.cat(P[k]), torch.cat(G[k]))
        else:
            corrs[k] = 0.0
    net.train()
    if use_ema: net.load_state_dict(bak)
    return corrs

def save_ckpt(path, it, best):
    tmp = path + ".tmp"
    torch.save({"net": ema, "raw_net": net.state_dict(), "args": vars(a), "iter": int(it),
                "unet_dim": a.unet_dim, "attn": True, "xd": XD,
                "xm": env.xm.cpu(), "xs": env.xs.cpu(), "best_corr": float(best),
                "det_z64": True}, tmp)
    os.replace(tmp, path)

best, best_it, noimp = -1e9, 0, 0
print("train Det-z64 ...", flush=True)
net.train()
for it in range(1, a.iters + 1):
    bi = ti[torch.randint(0, len(ti), (a.bs,), device=env.idx_dev)]
    pred, x0, m = forward_det(net, bi)
    loss = ((pred - x0) ** 2 * m).sum() / (m.sum() * XD + 1e-8)
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        for k, v in net.state_dict().items():
            if v.dtype.is_floating_point: ema[k].mul_(a.ema).add_(v.detach(), alpha=1 - a.ema)
            else: ema[k].copy_(v)
    if it % 200 == 0:
        print(f"  it {it} | L2 {loss.item():.4f} | lr {sch.get_last_lr()[0]:.2e}", flush=True)
    if a.eval_every > 0 and it % a.eval_every == 0:
        corrs = evaluate()
        mc = float(np.mean(list(corrs.values())))
        print(f"  [eval it {it}] mean_corr={mc:.4f} | " + " ".join(f"{k}:{corrs[k]:+.3f}" for k in ATTRS), flush=True)
        if mc > best + 1e-4:
            best, best_it, noimp = mc, it, 0
            save_ckpt(f"{a.out}/det_z64_best.pt", it, best)
            print(f"  [eval] NEW BEST {best:.4f} @ it {it}", flush=True)
        else:
            noimp += 1
            print(f"  [eval] no improve ({noimp}/{a.early_stop}); best {best:.4f} @ it {best_it}", flush=True)
            if a.early_stop > 0 and noimp >= a.early_stop:
                print(f"  EARLY STOP @ it {it}", flush=True); break
    if it % 2000 == 0:
        save_ckpt(f"{a.out}/det_z64_latest.pt", it, best)

save_ckpt(f"{a.out}/det_z64_final.pt", a.iters, best)
print(f"DONE. best mean_corr {best:.4f} @ it {best_it}", flush=True)
