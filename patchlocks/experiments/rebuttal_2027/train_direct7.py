"""Arm 3 (Direct-7): direct regression of the eval character parameters (CONTRACT.md).

ParamHead precedent (train_param_head.py): small CNN over the frozen fused condition
G_COND + global DINO token -> MLP -> the 6 eval ATTRS (arc, curl, clump, straight,
lift_arc, tip_spread). GT targets = env.decode_params on the GT z64 at fixed-seed
texels. Same env/e1_dir as the diffusion arm => identical conditions. Selection on
the seed-0 val split mean Pearson corr; test set B never touched.
(radius_ratio is not regressed: it does not enter the geometry measurement; the
primitive synthesis uses the fixed paper radius policy for every arm.)

Smoke:
  cd /home/rog/Desktop/diffusionF && /home/rog/miniconda3/envs/difflocks/bin/python \
    patchlocks/experiments/rebuttal_2027/train_direct7.py --max_dirs 80 --iters 60 \
    --eval_every 30 --bs 16 --out /tmp/d7_smoke
Full:
  ... train_direct7.py --out patchlocks/experiments/rebuttal_2027/direct7 \
      --iters 20000 --bs 32 --eval_every 1000 --early_stop 6
"""
import os, sys, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.join(os.getcwd(), "patchlocks", "v14"))
from v14_common import build_env, GLOBC

ap = argparse.ArgumentParser()
ap.add_argument("--renders", default="patchlocks/datasets/v11_longfur/renders_mv")
ap.add_argument("--e1_dir", default="patchlocks/datasets/v16/e2a_cw15")
ap.add_argument("--out", default="patchlocks/experiments/rebuttal_2027/direct7")
ap.add_argument("--iters", type=int, default=20000)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--head_dim", type=int, default=128)
ap.add_argument("--gt_len_k", type=int, default=64, help="texels for GT target stability")
ap.add_argument("--eval_every", type=int, default=1000)
ap.add_argument("--early_stop", type=int, default=6)
ap.add_argument("--max_dirs", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
ATTRS = ["arc", "curl", "clump", "straight", "lift_arc", "tip_spread"]

env = build_env(renders=a.renders, e1_dir=a.e1_dir, max_dirs=a.max_dirs,
                len_k=a.gt_len_k, verbose=True)
fetch, ti, vi, dirs = env.fetch, env.ti, env.vi, env.dirs
N = len(dirs)

print("decode GT z64 -> the 6 eval ATTRS targets ...", flush=True)
TGT = torch.full((N, len(ATTRS)), float("nan"), device=dev)
valid = torch.zeros(N, dtype=torch.bool, device=dev)
with torch.no_grad():
    allidx = torch.arange(N, device=env.idx_dev)
    for s in range(0, N, 64):
        idx = allidx[s:s + 64]
        z = fetch(env.Xn, idx); mm = fetch(env.MASK, idx)
        ib, ihw, kept = env.sample_texels(mm, a.gt_len_k, seed=12345)
        if ib is None: continue
        dp = env.decode_params(z, ib, ihw, a.gt_len_k)
        kept_idx = [int(idx[i]) for i in range(len(idx)) if bool(kept[i])]
        for row, gi in enumerate(kept_idx):
            for c, k in enumerate(ATTRS):
                TGT[gi, c] = dp[k][row]
            valid[gi] = True
print(f"GT targets for {int(valid.sum())}/{N} patches", flush=True)
ti = ti[valid[ti.to(dev)].cpu() if ti.device.type == "cpu" else valid[ti]]
vi = vi[valid[vi.to(dev)].cpu() if vi.device.type == "cpu" else valid[vi]]
print(f"train {len(ti)} / val {len(vi)}", flush=True)

def _take(x, idx): return x[idx.to(x.device)]
tt = _take(TGT, ti)
smean = tt.mean(0, keepdim=True); sstd = tt.std(0, unbiased=False).clamp_min(1e-6)[None]

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

torch.manual_seed(a.seed)
head = Direct7Head(env.G_COND.shape[1], GLOBC, a.head_dim, n_out=len(ATTRS)).to(dev)
print(f"Direct7 params: {sum(p.numel() for p in head.parameters())/1e6:.3f}M", flush=True)
opt = torch.optim.AdamW(head.parameters(), a.lr, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(a.iters, 1))

def _pcorr(u, v):
    u, v = u.float(), v.float()
    if u.numel() < 3 or u.std(unbiased=False) < 1e-6 or v.std(unbiased=False) < 1e-6: return 0.0
    um, vm = u - u.mean(), v - v.mean()
    return float((um * vm).mean() / (um.std(unbiased=False) * vm.std(unbiased=False) + 1e-12))

@torch.no_grad()
def evaluate():
    head.eval()
    P, G = [], []
    for s in range(0, len(vi), 128):
        jj = vi[s:s + 128]
        ps = head(fetch(env.G_COND, jj), fetch(env.GLOB, jj)) * sstd + smean
        P.append(ps); G.append(_take(TGT, jj))
    P, G = torch.cat(P), torch.cat(G)
    head.train()
    return {ATTRS[i]: _pcorr(P[:, i], G[:, i]) for i in range(len(ATTRS))}

def save_ckpt(path, it, best):
    tmp = path + ".tmp"
    torch.save({"head": head.state_dict(), "args": vars(a), "iter": int(it),
                "attrs": ATTRS, "smean": smean.cpu(), "sstd": sstd.cpu(),
                "cond_ch": int(env.G_COND.shape[1]), "glob_ch": int(GLOBC),
                "head_dim": a.head_dim, "best_corr": float(best), "direct7": True}, tmp)
    os.replace(tmp, path)

best, best_it, noimp = -1e9, 0, 0
print("train Direct-7 ...", flush=True)
for it in range(1, a.iters + 1):
    bi = ti[torch.randint(0, len(ti), (a.bs,), device=env.idx_dev)]
    tgt = (_take(TGT, bi) - smean) / sstd
    ps = head(fetch(env.G_COND, bi), fetch(env.GLOB, bi))
    loss = F.smooth_l1_loss(ps, tgt)
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if it % 200 == 0:
        print(f"  it {it} | loss {loss.item():.4f} | lr {sch.get_last_lr()[0]:.2e}", flush=True)
    if a.eval_every > 0 and it % a.eval_every == 0:
        corrs = evaluate()
        mc = float(np.mean(list(corrs.values())))
        print(f"  [eval it {it}] mean_corr={mc:.4f} | " + " ".join(f"{k}:{corrs[k]:+.3f}" for k in ATTRS), flush=True)
        if mc > best + 1e-4:
            best, best_it, noimp = mc, it, 0
            save_ckpt(f"{a.out}/direct7_best.pt", it, best)
            print(f"  [eval] NEW BEST {best:.4f} @ it {it}", flush=True)
        else:
            noimp += 1
            if a.early_stop > 0 and noimp >= a.early_stop:
                print(f"  EARLY STOP @ it {it} (best {best:.4f} @ {best_it})", flush=True); break

save_ckpt(f"{a.out}/direct7_final.pt", a.iters, best)
print(f"DONE. best mean_corr {best:.4f} @ it {best_it}", flush=True)
