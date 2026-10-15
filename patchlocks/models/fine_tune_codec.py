"""Step 3b: fine-tune the DiffLocks StrandCodec on procedural fur strands.
Init from pretrained strand_codec.pt, normalize with fur-pool stats, weighted pos/dir/curv loss
(mirrors DiffLocks weights 0.5/1.0/20, no KL since do_vae=False).
Run in difflocks env from repo root:
  python patchlocks/models/fine_tune_codec.py --iters 4000
"""
import os, sys, argparse, json
sys.path.insert(0, os.getcwd())
import numpy as np, torch
from models.strand_codec import StrandCodec, normalize_gt_data
from utils.strand_util import compute_dirs

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="patchlocks/datasets/pool/strands_256.npy")
ap.add_argument("--norm", default="patchlocks/datasets/pool/fur_norm.npz")
ap.add_argument("--iters", type=int, default=4000)
ap.add_argument("--bs", type=int, default=8192)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--out", default="patchlocks/models/fur_strand_codec.pt")
a = ap.parse_args()

dev = "cuda"
pool = torch.tensor(np.load(a.pool), dtype=torch.float32)          # (T,256,3) root-relative
T = pool.shape[0]
perm = torch.randperm(T); val_idx = perm[:4096]; tr_idx = perm[4096:]
val = pool[val_idx].to(dev)
print(f"pool {T} strands | train {len(tr_idx)} val {len(val_idx)}")

fn = np.load(a.norm)
def t(x): return torch.tensor(x, dtype=torch.float32, device=dev)
# curvature stats from pool
_c = np.diff(np.diff(pool.numpy(), axis=1), axis=1).reshape(-1, 3)
norm = {"xyz_mean": t(fn["xyz_mean"]), "xyz_std": t(fn["xyz_std"]),
        "dir_mean": t(fn["dir_mean"]), "dir_std": t(fn["dir_std"]),
        "curv_mean": t(_c.mean(0)), "curv_std": t(_c.std(0) + 1e-8)}
sx, sd, sc = norm["xyz_std"], norm["dir_std"], norm["curv_std"]

codec = StrandCodec(do_vae=False, decode_type="dir", scale_init=30.0,
                    nr_verts_per_strand=256, nr_values_to_decode=255, dim_per_value_decoded=3).to(dev)
codec.load_state_dict(torch.load("./checkpoints/strand_vae/strand_codec.pt"))   # pretrained init
print("loaded pretrained init")

opt = torch.optim.AdamW(codec.parameters(), lr=a.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters)

def encdec(pos):
    gt = {"strand_positions": pos, "strand_directions": compute_dirs(pos, append_last_dir=False)}
    z = codec.encoder(normalize_gt_data(gt, norm))["z"]
    pred = codec.decoder(z, None, norm)
    key = "strand_positions" if "strand_positions" in pred else [k for k in pred if "pos" in k.lower()][0]
    return pred[key]

def loss_fn(pred, gt):
    lp = (((pred - gt) / sx) ** 2).mean()
    dp, dg = torch.diff(pred, dim=1), torch.diff(gt, dim=1)
    ld = (((dp - dg) / sd) ** 2).mean()
    cp, cg = torch.diff(dp, dim=1), torch.diff(dg, dim=1)
    lc = (((cp - cg) / sc) ** 2).mean()
    return 0.5 * lp + 1.0 * ld + 20.0 * lc, (lp.item(), ld.item(), lc.item())

@torch.no_grad()
def val_err():
    codec.eval(); e = (encdec(val) - val).norm(dim=-1).mean().item(); codec.train(); return e

print(f"pretrained-init val round-trip err = {val_err():.5f} (our units)")
codec.train()
best = 1e9
for it in range(1, a.iters + 1):
    idx = tr_idx[torch.randint(0, len(tr_idx), (a.bs,))]
    pos = pool[idx].to(dev)
    pred = encdec(pos)
    loss, (lp, ld, lc) = loss_fn(pred, pos)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if it % 500 == 0 or it == 1:
        e = val_err()
        print(f"it {it:5d} | loss {loss.item():.4f} (pos {lp:.3f} dir {ld:.3f} curv {lc:.4f}) | val_err {e:.5f}", flush=True)
        if e < best:
            best = e
            torch.save({"state_dict": codec.state_dict(),
                        "fur_norm": {k: v.cpu().numpy() for k, v in norm.items()}}, a.out)
print(f"DONE best val_err={best:.5f} -> {a.out}")
