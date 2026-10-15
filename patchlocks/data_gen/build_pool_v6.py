"""Build the codec fine-tune strand pool from the v6 REDESIGNED renders (NOT gen_fur_patch).
Pulls 'strands' straight out of v6/renders/*/patch.npz, resamples to 256, root-relative.
Outputs strands_256.npy + fur_norm.npz + pool_meta.json (matches fine_tune_codec.py expectations).
Run: python patchlocks/data_gen/build_pool_v6.py --renders patchlocks/datasets/v6/renders --out patchlocks/datasets/v6_codec_pool --per 100
"""
import os, sys, glob, json, argparse
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--renders", default="patchlocks/datasets/v6/renders")
ap.add_argument("--out", default="patchlocks/datasets/v6_codec_pool")
ap.add_argument("--per", type=int, default=100, help="strands sampled per patch (census-weighted pool)")
ap.add_argument("--M", type=int, default=256)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(a.seed)
def resample(s, M):
    seg = np.linalg.norm(np.diff(s, axis=0), axis=1); cl = np.concatenate([[0], np.cumsum(seg)]); tot = cl[-1] + 1e-9
    t = np.linspace(0, tot, M); return np.stack([np.interp(t, cl, s[:, i]) for i in range(3)], 1)
dirs = sorted(glob.glob(a.renders + "/*/patch.npz"))
print(f"{len(dirs)} v6 patches", flush=True)
chunks = []; coat_ct = {}
for i, p in enumerate(dirs):
    coat = "_".join(os.path.basename(os.path.dirname(p)).split("_")[:-1])
    S = np.load(p, allow_pickle=True)["strands"].astype(np.float32)
    if len(S) == 0: continue
    idx = rng.choice(len(S), min(a.per, len(S)), replace=False)
    s256 = np.stack([resample(S[j], a.M) for j in idx]).astype(np.float32)
    s256 = s256 - s256[:, :1]                                  # root-relative
    chunks.append(s256); coat_ct[coat] = coat_ct.get(coat, 0) + len(idx)
    if (i + 1) % 500 == 0: print(f"  {i+1}/{len(dirs)} | strands={sum(c.shape[0] for c in chunks)}", flush=True)
pool = np.concatenate(chunks, 0).astype(np.float32)
np.save(f"{a.out}/strands_256.npy", pool)
xyz = pool.reshape(-1, 3); d = np.diff(pool, axis=1).reshape(-1, 3)
np.savez(f"{a.out}/fur_norm.npz", xyz_mean=xyz.mean(0), xyz_std=xyz.std(0), dir_mean=d.mean(0), dir_std=d.std(0))
# upright content check (tip elevation)
t = pool[:, -1]; ch = np.linalg.norm(t, axis=1); v = ch > 1e-6
elev = np.degrees(np.arcsin(np.clip(t[v, 2] / ch[v], -1, 1)))
json.dump({"n_strands": int(pool.shape[0]), "per": a.per, "M": a.M, "coat_counts": coat_ct,
           "elev_mean": float(elev.mean()), "frac_upright_gt45": float((elev > 45).mean())},
          open(f"{a.out}/pool_meta.json", "w"), indent=2)
print(f"POOL DONE {pool.shape} -> {a.out} | elev mean {elev.mean():.1f} | upright>45deg {100*(elev>45).mean():.1f}% | coats {coat_ct}")
