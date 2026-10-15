"""Generate the v5 REALISTIC pet-fur patch batch. Distributes N patches across the 10 archetypes
(cat/dog, single/double coat); color scheme + finish + structure are randomized per patch inside the
generator, plus light size jitter (scale DR). Names <archetype>_<seed>.npz so render dirs are named by
archetype (stratified val + per-archetype analysis downstream).
Run:  python patchlocks/data_gen/gen_v5_batch.py --n 5000 --out patchlocks/datasets/v5/patches
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import gen_fur_patch_v5 as v5
from gen_fur_patch_v5 import save_patch, V6_COATS, V6_SHARES

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000); ap.add_argument("--seed0", type=int, default=600000)
    ap.add_argument("--out", default="patchlocks/datasets/v6/patches")
    ap.add_argument("--size_jit", type=float, default=0.25, help="size ~ U(1-s,1+s) scale DR; 0 disables")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    # v6: census-weighted, deterministic per-coat counts (not uniform round-robin -> kills 'generic medium-fluff' bias)
    counts = [int(round(s * a.n)) for s in V6_SHARES]
    while sum(counts) < a.n: counts[counts.index(max(counts))] += 1
    while sum(counts) > a.n: counts[counts.index(max(counts))] -= 1
    arch_list = [c for coat, k in zip(V6_COATS, counts) for c in [coat] * k]
    print(f"v6 batch: {a.n} patches, census shares {dict(zip(V6_COATS, counts))} -> {a.out}", flush=True)
    for i in range(a.n):
        arch = arch_list[i]; seed = a.seed0 + i
        sz = 1.0 if a.size_jit <= 0 else float(np.random.default_rng(seed ^ 0x9E3779B9).uniform(1 - a.size_jit, 1 + a.size_jit))
        d = v5.generate_patch(seed=seed, archetype=arch, size=sz)
        save_patch(d, os.path.join(a.out, f"{arch}_{seed}.npz"))
        if (i + 1) % 250 == 0:
            p = d["params"]; print(f"  {i+1}/{a.n} last={arch} N={len(d['strands'])} color={p['color_scheme']} finish={p['finish']} size={sz:.2f}", flush=True)
    print(f"V5-GEN-DONE {a.n} patches -> {a.out}", flush=True)
