"""OOD generalization test set: coats whose parameters lie OUTSIDE the training envelope.

test-B used fresh seeds but the SAME 9 coats + ranges -> in-distribution. This set instead
extrapolates curl / length / curl_freq / elevation BEYOND every training coat's range, so the
diffusion has never seen these parameter regimes. Same generator machinery (so codec + bake +
eval work identically); only the coat parameter dicts are pushed OOD. Seeds 810000+ (disjoint
from train 210000-211599, B 610000+, all render seeds).

Question: does the fair-baseline result (diffusion > deterministic > scalar-synth) HOLD OOD, or
does the advantage collapse when the prior is asked for regimes it never trained on?

Run: /home/rog/miniconda3/envs/difflocks/bin/python \
  patchlocks/experiments/rebuttal_2027/gen_ood_patches.py --n_per 20 --out <dir>
"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), "patchlocks", "v11_longfur"))
import gen_realdog_v11 as G

# OOD coats: base training dict + parameters pushed strictly beyond ALL training ranges.
# (training maxima: curl 0.070, length 0.85, curl_freq 2.6, clump 0.50, elev 80, gravity 0.72)
OOD = {
    "curly_xtreme":  {**G.COATS["curly"],  "curl": (0.075, 0.11),  "clump": (0.50, 0.72),
                      "curl_freq": (2.2, 3.4), "length": (0.12, 0.26)},          # tighter-than-any curls
    "long_xtreme":   {**G.COATS["long_silky"], "length": (0.88, 1.25), "gravity": (0.72, 0.92),
                      "clump": (0.10, 0.35)},                                    # longer drape than any train
    "wiry_xtreme":   {**G.COATS["wiry"],   "curl": (0.028, 0.052), "curl_freq": (2.6, 4.2),
                      "clump": (0.20, 0.38)},                                    # coarser crinkle
    "standoff_xtreme": {**G.COATS["fluffy_double"], "elev_floor": 80.0, "elev_max": 92.0,
                        "length": (0.42, 0.60), "curl": (0.005, 0.020)},        # more vertical than any train
}
SCHEMES = ["solid", "two_tone", "bicolor", "tabby", "tipped"]

ap = argparse.ArgumentParser()
ap.add_argument("--n_per", type=int, default=20, help="patches per OOD coat")
ap.add_argument("--out", default="patchlocks/experiments/rebuttal_2027/ood_patches")
ap.add_argument("--seed0", type=int, default=810000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

# register OOD coats into the generator's global tables
G.COATS.update(OOD)

idx = 0
report = {c: [] for c in OOD}
for coat in OOD:
    for j in range(a.n_per):
        seed = a.seed0 + idx
        scheme = SCHEMES[idx % 5]
        name = f"rd_{idx:04d}_{coat}_{scheme}"
        d = G.generate_realdog_patch(seed=seed, coat=coat, color_scheme=scheme)
        G.save_patch(d, f"{a.out}/{name}.npz")
        # sanity stats on the raw generated strands (root-centered arc/curl proxies)
        S = d["strands"].astype(np.float32)      # (N,L,3)
        seg = np.linalg.norm(np.diff(S, axis=1), axis=2)
        arc = seg.sum(1)
        report[coat].append(float(np.median(arc)))
        idx += 1
    print(f"{coat}: {a.n_per} patches, median arc {np.median(report[coat]):.3f}", flush=True)

print(f"\nOOD_GEN_DONE n={idx} -> {a.out}")
print("per-coat median arc (length proxy):",
      {c: round(float(np.median(v)), 3) for c, v in report.items()})
