"""C: bootstrap 95% CIs over per-patch values from eval_3arm.py JSON dumps.

For each arm and metric: 10k patch-level resamples -> CI for per-param nMAE and
corr, plus paired deltas between arms (v16 minus det / v16 minus d7s / v16 minus
d7_reg) so 'is the diffusion advantage significant' is answered directly.

Run: python bootstrap_ci.py --json eval3arm_testB.json --out ci_testB.json
"""
import json, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--json", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--nboot", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

d = json.load(open(a.json))
ATTRS = d["meta"]["attrs"]
pp = d["per_patch"]
n = len(pp)
rng = np.random.RandomState(a.seed)
ARMS = [k for k in ["v16", "det", "d7s", "d7_reg", "v16s"] if k in pp[0]]
PAIRS = [(x, y) for x, y in
         [("v16", "det"), ("v16", "d7s"), ("v16", "d7_reg"), ("v16s", "d7s")]
         if x in ARMS and y in ARMS]

G = {k: np.array([e["gt"][k] for e in pp]) for k in ATTRS}
A = {arm: {k: np.array([e[arm][k] for e in pp]) for k in ATTRS} for arm in ARMS}

def nmae(pred, gt):
    s = gt.std()
    return np.abs(pred - gt).mean() / (s + 1e-12)

def corr(u, v):
    if u.std() < 1e-9 or v.std() < 1e-9: return 0.0
    return np.corrcoef(u, v)[0, 1]

idx = rng.randint(0, n, size=(a.nboot, n))
out = {"meta": {"json": a.json, "n": n, "nboot": a.nboot}, "arms": {}, "deltas": {}}

boot = {arm: {"nmae": {k: [] for k in ATTRS}, "corr": {k: [] for k in ATTRS},
              "mean_nmae": [], "mean_corr": []} for arm in ARMS}
for b in range(a.nboot):
    ii = idx[b]
    for arm in ARMS:
        nm, co = [], []
        for k in ATTRS:
            v = nmae(A[arm][k][ii], G[k][ii]); boot[arm]["nmae"][k].append(v); nm.append(v)
            c = corr(A[arm][k][ii], G[k][ii]); boot[arm]["corr"][k].append(c); co.append(c)
        boot[arm]["mean_nmae"].append(np.mean(nm))
        boot[arm]["mean_corr"].append(np.mean(co))

def ci(x):
    x = np.asarray(x)
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 50)), float(np.percentile(x, 97.5))]

for arm in ARMS:
    out["arms"][arm] = {
        "mean_nmae_ci": ci(boot[arm]["mean_nmae"]),
        "mean_corr_ci": ci(boot[arm]["mean_corr"]),
        "nmae_ci": {k: ci(boot[arm]["nmae"][k]) for k in ATTRS},
        "corr_ci": {k: ci(boot[arm]["corr"][k]) for k in ATTRS},
    }

# paired deltas (same bootstrap indices -> proper paired comparison)
for base, other in PAIRS:
    dn = np.array(boot[base]["mean_nmae"]) - np.array(boot[other]["mean_nmae"])
    dc = np.array(boot[base]["mean_corr"]) - np.array(boot[other]["mean_corr"])
    out["deltas"][f"{base}_minus_{other}"] = {
        "mean_nmae_delta_ci": ci(dn),          # negative = base better (lower nMAE)
        "mean_corr_delta_ci": ci(dc),          # positive = base better
        "per_param_nmae_delta_ci": {k: ci(np.array(boot[base]["nmae"][k]) - np.array(boot[other]["nmae"][k])) for k in ATTRS},
        "per_param_corr_delta_ci": {k: ci(np.array(boot[base]["corr"][k]) - np.array(boot[other]["corr"][k])) for k in ATTRS},
    }

json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({arm: out["arms"][arm]["mean_nmae_ci"] for arm in ARMS}, indent=1))
print("deltas (v16-other, mean nMAE; negative=v16 better):")
for k, v in out["deltas"].items():
    print(f"  {k}: {v['mean_nmae_delta_ci']}")
print(f"saved -> {a.out}")
