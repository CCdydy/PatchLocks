"""Compare recomputed result JSONs (outputs/tables/) with the archived ones used in the paper.

usage: python pipeline/compare_tables.py [--tol 1e-6]

For each outputs/tables/<name>.json that has an archived counterpart in
patchlocks/experiments/rebuttal_2027/, walks both JSON trees and reports how many numeric leaves
differ by more than --tol, the largest absolute difference, and where it occurs.
Non-numeric leaves (paths, timestamps) are ignored.
"""
import argparse
import json
import math
import os

ARCHIVE = "patchlocks/experiments/rebuttal_2027"
NEW = "outputs/tables"


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield path, float(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()
    if not os.path.isdir(NEW):
        raise SystemExit(f"{NEW} does not exist; run pipeline/06_evaluate_paper_tables.sh first")
    print(f"{'result':28s} {'numbers':>8s} {'differ':>7s} {'max |diff|':>11s}  where")
    for name in sorted(os.listdir(NEW)):
        if not name.endswith(".json"):
            continue
        ref_path = os.path.join(ARCHIVE, name)
        if not os.path.exists(ref_path):
            print(f"{name:28s} (no archived counterpart)")
            continue
        new = dict(leaves(json.load(open(os.path.join(NEW, name)))))
        ref = dict(leaves(json.load(open(ref_path))))
        common = sorted(set(new) & set(ref))
        worst, where, n_diff = 0.0, "", 0
        for k in common:
            r, n = ref[k], new[k]
            if math.isnan(r) and math.isnan(n):
                continue
            d = abs(r - n)
            if d > a.tol:
                n_diff += 1
            if d > worst:
                worst, where = d, k
        extra = len(set(new) ^ set(ref))
        note = f"  ({extra} keys only on one side)" if extra else ""
        print(f"{name:28s} {len(common):8d} {n_diff:7d} {worst:11.3g}  {where}{note}")


if __name__ == "__main__":
    main()
