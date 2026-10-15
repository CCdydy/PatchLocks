"""Merge the label-free smoothed character field into a V2 region prior.

This keeps the V1 grow contract intact:
  - build_v2_region_prior.py owns density / color / direction proposal plumbing.
  - extract_fur_character.py owns local fur character measurements.
  - this script injects the smooth character field as patch-level grow priors.

The output is still a standard v2_region_prior.npz, so grow_v2_region_prior.py
does not need a special code path.
"""
import argparse
import os

import numpy as np


def copy_npz(npz):
    return {k: npz[k] for k in npz.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True, help="v2_region_prior.npz")
    ap.add_argument("--character", required=True, help="character_prior.npz with character_field_* arrays")
    ap.add_argument("--out", required=True)
    ap.add_argument("--length", choices=["replace", "multiply", "off"], default="replace")
    ap.add_argument("--lift", choices=["replace", "multiply", "off"], default="replace")
    ap.add_argument("--clip_length", type=float, nargs=2, default=(0.35, 2.8))
    ap.add_argument("--clip_lift", type=float, nargs=2, default=(0.35, 2.2))
    args = ap.parse_args()

    prior = np.load(args.prior, allow_pickle=True)
    char = np.load(args.character, allow_pickle=True)
    out = copy_npz(prior)

    P = len(prior["final_density"])
    def require(name):
        if name not in char.files:
            raise KeyError(f"{args.character} does not contain {name}")
        arr = char[name].astype(np.float32)
        if len(arr) != P:
            raise ValueError(f"{name} length mismatch: {len(arr)} vs prior P={P}")
        return arr

    if args.length != "off":
        lr = np.clip(require("patch_length_ratio_field"), args.clip_length[0], args.clip_length[1]).astype(np.float32)
        if args.length == "multiply" and "patch_length_ratio" in prior.files:
            base = prior["patch_length_ratio"].astype(np.float32)
            lr = np.clip(base * lr, args.clip_length[0], args.clip_length[1]).astype(np.float32)
        out["patch_length_ratio"] = lr

    if args.lift != "off":
        ls = np.clip(require("patch_lift_scale_field"), args.clip_lift[0], args.clip_lift[1]).astype(np.float32)
        if args.lift == "multiply" and "patch_lift_scale" in prior.files:
            base = prior["patch_lift_scale"].astype(np.float32)
            ls = np.clip(base * ls, args.clip_lift[0], args.clip_lift[1]).astype(np.float32)
        out["patch_lift_scale"] = ls

    passthrough = [
        "character_field_arc_over_psize",
        "character_field_straightness",
        "character_field_lift_over_arc",
        "character_field_curl",
        "character_field_tip_spread",
        "character_field_clump",
        "character_field_radius_over_arc",
        "character_field_reliability",
        "patch_radius_over_arc_field",
    ]
    for k in passthrough:
        if k in char.files:
            out[k] = char[k].astype(np.float32)
    if "character_field_reliability" in char.files:
        out["patch_character_conf"] = char["character_field_reliability"].astype(np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **out)
    print(f"wrote character-field prior -> {args.out}", flush=True)
    if "patch_length_ratio" in out:
        v = out["patch_length_ratio"].astype(np.float32)
        print(f"  patch_length_ratio [{v.min():.3f},{v.max():.3f}] med={np.median(v):.3f}", flush=True)
    if "patch_lift_scale" in out:
        v = out["patch_lift_scale"].astype(np.float32)
        print(f"  patch_lift_scale  [{v.min():.3f},{v.max():.3f}] med={np.median(v):.3f}", flush=True)
    if "patch_character_conf" in out:
        v = out["patch_character_conf"].astype(np.float32)
        print(f"  character_conf    [{v.min():.3f},{v.max():.3f}] med={np.median(v):.3f}", flush=True)


if __name__ == "__main__":
    main()
