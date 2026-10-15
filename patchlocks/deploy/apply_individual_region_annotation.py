#!/usr/bin/env python3
"""Apply per-animal region annotations to a V2 region prior.

This is intentionally small: the learned prior remains unchanged, while
animal-specific semantic labels can be corrected before Step2/grow consume
region policies for ears, paws, legs, face, and tail.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _resolve(path: str, base: Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = base / p
    return str(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True, help="input v2_region_prior.npz")
    ap.add_argument("--annotation", required=True, help="per-animal annotation json")
    ap.add_argument("--out", required=True, help="output annotated prior")
    args = ap.parse_args()

    ann_path = Path(args.annotation).resolve()
    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    prior = np.load(args.prior, allow_pickle=True)
    out = {k: prior[k] for k in prior.files}

    notes = []
    patch_region_npz = ann.get("patch_region_npz", "")
    if patch_region_npz:
        reg_path = _resolve(patch_region_npz, ann_path.parent)
        reg = np.load(reg_path, allow_pickle=True)
        if "patch_region" not in reg.files:
            raise KeyError(f"{reg_path} does not contain patch_region")
        patch_region = reg["patch_region"].astype("U24")
        if "patch_region" in out and len(patch_region) != len(out["patch_region"]):
            raise ValueError(
                f"patch_region length mismatch: annotation {len(patch_region)} vs prior {len(out['patch_region'])}"
            )
        out["patch_region"] = patch_region
        names = np.asarray(sorted(set(patch_region.astype(str).tolist())), dtype="U24")
        out["region_names"] = names
        notes.append(f"patch_region <- {reg_path}")

    out["individual_annotation"] = np.asarray(os.path.abspath(args.annotation))
    out["individual_annotation_animal"] = np.asarray(str(ann.get("animal", "")))
    out["individual_annotation_notes"] = np.asarray("; ".join(notes) if notes else "none")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    vals, counts = np.unique(out["patch_region"].astype(str), return_counts=True) if "patch_region" in out else ([], [])
    print(
        f"wrote annotated prior -> {args.out} | "
        f"animal={ann.get('animal','')} | regions={dict(zip(vals.tolist(), counts.tolist()))} | "
        f"notes={out['individual_annotation_notes']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
