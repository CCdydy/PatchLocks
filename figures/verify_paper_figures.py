#!/usr/bin/env python3
"""Compare the rebuilt figures in outputs/figures/ with the published ones.

    python figures/verify_paper_figures.py --paper-figures /path/to/PatchLocks_paper/Figures \
        [--intermediates /path/to/PatchLocks_paper_unused_20260827/Figures]

For every figure included by PatchLocks.tex / PatchLocks_supp.tex it reports
BYTE-IDENTICAL, PIXEL-IDENTICAL (same decoded RGB, different encoding), or the
max abs difference / mean abs difference / fraction of differing pixels / PSNR.
`--intermediates` additionally checks the full-resolution montages that the
supplement panels and _supp60 copies are cut from. The paper's pipeline.png is a
manual draw.io export; its reference copy in figures/pipeline/ is checked instead.
Exit status is 0 only if every checked figure is at least pixel-identical.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "figures"

# (paper-relative path, label, built path relative to outputs/figures, or None if manual)
FIGURES = [
    ("final_v16_4animals/TEASER_4ANIMALS_CLEAN.png", "fig:teaser", "final_v16_4animals/TEASER_4ANIMALS_CLEAN.png"),
    ("pipeline.png", "fig:overview", None),
    ("final_v16_4animals/NF_COMPARE_2ANIMALS.png", "fig:nfcompare", "final_v16_4animals/NF_COMPARE_2ANIMALS.png"),
    ("fig_baselines_qual.png", "fig:baselines_qual", "fig_baselines_qual.png"),
    ("supplement_panels/global_turntable_A.jpg", "fig:qual_full", "supplement_panels/global_turntable_A.jpg"),
    ("supplement_panels/global_turntable_B.jpg", "fig:qual_full_more", "supplement_panels/global_turntable_B.jpg"),
    ("supplement_panels/nf_compare_A.jpg", "fig:nfcompare_full", "supplement_panels/nf_compare_A.jpg"),
    ("supplement_panels/nf_compare_B.jpg", "fig:nfcompare_full_more", "supplement_panels/nf_compare_B.jpg"),
    ("_supp60/final_v16_4animals/BASELINE_GRID_NF.png", "fig:baselinegrid", "_supp60/final_v16_4animals/BASELINE_GRID_NF.png"),
    ("_supp60/final_v16_4animals/ALL_PATCHES_AND_CHARACTER_TIGHT.png", "fig:patch", "_supp60/final_v16_4animals/ALL_PATCHES_AND_CHARACTER_TIGHT.png"),
    ("PATCH_RESULT_SINGLE_COLUMN.png", "fig:patch_result", "PATCH_RESULT_SINGLE_COLUMN.png"),
    ("supplement_panels/visual_gate_A.jpg", "fig:gate_a", "supplement_panels/visual_gate_A.jpg"),
    ("supplement_panels/visual_gate_B.jpg", "fig:gate_b", "supplement_panels/visual_gate_B.jpg"),
]

# full-resolution intermediates (archived in PatchLocks_paper_unused_20260827/Figures)
INTERMEDIATES = [
    ("final_v16_4animals/ALL_GLOBAL_4ANIMALS.png", "-> global_turntable_A/B"),
    ("final_v16_4animals/NF_COMPARE_4ANIMALS.png", "-> nf_compare_A/B"),
    ("final_v16_4animals/BASELINE_GRID_NF.png", "-> _supp60 BASELINE_GRID_NF"),
    ("final_v16_4animals/ALL_PATCHES_AND_CHARACTER_TIGHT.png", "-> _supp60 ALL_PATCHES..."),
]


def compare(built: Path, ref: Path):
    if not ref.exists():
        return False, "REFERENCE MISSING"
    if not built.exists():
        return False, "NOT BUILT"
    if built.read_bytes() == ref.read_bytes():
        return True, "BYTE-IDENTICAL"
    a, b = Image.open(built), Image.open(ref)
    if a.size != b.size:
        return False, f"SIZE {a.size} != {b.size}"
    x = np.asarray(a.convert("RGB"), dtype=np.int32)
    y = np.asarray(b.convert("RGB"), dtype=np.int32)
    d = np.abs(x - y)
    if d.max() == 0:
        return True, "PIXEL-IDENTICAL"
    mse = float((d.astype(np.float64) ** 2).mean())
    psnr = 10 * np.log10(255.0 ** 2 / mse)
    return False, (f"DIFF max={d.max()} mean={d.mean():.4f} "
                   f"px%={100 * (d.max(2) > 0).mean():.3f} PSNR={psnr:.2f}dB")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paper-figures", default=os.environ.get("PAPER_FIGURES"),
                    help="the paper's Figures/ directory (or env PAPER_FIGURES)")
    ap.add_argument("--intermediates", default=os.environ.get("PAPER_INTERMEDIATES"),
                    help="optional: archived full-resolution montages (unused_*/Figures)")
    ap.add_argument("--built", default=str(OUT), help="build output dir (default outputs/figures)")
    args = ap.parse_args()
    if not args.paper_figures:
        ap.error("--paper-figures (or PAPER_FIGURES) is required")
    paper, built = Path(args.paper_figures), Path(args.built)

    rows, ok_all = [], True
    for rel, label, out_rel in FIGURES:
        if out_rel is None:
            ok, status = compare(REPO / "figures/pipeline/pipeline_published.png", paper / rel)
            status = "MANUAL draw.io export; reference copy " + status
        else:
            ok, status = compare(built / out_rel, paper / rel)
        ok_all &= ok
        rows.append((rel, label, status))
    if args.intermediates:
        for rel, note in INTERMEDIATES:
            ok, status = compare(built / rel, Path(args.intermediates) / rel)
            ok_all &= ok
            rows.append((rel, "(intermediate " + note + ")", status))

    w0 = max(len(r[0]) for r in rows)
    w1 = max(len(r[1]) for r in rows)
    print(f"{'figure':<{w0}}  {'label':<{w1}}  result")
    print("-" * (w0 + w1 + 40))
    for rel, label, status in rows:
        print(f"{rel:<{w0}}  {label:<{w1}}  {status}")
    print("ALL OK" if ok_all else "SOME FIGURES DIFFER")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
