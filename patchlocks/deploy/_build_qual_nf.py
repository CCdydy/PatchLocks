#!/usr/bin/env python3
"""fig:qual in NeuralFur-Fig.15 "render wall" style -- pure white, airy, no labels.

House-style rules (measured from NF, see figure_workbench/_reference_neuralfur/
ANATOMY.md): white canvas, alpha-cutout subjects (no shadows, no cell frames),
column gutter ~16% of cell, subjects fit to a common box and bottom-aligned on a
shared baseline. Rows = animals (beagle, white tiger, panda, cat), columns = our
synchronized turntable views [1,2,3,0]; row identity is carried by the caption
(NF Fig.15 has zero in-figure text).

Source renders: transparent RGBA turntable views (G4 env overrides the dir).

Run from repo root:
  ~/miniconda3/envs/difflocks/bin/python <this file>
"""
import os
import numpy as np
from PIL import Image

G4 = os.environ.get("G4", "v2_results/refine_v16_fluffy/global4_champagne")
OUT = os.environ.get("Q_OUT", "PatchLocks_paper/Figures/final_v16_4animals/ALL_GLOBAL_4ANIMALS.png")
ANIMALS = ["beagle", "whiteTiger", "panda", "cat"]
ORDER = [int(x) for x in os.environ.get("Q_VIEWS", "1,2,3,0").split(",")]

# layout (NF-measured proportions, consistent with 06_baseline_grid)
C = 500
GUT = 80                       # 0.16*C column gutter
ROWGUT = 56
PAD = 44
WHITE = (255, 255, 255)
FIT_W, FIT_H = 0.92, 0.90
BASELINE = 0.95


def _place(rgb, mask):
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return Image.fromarray(rgb).resize((C, C), Image.LANCZOS)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bw, bh = x1 - x0, y1 - y0
    scale = min(FIT_W * C / bw, FIT_H * C / bh)
    nw, nh = max(1, int(round(bw * scale))), max(1, int(round(bh * scale)))
    crop = Image.fromarray(rgb[y0:y1, x0:x1]).resize((nw, nh), Image.LANCZOS)
    cell = Image.new("RGB", (C, C), WHITE)
    cell.paste(crop, ((C - nw) // 2, int(round(BASELINE * C)) - nh))
    return cell


def render_cell(path):
    rgba = np.array(Image.open(path).convert("RGBA")).astype(np.float32)
    a = rgba[:, :, 3:4] / 255.0
    rgb = (rgba[:, :, :3] * a + 255.0 * (1 - a)).astype(np.uint8)
    return _place(rgb, rgba[:, :, 3] > 128)


def main():
    NC = len(ORDER)
    TR = os.environ.get("Q_TRANSPOSE", "0") == "1"   # animals as columns, views as rows
    rows, cols = (NC, len(ANIMALS)) if TR else (len(ANIMALS), NC)
    W = PAD + cols * C + (cols - 1) * GUT + PAD
    H = PAD + rows * C + (rows - 1) * ROWGUT + PAD
    sheet = Image.new("RGB", (W, H), WHITE)
    for ai, a in enumerate(ANIMALS):
        for vi, v in enumerate(ORDER):
            r, c = (vi, ai) if TR else (ai, vi)
            cell = render_cell(f"{G4}/{a}/furview_{v}.png")
            sheet.paste(cell, (PAD + c * (C + GUT), PAD + r * (C + ROWGUT)))
    sheet.save(OUT)
    print("qual-nf", sheet.size, "src:", G4)


if __name__ == "__main__":
    main()
