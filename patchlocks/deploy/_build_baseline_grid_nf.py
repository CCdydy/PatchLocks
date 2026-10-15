#!/usr/bin/env python3
"""fig:baselinegrid, NeuralFur-Fig.5 style -- white background, light clay meshes,
methods across columns (Input | NeuS | GenZoo | Ours), animals down rows.

NF house-style rules applied (measured from NeuralFur Fig.5/16, see
figure_workbench/_reference_neuralfur/ANATOMY.md):
  * pure-white canvas, no cell frames, no shadows (alpha cutouts);
  * column gutter ~16% of cell width, generous outer margin;
  * NO row labels (the Input column identifies the animal) and NO top header:
    method labels sit BELOW the bottom row, serif (citations live in the LaTeX caption);
  * every subject fit to a common box and bottom-aligned on a shared baseline.

Run from repo root:
  ~/miniconda3/envs/difflocks/bin/python <this file>
(or via the workbench rebuild.sh, which cds to repo root first).
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SURF = "_baselines/surf_render"
OURS = "v2_results/refine_v16_fluffy/global4_champagne"
PANDA_IMG = "patchlocks/datasets/neuralfur/extracted/data/Artemis/panda_processed_GH2/walk/images_2/0009.png"
PHOTO = {  # (data_dir, view) for the masked input photos
    "beagle": ("patchlocks/deploy/neuralfur_beagle_official/realviews/nf_data_f210", "0021"),
    "whiteTiger": ("patchlocks/deploy/neuralfur_whiteTiger_official/realviews/nf_data_roaringwalk_f5", "0009"),
    "cat": ("patchlocks/deploy/neuralfur_cat_official/realviews/nf_data_walk_final_f0", "0009"),
}
OUT = os.environ.get("BG_OUT", "outputs/figures/final_v16_4animals/BASELINE_GRID_NF.png")
ANIMALS = ["beagle", "whiteTiger", "panda", "cat"]
FS = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"   # NF labels are serif

# --- layout constants (NF-measured proportions) ---------------------------------
C = 500                  # cell size (px)
GUT = 80                 # column gutter = 0.16*C (NF: gap ~17% of cell width)
ROWGUT = 56              # row gutter, slightly tighter than columns
FOOT = 88                # bottom label band
PAD = 44                 # outer white margin
WHITE = (255, 255, 255)
INK = (20, 20, 20)
FIT_W, FIT_H = 0.90, 0.88   # subject fit box inside the cell
BASELINE = 0.94             # feet sit at this fraction of cell height
KIND_VIEW = {"neus": 1, "genzoo": 0}


def _place(rgb, mask):
    """Scale (fit-inside) the masked subject and bottom-align it on the shared
    baseline, centered horizontally, on a pure-white CxC cell."""
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


def _on_white(rgba):
    a = rgba[:, :, 3:4] / 255.0
    rgb = (rgba[:, :, :3] * a + 255.0 * (1 - a)).astype(np.uint8)
    return rgb, rgba[:, :, 3] / 255.0


def render_cell(path):
    rgb, a = _on_white(np.array(Image.open(path).convert("RGBA")).astype(np.float32))
    return _place(rgb, a > 0.5)


def input_cell(animal):
    if animal == "panda":
        img = np.array(Image.open(PANDA_IMG).convert("RGB")).astype(np.float32)
        mask = (255.0 - img).max(2) > 12
        rgb = np.where(mask[:, :, None], img, 255.0).astype(np.uint8)
        return _place(rgb, mask)
    d, v = PHOTO[animal]
    img = np.array(Image.open(f"{d}/images_2/{v}.png").convert("RGB")).astype(np.float32)
    m = np.array(Image.open(f"{d}/masks_2/body/{v}.png").convert("L")).astype(np.float32) / 255.0
    if m.shape != img.shape[:2]:
        m = np.array(Image.fromarray((m * 255).astype(np.uint8)).resize(
            (img.shape[1], img.shape[0]), Image.LANCZOS)).astype(np.float32) / 255.0
    mask = m > 0.4
    rgb = np.where(mask[:, :, None], img, 255.0).astype(np.uint8)
    return _place(rgb, mask)


def surf_cell(a, kind):
    p = f"{SURF}/{a}_{kind}/furview_{KIND_VIEW[kind]}.png"
    return render_cell(p) if os.path.exists(p) else None


def main():
    cols = [("Image", input_cell),
            ("NeuS", lambda a: surf_cell(a, "neus")),
            ("GenZoo", lambda a: surf_cell(a, "genzoo")),
            ("Ours", lambda a: render_cell(f"{OURS}/{a}/furview_1.png"))]
    nc = len(cols)
    W = PAD + nc * C + (nc - 1) * GUT + PAD
    H = PAD + 4 * C + 3 * ROWGUT + FOOT + PAD
    sheet = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(sheet)
    fh = ImageFont.truetype(FS, 40)
    for r, a in enumerate(ANIMALS):
        y = PAD + r * (C + ROWGUT)
        for c, (_, fn) in enumerate(cols):
            cell = fn(a)
            if cell is not None:
                sheet.paste(cell, (PAD + c * (C + GUT), y))
    # method labels BELOW the bottom row (NF convention), serif; \cite goes in the caption
    ly = PAD + 4 * C + 3 * ROWGUT + 26
    for c, (head, _) in enumerate(cols):
        x = PAD + c * (C + GUT)
        d.text((x + (C - d.textlength(head, font=fh)) / 2, ly), head, font=fh, fill=INK)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    sheet.save(OUT)
    print("nf-grid", sheet.size, "cols:", [h for h, _ in cols])


if __name__ == "__main__":
    main()
