#!/usr/bin/env python3
"""patch_capability_v16_clean rebuild — normalized display material.

Rows (held-out coat archetypes): short-dense, wavy, curly, long.
Cols: conditioning input | ground-truth 3D | prediction, where GT and prediction
share camera AND display material (render_patch_samecam neutral renders), so the
column comparison shows geometry only --- same treatment as the cover figure.

Run: ~/miniconda3/envs/difflocks/bin/python patchlocks/deploy/_build_capability_clean.py
"""
import os

from PIL import Image, ImageDraw, ImageFont

SRC = "v2_results/paper_cover_patch/dataset_3coats"
OUT = os.environ.get("CAP_OUT", "outputs/figures/patch_capability_v16_clean.png")
ROWS = ["rd_0081_short_dense_two_tone", "rd_0418_wavy_tipped",
        "rd_0592_curly_solid", "rd_0849_long_golden_two_tone"]
INK, GUT, C = (20, 20, 20), 14, 560
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
HEADS = ["Input", "Ground-truth 3D", "Ours"]

def cell(path):
    return Image.open(path).convert("RGB").resize((C, C), Image.LANCZOS)

def main():
    HDR = 66
    W = 3 * C + 2 * GUT + 2 * GUT
    H = HDR + 4 * C + 3 * GUT + GUT
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    f = ImageFont.truetype(FB, 44)
    for c, head in enumerate(HEADS):
        x = GUT + c * (C + GUT)
        tw = d.textlength(head, font=f)
        d.text((x + (C - tw) / 2, 8), head, font=f, fill=INK)
    for r, s in enumerate(ROWS):
        y = HDR + r * (C + GUT)
        sheet.paste(cell(f"{SRC}/{s}/input_2d.png"), (GUT, y))
        sheet.paste(cell(f"{SRC}/{s}/gt_render/rgb_0.png"), (GUT + C + GUT, y))
        sheet.paste(cell(f"{SRC}/{s}/pred_render/rgb_0.png"), (GUT + 2 * (C + GUT), y))
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    sheet.save(OUT)
    print("wrote", OUT, sheet.size)

if __name__ == "__main__":
    main()
