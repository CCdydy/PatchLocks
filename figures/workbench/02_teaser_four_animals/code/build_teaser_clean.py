#!/usr/bin/env python3
"""fig:teaser -- TEASER_4ANIMALS_CLEAN.png (main paper, Fig. 1).

Four camera-relative, soft-lit, shared-champagne grooms (beagle, white tiger,
panda, cat), no labels, 3600x600, white background. The published figure was
composed ad hoc (no script survived); this script reproduces it byte-for-byte.

Recipe (recovered by fitting against the published PNG):
  * crop each 1600x1600 RGBA render to CROP[animal] (= the alpha>0 bounding box
    plus a ~1.1% margin; the exact boxes are listed below because the original
    margin rule is unknown -- they were recovered from the published pixels);
  * resize the RGBA crop with LANCZOS to height H=592, width int(w*H/h);
  * paste left-to-right with alpha as mask on a white 3600x600 RGB canvas,
    starting at x=4, y=4, 4 px gaps.

Inputs: v2_results/refine_v16_fluffy/teaser_camera_relative_preview/<a>/
camera_relative_soft_champagne_1600.png (render_dog_fur_pro.py
--camera_relative_light --camera_relative_soft, Blender/Cycles).

Run from repo root:
  python figures/workbench/02_teaser_four_animals/code/build_teaser_clean.py
"""
import os

from PIL import Image

SRC = os.environ.get("TEASER_SRC", "v2_results/refine_v16_fluffy/teaser_camera_relative_preview")
NAME = os.environ.get("TEASER_RENDER", "camera_relative_soft_champagne_1600.png")
OUT = os.environ.get("TEASER_OUT", "outputs/figures/final_v16_4animals/TEASER_4ANIMALS_CLEAN.png")

W, H_CANVAS = 3600, 600
H = 592          # tile height
MARGIN = 4       # left / top margin
GAP = 4          # gap between tiles

# (left, top, right, bottom) in the 1600x1600 renders. alpha>0 bboxes are
# beagle (312,442,1250,1151), whiteTiger (309,556,1354,1082),
# panda (249,515,1229,1126), cat (344,302,1231,1090).
CROP = {
    "beagle": (301, 435, 1261, 1159),
    "whiteTiger": (297, 550, 1366, 1087),
    "panda": (238, 508, 1240, 1133),
    "cat": (335, 293, 1241, 1099),
}


def main():
    canvas = Image.new("RGB", (W, H_CANVAS), (255, 255, 255))
    x = MARGIN
    for a, box in CROP.items():
        tile = Image.open(f"{SRC}/{a}/{NAME}").convert("RGBA").crop(box)
        tile = tile.resize((int(tile.width * H / tile.height), H), Image.LANCZOS)
        canvas.paste(tile, (x, MARGIN), tile)
        x += tile.width + GAP
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    canvas.save(OUT)
    print("teaser", canvas.size, "->", OUT)


if __name__ == "__main__":
    main()
