"""Compose the paper-styled fair-baseline qualitative figure from grid5 panels.
Adds left coat row-labels, thin separators, serif headers; white background.
Order: Input | GT | Diffusion (ours) | Deterministic | Predicted stats.
(ours placed right after GT so the eye compares ours->GT first; weakest arms last.)

Run: python \
       patchlocks/experiments/rebuttal_2027/paper_draft/compose_qual.py
"""
import os, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Paper figure = frozen test-set picks in grid5_B, rows curly, long, wavy, wiry.
# (Earlier val80 draft: QUAL_ROOT=.../grid5 QUAL_ROWS=short,wiry,wavy,ultra_short.)
ROOT = os.environ.get("QUAL_ROOT", "patchlocks/experiments/rebuttal_2027/grid5_B")
OUT = os.environ.get("QUAL_OUT", "outputs/figures/fig_baselines_qual.png")
picks = json.load(open(f"{ROOT}/picks.json"))
ROW_ORDER = os.environ.get("QUAL_ROWS", "curly,long,wavy,wiry").split(",")
ROW_LABEL = {"short": "short", "wiry": "wiry", "wavy": "wavy", "ultra_short": "ultra-short",
             "curly": "curly", "long": "long"}
picks = {p["coat"]: p for p in picks}
rows = [picks[c] for c in ROW_ORDER if c in picks]

# columns: (header, source-key). "input" = the render dir rgb_0; others = render_<arm>/*.png
COLS = [("Input", "input"), ("Ground truth", "gt"),
        ("Diffusion (ours)", "v16"), ("Deterministic", "det"),
        ("Predicted stats", "d7s")]

S = 360           # panel px
PADX = 4          # gap between panels
PADY = 4
LBLW = 92         # left label strip
HDR = 40          # top header strip
FS_HDR, FS_ROW = 25, 23

def load_panel(r, key):
    if key == "input":
        p = f"{r['dir']}/rgb_0.png"
    else:
        rd = f"{ROOT}/{r['coat']}/render_{key}"
        pngs = sorted(f for f in os.listdir(rd) if f.endswith(".png"))
        p = os.path.join(rd, pngs[0])
    return Image.open(p).convert("RGB").resize((S, S), Image.LANCZOS)

ncol, nrow = len(COLS), len(rows)
W = LBLW + ncol * S + (ncol + 1) * PADX
H = HDR + nrow * S + (nrow + 1) * PADY
canvas = Image.new("RGB", (W, H), (255, 255, 255))
dr = ImageDraw.Draw(canvas)
def font(sz, bold=False):
    fam = "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf"
    for base in ("/usr/share/fonts/truetype/dejavu/",):
        try: return ImageFont.truetype(base + fam, sz)
        except Exception: pass
    return ImageFont.load_default()
f_hdr, f_hdr_b, f_row = font(FS_HDR), font(FS_HDR, True), font(FS_ROW)

# column headers (ours in bold)
for ci, (name, key) in enumerate(COLS):
    x0 = LBLW + PADX + ci * (S + PADX)
    fnt = f_hdr_b if key == "v16" else f_hdr
    tw = dr.textlength(name, font=fnt)
    dr.text((x0 + (S - tw) / 2, 6), name, fill=(15, 15, 15), font=fnt)

# rows
for ri, r in enumerate(rows):
    y0 = HDR + PADY + ri * (S + PADY)
    # left rotated-free label (horizontal, centered in strip)
    lbl = ROW_LABEL[r["coat"]]
    tw = dr.textlength(lbl, font=f_row)
    lab = Image.new("RGB", (S, FS_ROW + 8), (255, 255, 255))
    ld = ImageDraw.Draw(lab); ld.text(((S - tw) / 2, 2), lbl, fill=(15, 15, 15), font=f_row)
    lab = lab.rotate(90, expand=True)
    canvas.paste(lab, (max(0, (LBLW - lab.width) // 2), y0 + (S - lab.height) // 2))
    for ci, (name, key) in enumerate(COLS):
        x0 = LBLW + PADX + ci * (S + PADX)
        canvas.paste(load_panel(r, key), (x0, y0))

# thin separator after GT (col idx 1) to set off the input|GT reference block from predictors
sep_x = LBLW + PADX + 2 * (S + PADX) - PADX // 2 - 1
dr.line([(sep_x, HDR), (sep_x, H - 2)], fill=(180, 180, 180), width=2)

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
canvas.save(OUT)
print(f"saved -> {OUT}  ({W}x{H})")
