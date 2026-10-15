#!/usr/bin/env python3
"""fig:nfcompare in NF house style -- white canvas, bottom serif labels, airy gutters.

Columns: Input | NeuralFur | NeuralFur (zoom) | Ours | Ours (zoom).
Zoom pairs crop the SAME image window (red box) from the 2x-resolution render of
the same view -- this red-box zoom design exceeds NF Fig.17 and is kept.
Full-body and zoom cells keep their original crop geometry (window coordinates
must stay aligned); NF-ization here is canvas/labels/spacing only.

Sources: nf_official_render/render_<a>_champ{,_ours} (2560, unified paper
material on BOTH sides) and the masked input photos.

Run from repo root: ~/miniconda3/envs/difflocks/bin/python <this file>
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

NF = "patchlocks/deploy/nf_official_render"
PANDA_IMG = "patchlocks/datasets/neuralfur/extracted/data/Artemis/panda_processed_GH2/walk/images_2/0009.png"
PHOTO = {
    "beagle": ("patchlocks/deploy/neuralfur_beagle_official/realviews/nf_data_f210", "0021"),
    "whiteTiger": ("patchlocks/deploy/neuralfur_whiteTiger_official/realviews/nf_data_roaringwalk_f5", "0009"),
    "cat": ("patchlocks/deploy/neuralfur_cat_official/realviews/nf_data_walk_final_f0", "0009"),
}
OUT = os.environ.get("NF_OUT", "outputs/figures/final_v16_4animals/NF_COMPARE_4ANIMALS.png")
ANIMALS = os.environ.get("NF_ANIMALS", "beagle,whiteTiger,panda,cat").split(",")
# Per-animal "Ours" render dir override, e.g. NF_OURS="panda=render_panda_champ_ours_len120"
# (main-paper Fig. 3 uses the length-1.2 panda groom; the supp. 4-animal sheet does not).
NF_OURS = dict(kv.split("=", 1) for kv in os.environ.get("NF_OURS", "").split(",") if kv)
# Animals whose "Ours" full-body cell is cropped from the Ours render's own alpha
# instead of the NeuralFur render's (needed when the Ours groom is longer and its
# silhouette no longer fits the NeuralFur crop), e.g. NF_OURS_OWN_CROP=panda.
OWN_CROP = set(a for a in os.environ.get("NF_OURS_OWN_CROP", "").split(",") if a)
FS = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"

# zoom window centers in 1280 coords, tuned on camdist 2.3 then rescaled to the
# camdist 2.05 framing of the unified renders (same values as the dark builder)
WIN = {"beagle": (483, 616), "whiteTiger": (434, 645),
       "panda": (490, 629), "cat": (640, 640)}
WIN = {k: (int(640 + (x - 640) * 1.122), int(640 + (y - 640) * 1.122))
       for k, (x, y) in WIN.items()}
S1280 = 320                    # zoom half-window: 320px at 1280 = 640 at 2560
BOX = (235, 75, 55)

C = 470
GUT = 72                       # ~0.15*C (NF airiness)
ROWGUT = 52
FOOT = 84
PAD = 40
WHITE = (255, 255, 255)
INK = (20, 20, 20)


def comp_white(path, size1280=False):
    im = np.array(Image.open(path).convert("RGBA")).astype(np.float32)
    a = im[:, :, 3:4] / 255.0
    rgb = (im[:, :, :3] * a + 255.0 * (1 - a)).astype(np.uint8)
    if size1280 and rgb.shape[0] != 1280:
        rgb = np.array(Image.fromarray(rgb).resize((1280, 1280), Image.LANCZOS))
    return rgb


def body_crop_square(path, margin=1.12):
    im = Image.open(path)
    if im.width != 1280:
        im = im.resize((1280, 1280), Image.LANCZOS)
    al = np.array(im)[:, :, 3]
    ys, xs = np.where(al > 10)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * margin / 2
    cx = min(max(cx, half), 1280 - half); cy = min(max(cy, half), 1280 - half)
    return (int(cx - half), int(cy - half), int(cx + half), int(cy + half))


def cell_full(path, win, crop):
    rgb = comp_white(path, size1280=True)
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    cx, cy = win
    d.rectangle([cx - S1280 // 2, cy - S1280 // 2, cx + S1280 // 2, cy + S1280 // 2],
                outline=BOX, width=6)
    return im.crop(crop).resize((C, C), Image.LANCZOS)


def cell_zoom(path_hi, win):
    rgb = comp_white(path_hi)                       # 2560
    cx, cy = win[0] * 2, win[1] * 2
    s = S1280
    return Image.fromarray(rgb[cy - s:cy + s, cx - s:cx + s]).resize((C, C), Image.LANCZOS)


def input_cell(animal):
    if animal == "panda":
        img = np.array(Image.open(PANDA_IMG).convert("RGB")).astype(np.float32)
        mask = (255.0 - img).max(2) > 12
        rgb = np.where(mask[:, :, None], img, 255.0).astype(np.uint8)
    else:
        d, v = PHOTO[animal]
        img = np.array(Image.open(f"{d}/images_2/{v}.png").convert("RGB")).astype(np.float32)
        m = np.array(Image.open(f"{d}/masks_2/body/{v}.png").convert("L")).astype(np.float32) / 255.0
        if m.shape != img.shape[:2]:
            m = np.array(Image.fromarray((m * 255).astype(np.uint8)).resize(
                (img.shape[1], img.shape[0]), Image.LANCZOS)).astype(np.float32) / 255.0
        mask = m > 0.4
        rgb = np.where(mask[:, :, None], img, 255.0).astype(np.uint8)
    ys, xs = np.where(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * 0.56
    H, W = rgb.shape[:2]
    sq = Image.new("RGB", (int(2 * half), int(2 * half)), WHITE)
    box = (int(max(0, cx - half)), int(max(0, cy - half)),
           int(min(W, cx + half)), int(min(H, cy + half)))
    sq.paste(Image.fromarray(rgb).crop(box), (0, 0))
    return sq.resize((C, C), Image.LANCZOS)


def main():
    heads = ["Input", "NeuralFur", "NeuralFur (zoom)", "Ours", "Ours (zoom)"]
    nc = 5
    W = PAD + nc * C + (nc - 1) * GUT + PAD
    NR = len(ANIMALS)
    H = PAD + NR * C + (NR - 1) * ROWGUT + FOOT + PAD
    sheet = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(sheet)
    fh = ImageFont.truetype(FS, 38)
    for r, a in enumerate(ANIMALS):
        y = PAD + r * (C + ROWGUT)
        win = WIN[a]
        crop = body_crop_square(f"{NF}/render_{a}_champ/furview_1.png")
        ours = f"{NF}/{NF_OURS.get(a, f'render_{a}_champ_ours')}/furview_1.png"
        ours_crop = body_crop_square(ours) if a in OWN_CROP else crop
        cells = [
            input_cell(a),
            cell_full(f"{NF}/render_{a}_champ/furview_1.png", win, crop),
            cell_zoom(f"{NF}/render_{a}_champ/furview_1.png", win),
            cell_full(ours, win, ours_crop),
            cell_zoom(ours, win),
        ]
        for c, im in enumerate(cells):
            sheet.paste(im, (PAD + c * (C + GUT), y))
    ly = PAD + NR * C + (NR - 1) * ROWGUT + 24
    for c, head in enumerate(heads):
        x = PAD + c * (C + GUT)
        d.text((x + (C - d.textlength(head, font=fh)) / 2, ly), head, font=fh, fill=INK)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    # the published Fig. 3 PNG was written with optimize=True (pixels are identical either way)
    sheet.save(OUT, optimize=os.environ.get("NF_PNG_OPTIMIZE", "0") == "1")
    print("nfcompare-nf", sheet.size)


if __name__ == "__main__":
    main()
