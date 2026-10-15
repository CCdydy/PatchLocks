#!/usr/bin/env python3
"""Paper figure builder — unified dark-panel style.

Style: most diagnostic figures use charcoal cells.  The teaser is deliberately
different: transparent melanin-shaded reconstructions and magnified fur crops, matching
the presentation convention used by NeuralFur while keeping only our outputs.

Run: ~/miniconda3/envs/difflocks/bin/python patchlocks/deploy/_build_paper_figs_dark.py <fig>
  fig in: teaser qual detail
"""
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ROOT = "."
HI = "patchlocks/deploy/nf_official_render"       # 2560 hi-res view-1 renders
import os
G4 = os.environ.get("G4", "v2_results/refine_v16_fluffy/global4")  # 1280 turntable views 0-3
DET = os.environ.get("DET", "v2_results/refine_v16_fluffy/details_champagne")
OUTD = os.environ.get("OUTD", "outputs/figures/final_v16_4animals")
os.makedirs(OUTD, exist_ok=True)
TEASER_RGBA = "v2_results/refine_v16_fluffy/teaser_color_rgba"

DARK = (46, 46, 46)
CANVAS = (255, 255, 255)
INK = (20, 20, 20)
TAG = (200, 200, 200)
ACCENT = (196, 44, 34, 255)          # crimson accents (teaser magnifier/ROI)
HALO = (255, 255, 255, 225)          # white under-stroke: keeps accents readable on fur
GUT = 14
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Teaser camera distances -- MUST match _render_teaser_color_rgba.sh so the ROI
# box on the overview lands exactly on the region the close-up magnifies.
FULL_CAMDIST = 2.05
DETAIL_CAMDIST = 0.42

ANIMALS = [("beagle", "Beagle"), ("whiteTiger", "White tiger"),
           ("panda", "Panda"), ("cat", "Cat")]

PHOTO = {  # (data_dir, view) for the input chips
    "beagle": ("patchlocks/deploy/neuralfur_beagle_official/realviews/nf_data_f210", "0021"),
    "whiteTiger": ("patchlocks/deploy/neuralfur_whiteTiger_official/realviews/nf_data_roaringwalk_f5", "0009"),
    "cat": ("patchlocks/deploy/neuralfur_cat_official/realviews/nf_data_walk_final_f0", "0009"),
}
PANDA_IMGDIR = "patchlocks/datasets/neuralfur/extracted/data/Artemis/panda_processed_GH2/walk"


def load_rgba(path):
    return np.array(Image.open(path).convert("RGBA")).astype(np.float32)


def comp_dark(path, bg=DARK):
    im = load_rgba(path)
    rgb, a = im[:, :, :3] / 255.0, im[:, :, 3:4] / 255.0
    b = np.array(bg, np.float32) / 255.0
    return Image.fromarray(((rgb * a + b * (1 - a)) * 255).astype(np.uint8))


def alpha_bbox(path, thr=10):
    a = np.array(Image.open(path))[:, :, 3]
    ys, xs = np.where(a > thr)
    return xs.min(), ys.min(), xs.max(), ys.max()


def tight_dark(path, margin=0.07, square=False):
    """Composite on dark and crop to alpha bbox + margin."""
    x0, y0, x1, y1 = alpha_bbox(path)
    im = comp_dark(path)
    w, h = x1 - x0, y1 - y0
    mx, my = int(w * margin), int(h * margin)
    if square:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(w, h) * (1 + 2 * margin) / 2
        box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    else:
        box = (x0 - mx, y0 - my, x1 + mx, y1 + my)
    box = (max(0, box[0]), max(0, box[1]),
           min(im.size[0], box[2]), min(im.size[1], box[3]))
    return im.crop(box)


def photo_chip(animal, size, view=None):
    """Masked input photo on near-white, tight square crop."""
    if animal == "panda":
        # ARTEMIS render already sits on pure white -- keep it as-is (photo look)
        img = np.array(Image.open(f"{PANDA_IMGDIR}/images_2/0009.png").convert("RGB")).astype(np.float32)
        m = (255.0 - img).max(axis=2) > 12
        ys, xs = np.where(m)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(x1 - x0, y1 - y0) * 0.52
        H, W = img.shape[:2]
        box = (int(max(0, cx - half)), int(max(0, cy - half)),
               int(min(W, cx + half)), int(min(H, cy + half)))
        chip = Image.fromarray(img.astype(np.uint8)).crop(box)
        return chip.resize((size, size), Image.LANCZOS)
    else:
        d, v = PHOTO[animal]
        if view is not None:
            v = view
        img = np.array(Image.open(f"{d}/images_2/{v}.png").convert("RGB")).astype(np.float32)
        msk = np.array(Image.open(f"{d}/masks_2/body/{v}.png").convert("L")).astype(np.float32) / 255.0
    if msk.shape != img.shape[:2]:
        msk = np.array(Image.fromarray((msk * 255).astype(np.uint8)).resize(
            (img.shape[1], img.shape[0]), Image.LANCZOS)).astype(np.float32) / 255.0
    m = msk[:, :, None]
    b = np.array((250, 250, 250), np.float32)   # photo look: subject on near-white
    out = img * m + b * (1 - m)
    ys, xs = np.where(msk > 0.4)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * 0.52
    H, W = img.shape[:2]
    box = (int(max(0, cx - half)), int(max(0, cy - half)),
           int(min(W, cx + half)), int(min(H, cy + half)))
    chip = Image.fromarray(out.astype(np.uint8)).crop(box)
    return chip.resize((size, size), Image.LANCZOS)


def alpha_crop_rgba_with_box(path, margin=0.04):
    """Return the tight transparent crop and its box in the source render."""
    im = Image.open(path).convert("RGBA")
    a = np.asarray(im)[:, :, 3]
    ys, xs = np.where(a > 8)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0, y1 - y0
    mx, my = int(w * margin), int(h * margin)
    box = (max(0, x0 - mx), max(0, y0 - my),
           min(im.width, x1 + mx), min(im.height, y1 + my))
    return im.crop(box), box, im.size


def alpha_crop_rgba_shadow(path, pad_x=0.05, pad_top=0.02, pad_bot=0.04):
    """Load a shadow-catcher render as a CLEAN cutout. The physical catcher
    shadow is stripped entirely (its density is scene-dependent and saturates to
    ink under the body); the figure draws a uniform synthesized contact shadow
    instead. Solid = alpha>128 excluding near-black catcher ink. Returns
    (crop, crop_box, src_size, feet_frac, foot_extent) where foot_extent is the
    (x0, x1) span of the silhouette's bottom 12% inside the crop (0..1)."""
    im = Image.open(path).convert("RGBA")
    arr = np.array(im)
    a = arr[:, :, 3].astype(np.int32)
    rgbsum = arr[:, :, :3].astype(np.int32).sum(2)
    ink = (a > 128) & (rgbsum < 45)          # opaque catcher shadow, not fur
    solid = (a > 128) & ~ink
    arr[:, :, 3] = np.where(solid, arr[:, :, 3], 0).astype(np.uint8)
    im = Image.fromarray(arr)
    ys, xs = np.where(solid)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0, y1 - y0
    box = (max(0, int(x0 - w * pad_x)), max(0, int(y0 - h * pad_top)),
           min(im.width, int(x1 + w * pad_x)), min(im.height, int(y1 + h * pad_bot)))
    feet = (y1 - box[1]) / (box[3] - box[1])
    band = solid[max(y0, y1 - int(0.12 * h)):y1 + 1, :]
    bxs = np.where(band.any(0))[0]
    foot = ((bxs.min() - box[0]) / (box[2] - box[0]),
            (bxs.max() - box[0]) / (box[2] - box[0]))
    return im.crop(box), box, im.size, feet, foot


def draw_contact_shadow(sheet, cx, cy, half_w, strength=0.24):
    """Uniform soft elliptical contact shadow (identical recipe per animal)."""
    from PIL import ImageFilter
    h = max(18, int(half_w * 0.16))
    pad = int(half_w * 0.6)
    lay = Image.new("L", (2 * (int(half_w) + pad), 2 * (h + pad)), 0)
    dd = ImageDraw.Draw(lay)
    dd.ellipse((pad, pad, pad + 2 * int(half_w), pad + 2 * h), fill=255)
    lay = lay.filter(ImageFilter.GaussianBlur(radius=h * 0.9))
    al = (np.asarray(lay, np.float32) * strength).astype(np.uint8)
    sh = np.zeros((*al.shape, 4), np.uint8); sh[:, :, 3] = al
    sheet.alpha_composite(Image.fromarray(sh),
                          (int(cx - al.shape[1] / 2), int(cy - al.shape[0] / 2)))


def stroke_roi(d, box, r=10):
    """Rounded accent box with a white halo under-stroke (readable on any fur)."""
    d.rounded_rectangle(box, radius=r, outline=HALO, width=8)
    d.rounded_rectangle(box, radius=r, outline=ACCENT, width=4)


def stroke_connector(d, p0, p1):
    d.line((*p0, *p1), fill=HALO, width=7)
    d.line((*p0, *p1), fill=ACCENT, width=3)
    for (px, py) in (p0, p1):
        d.ellipse((px - 7, py - 7, px + 7, py + 7), fill=ACCENT, outline=HALO, width=2)


# ---------------------------------------------------------------- teaser
def detail_coverage_roi(crop_box, src_size):
    """Normalized ROI, inside the tight alpha crop, that the close-up camera saw.

    The full (`side_color`) and detail (`detail_color`) renders share
    orientation, lens and target and differ only in camera distance, so the
    nearer detail camera frames a centered square whose side is
    DETAIL_CAMDIST/FULL_CAMDIST of the full frame. We express that square in the
    coordinates of the tight crop so the ROI box lands on what the inset shows.
    """
    cx0, cy0, cx1, cy1 = crop_box
    src_w, src_h = src_size
    frac = DETAIL_CAMDIST / FULL_CAMDIST
    sx0, sx1 = src_w * (1 - frac) / 2, src_w * (1 + frac) / 2
    sy0, sy1 = src_h * (1 - frac) / 2, src_h * (1 + frac) / 2
    return (max(0.0, (sx0 - cx0) / (cx1 - cx0)),
            max(0.0, (sy0 - cy0) / (cy1 - cy0)),
            min(1.0, (sx1 - cx0) / (cx1 - cx0)),
            min(1.0, (sy1 - cy0) / (cy1 - cy0)))


def paste_magnifier(sheet, d, img_path, box, zoom):
    """Paste a square close-up into `box`=(x, y) and stroke its accent frame."""
    zx, zy = box
    inset = Image.open(img_path).convert("RGBA").resize((zoom, zoom), Image.LANCZOS)
    sheet.alpha_composite(inset, (zx, zy))
    d.rounded_rectangle((zx, zy, zx + zoom, zy + zoom), radius=14,
                        outline=HALO, width=9)
    d.rounded_rectangle((zx, zy, zx + zoom, zy + zoom), radius=14,
                        outline=ACCENT, width=5)


def build_teaser():
    """Shared-melanin reconstructions with a real close-up magnifier per animal
    and a shadow-catcher contact shadow (fur shading untouched -- the shadow
    lives only in the render alpha). Feet (not shadows) share one baseline."""
    CELL_W, PITCH, ZOOM = 1150, 1005, 580     # hi-res layout (~1.35x)
    LABH, ROW_H, SHPAD = 72, 800, 60          # label strip / animal band / shadow overflow
    ITOP = LABH + 6
    ROW_TOP = ITOP + ZOOM + 26
    BASE_Y = ROW_TOP + ROW_H
    W, H = 3 * PITCH + CELL_W, BASE_Y + SHPAD
    sheet = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(FB, 46)

    for i, (a, label) in enumerate(ANIMALS):
        x0 = i * PITCH
        groom, crop_box, src_size, feet, foot = alpha_crop_rgba_shadow(
            f"{TEASER_RGBA}/{a}/side_color.png")

        # overview groom: fit by SOLID height, pin feet to the shared baseline
        scale = min(CELL_W * 0.90 / groom.width, ROW_H * 0.98 / (groom.height * feet))
        sw, sh = int(groom.width * scale), int(groom.height * scale)
        gx, gy = x0 + (CELL_W - sw) // 2, BASE_Y - int(feet * sh)
        fcx = gx + (foot[0] + foot[1]) / 2 * sw
        fhw = max((foot[1] - foot[0]) * sw * 0.58, sw * 0.30)
        draw_contact_shadow(sheet, fcx, BASE_Y - 2, fhw)
        sheet.alpha_composite(groom.resize((sw, sh), Image.LANCZOS), (gx, gy))

        # magnifier (dedicated close-up render) centered above the cell
        zx, zy = x0 + (CELL_W - ZOOM) // 2, ITOP
        paste_magnifier(sheet, d, f"{TEASER_RGBA}/{a}/detail_color.png", (zx, zy), ZOOM)

        # ROI box on the groom + halo connector to the magnifier
        rx0, ry0, rx1, ry1 = detail_coverage_roi(crop_box, src_size)
        bx0, by0, bx1, by1 = gx + rx0 * sw, gy + ry0 * sh, gx + rx1 * sw, gy + ry1 * sh
        stroke_connector(d, (zx + ZOOM / 2, zy + ZOOM), ((bx0 + bx1) / 2, by0))
        stroke_roi(d, (bx0, by0, bx1, by1))

        tw = d.textlength(label, font=font)
        d.text((x0 + (CELL_W - tw) / 2, 8), label, font=font, fill=INK)

    # flatten onto white: the figure sits on a white page, and flattening
    # removes any viewer inconsistency with semi-transparent PNG (soft shadows)
    flat = Image.new("RGBA", sheet.size, (255, 255, 255, 255))
    flat.alpha_composite(sheet)
    flat.convert("RGB").save(f"{OUTD}/TEASER_4ANIMALS_SIDE.png")
    print("teaser", sheet.size)


# ---------------------------------------------------------------- qual 4x4
def build_qual():
    ORDER = [1, 2, 3, 0]       # 3/4, side, front, rear-last
    C = 470
    LABW = 250
    font = ImageFont.truetype(FB, 40)
    W = LABW + 4 * C + 3 * GUT + GUT
    H = 4 * C + 3 * GUT + 2 * GUT
    sheet = Image.new("RGB", (W, H), CANVAS)
    d = ImageDraw.Draw(sheet)
    for r, (a, label) in enumerate(ANIMALS):
        y = GUT + r * (C + GUT)
        # wrap two-word labels
        lines = label.split(" ") if " " in label else [label]
        ty = y + C / 2 - 26 * len(lines)
        for li, ln in enumerate(lines):
            tw = d.textlength(ln, font=font)
            d.text((max(8, (LABW - 16 - tw) / 2), ty + li * 48), ln, font=font, fill=INK)
        for c, v in enumerate(ORDER):
            cell = tight_dark(f"{G4}/{a}/furview_{v}.png", margin=0.08, square=True)
            cell = cell.resize((C, C), Image.LANCZOS)
            sheet.paste(cell, (LABW + c * (C + GUT), y))
    sheet.save(f"{OUTD}/ALL_GLOBAL_4ANIMALS.png")
    print("qual", sheet.size)


# ---------------------------------------------------------------- detail
def build_detail():
    C = 470
    LABH = 50
    font = ImageFont.truetype(FB, 34)
    W = 4 * C + 3 * GUT
    H = LABH + 2 * C + GUT
    sheet = Image.new("RGB", (W, H), CANVAS)
    d = ImageDraw.Draw(sheet)
    for i, (a, label) in enumerate(ANIMALS):
        x = i * (C + GUT)
        tw = d.textlength(label, font=font)
        d.text((x + (C - tw) / 2, 6), label, font=font, fill=INK)
        for j, v in enumerate([1, 3]):
            im = comp_dark(f"{DET}/{a}/{a}_detail_v{v}.png")
            # detail renders are close-ups; center-crop square
            w, h = im.size
            s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            sheet.paste(im.resize((C, C), Image.LANCZOS),
                        (x, LABH + j * (C + GUT)))
    sheet.save(f"{OUTD}/ALL_DETAIL_4ANIMALS_WIDE.png")
    print("detail", sheet.size)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("teaser", "all"):
        build_teaser()
    if which in ("qual", "all"):
        build_qual()
    if which in ("detail", "all"):
        build_detail()
