#!/usr/bin/env python3
"""fig:patch rebuild — per animal (row): representative reliable 2D crop (with
patch-count badge) | canonical 3D character swatch | reliability over the surface.

Data is the ACTUAL v16 run (v2_results/refine_v16_fluffy): source_patch_ids from
each canon swatch (beagle 1 / whiteTiger 9 / panda 3 / cat 2), reliability from
character_prior.npz, projection (sel/uv/vis/cand_names) from each groom's
source_pack surface_patch_dataset.npz. Crops reuse the real_crop projection logic
of patchlocks/qa/v2_patch_io_gate.py.

Run: ~/miniconda3/envs/difflocks/bin/python patchlocks/deploy/_build_patchfig_dark.py
"""
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

R = "v2_results/refine_v16_fluffy"
OUT = "PatchLocks_paper/Figures/final_v16_4animals/ALL_PATCHES_AND_CHARACTER_TIGHT.png"
DARK = (46, 46, 46)
INK = (20, 20, 20)
GUT = 14
C = 470
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
ANIMALS = [("beagle", "Beagle"), ("whiteTiger", "White tiger"),
           ("panda", "Panda"), ("cat", "Cat")]


def crop_pad(img, x0, y0, x1, y1):
    h, w = img.shape[:2]
    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    return img[y0:y1, x0:x1]


def real_crop(pack, p, res=470, crop_margin=1.2):
    sel = pack["sel"].astype(np.int32)
    uvp = pack["uv"].astype(np.float32)
    visp = pack["vis"].astype(np.float32)
    valid = pack["valid"].astype(np.float32)
    cand_names = [str(x) for x in pack["cand_names"]]
    k = int(sel[p, 0])
    img = cv2.cvtColor(cv2.imread(cand_names[k]), cv2.COLOR_BGR2RGB)
    ih, iw = img.shape[:2]
    uv = uvp[p, k]
    px = (uv[..., 0] + 1.0) * 0.5 * iw
    py = (uv[..., 1] + 1.0) * 0.5 * ih
    m = (visp[p, k] > 0.5) & (valid[p] > 0.5)
    xs, ys = px[m], py[m]
    cx = 0.5 * (float(xs.min()) + float(xs.max()))
    cy = 0.5 * (float(ys.min()) + float(ys.max()))
    half = max(0.5 * crop_margin * max(float(xs.max()) - xs.min(),
                                       float(ys.max()) - ys.min()), 8.0)
    crop = crop_pad(img, int(cx - half), int(cy - half),
                    int(cx + half), int(cy + half))
    return Image.fromarray(cv2.resize(crop, (res, res),
                                      interpolation=cv2.INTER_CUBIC))


def swatch_cell(a, res=C, x_half=0.11, z_top=0.075):
    """Side-elevation slice of the canonical swatch: strands rooted in a thin
    central strip, drawn in the x-z plane at a SHARED scale across animals, so
    relative length / lift / curl are directly comparable row to row."""
    sw = np.load(f"{R}/{a}/canon_swatch_reliable_straight.npz", allow_pickle=True)
    S = sw["strands"].astype(np.float32)          # (N,32,3)
    roots = S[:, 0, :]
    ctr = roots.mean(0)
    # root the strip left-of-center: strands flow +x, so tips stay in frame
    band = (np.abs(roots[:, 1] - ctr[1]) < 0.05 * (roots[:, 1].max() - roots[:, 1].min())) \
        & (roots[:, 0] - ctr[0] > -0.135) & (roots[:, 0] - ctr[0] < -0.02)
    S = S[band]
    idx = np.random.RandomState(0).choice(len(S), min(260, len(S)), replace=False)
    S = S[idx]
    dpi = 100
    fig, ax = plt.subplots(figsize=(res / dpi, res / dpi), dpi=dpi)
    fig.patch.set_facecolor(np.array(DARK) / 255.0)
    ax.set_facecolor(np.array(DARK) / 255.0)
    for i in range(len(S)):
        z = S[i, :, 2]
        t = np.linspace(0.55, 0.97, len(z))
        ax.plot(S[i, :, 0] - ctr[0], z, color=(t[-1], t[-1], 1.0), lw=0.0)
        for j in range(len(z) - 1):
            ax.plot(S[i, j:j + 2, 0] - ctr[0], z[j:j + 2],
                    color=(t[j], t[j], min(1.0, t[j] + 0.05)), lw=0.9,
                    alpha=0.9, solid_capstyle="round")
    ax.axhline(0.0, color=(0.62, 0.62, 0.62), lw=2.0)   # surface line
    ax.set_xlim(-0.145, 0.145)
    ax.set_ylim(-0.006, z_top)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return Image.fromarray(buf).resize((res, res), Image.LANCZOS)


def reliability_cell(a, res=C):
    pr = np.load(f"{R}/{a}/character_prior.npz", allow_pickle=True)
    gp = np.load(f"{R}/{a}/grow/surfpack_v2fur.npz", allow_pickle=True)
    sp = np.load(str(gp["source_pack"]), allow_pickle=True)
    cen = sp["centers"].astype(np.float32)
    rel = np.nan_to_num(pr["reliability"].astype(np.float32))
    # side view: meshes are y-up (render --up y); horizontal = wider of x/z
    v = cen - cen.mean(0)
    hz = 0 if v[:, 0].var() >= v[:, 2].var() else 2
    x, y = v[:, hz], v[:, 1]
    dpi = 100
    fig, ax = plt.subplots(figsize=(res / dpi, res / dpi), dpi=dpi)
    o = np.argsort(rel)
    sc = ax.scatter(x[o], y[o], c=rel[o], cmap="viridis", s=13,
                    vmin=0, vmax=1.0, linewidths=0)   # fixed scale: rows comparable
    ax.set_aspect("equal")
    ax.set_axis_off()
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=12)
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return Image.fromarray(buf).resize((res, res), Image.LANCZOS)


def main():
    font = ImageFont.truetype(FB, 38)
    hfont = ImageFont.truetype(FB, 34)
    bfont = ImageFont.truetype(FB, 26)
    LABW = 240
    HDR = 56
    W = LABW + 3 * C + 2 * GUT + GUT
    H = HDR + 4 * C + 3 * GUT + GUT
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    heads = ["Reliable 2D patch", "Canonical 3D character", "Reliability map"]
    for j, hd in enumerate(heads):
        x = LABW + j * (C + GUT)
        tw = d.textlength(hd, font=hfont)
        d.text((x + (C - tw) / 2, 12), hd, font=hfont, fill=INK)
    for r, (a, label) in enumerate(ANIMALS):
        y = HDR + r * (C + GUT)
        lines = label.split(" ")
        ty = y + C / 2 - 26 * len(lines)
        for li, ln in enumerate(lines):
            tw = d.textlength(ln, font=font)
            d.text((max(8, (LABW - 16 - tw) / 2), ty + li * 48), ln,
                   font=font, fill=INK)
        gp = np.load(f"{R}/{a}/grow/surfpack_v2fur.npz", allow_pickle=True)
        sp = np.load(str(gp["source_pack"]), allow_pickle=True)
        sw = np.load(f"{R}/{a}/canon_swatch_reliable_straight.npz",
                     allow_pickle=True)
        pr = np.load(f"{R}/{a}/character_prior.npz", allow_pickle=True)
        ids = sw["source_patch_ids"].astype(int)
        rel = np.nan_to_num(pr["reliability"].astype(np.float32))
        best = int(ids[np.argmax(rel[ids])])
        crop = real_crop(sp, best)
        cd = ImageDraw.Draw(crop)
        badge = f"{len(ids)} reliable patch" + ("es" if len(ids) > 1 else "")
        bw = cd.textlength(badge, font=bfont)
        cd.rectangle([10, C - 52, 10 + bw + 22, C - 12], fill=(0, 0, 0))
        cd.text((21, C - 46), badge, font=bfont, fill=(255, 255, 255))
        sheet.paste(crop, (LABW, y))
        sheet.paste(swatch_cell(a), (LABW + C + GUT, y))
        sheet.paste(reliability_cell(a), (LABW + 2 * (C + GUT), y))
        print("row done", a, "ids", ids.tolist(), "best", best)
    sheet.save(OUT)
    print("saved", OUT, sheet.size)


if __name__ == "__main__":
    main()
