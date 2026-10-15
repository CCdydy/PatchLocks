#!/usr/bin/env python3
"""Build readable, publication-scale panels from existing patch results.

This script only changes presentation. It does not rerun inference, alter the
predicted strands, or overwrite the source figures.
"""

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Repository root (figures/workbench/11_supplement_panels/code/<this file>).
ROOT = Path(__file__).resolve().parents[4]
FIGURES = ROOT / "outputs" / "figures"
OUT = Path(os.environ.get("SUPP_OUT", FIGURES / "supplement_panels"))
# Full-resolution source montages (env overrides; defaults = the build outputs /
# the archived visual-gate render used by the paper).
GLOBAL_SRC = Path(os.environ.get(
    "SUPP_GLOBAL", FIGURES / "final_v16_4animals" / "ALL_GLOBAL_4ANIMALS.png"))
NF_SRC = Path(os.environ.get(
    "SUPP_NF", FIGURES / "final_v16_4animals" / "NF_COMPARE_4ANIMALS.png"))
GATE_SRC = Path(os.environ.get(
    "SUPP_GATE", ROOT / "patchlocks/datasets/v16/v16_visual_gate_agx/v16_visual_gate.png"))
# patch_capability panels are not used by the final paper; built only if present.
CAPABILITY_SRC = Path(os.environ.get(
    "SUPP_CAPABILITY", FIGURES / "patch_capability_v16_clean.png"))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size=size)


def save_jpeg(image: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(
        OUT / name,
        quality=94,
        subsampling=0,
        optimize=True,
        progressive=True,
    )


def build_capability_panels() -> None:
    if not CAPABILITY_SRC.exists():
        print(f"skip patch_capability panels (no {CAPABILITY_SRC})")
        return
    src = Image.open(CAPABILITY_SRC).convert("RGB")
    # The source is a regular four-row sheet. Repeat its column header so both
    # half-page panels remain self-contained when LaTeX floats them separately.
    header_end = round(src.height * 66 / 2362)
    row2_end = round(src.height * 1214 / 2362)
    row4_end = round(src.height * 2348 / 2362)
    header = src.crop((0, 0, src.width, header_end))
    panel_a = src.crop((0, 0, src.width, row2_end))
    lower = src.crop((0, row2_end, src.width, row4_end))
    panel_b = Image.new("RGB", (src.width, header.height + lower.height), "white")
    panel_b.paste(header, (0, 0))
    panel_b.paste(lower, (0, header.height))
    save_jpeg(panel_a, "patch_capability_A.jpg")
    save_jpeg(panel_b, "patch_capability_B.jpg")


def draw_centered(draw: ImageDraw.ImageDraw, xy, text: str, fnt, fill=(28, 28, 28)) -> None:
    x, y = xy
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((x - (box[2] - box[0]) / 2, y), text, font=fnt, fill=fill)


def build_visual_gate_panels() -> None:
    src = Image.open(GATE_SRC).convert("RGB")
    row_h = src.height // 8
    strip_h = max(24, round(row_h * 0.09))
    gutter = 190
    top = 68
    gap = 8
    labels = [
        "Ultra-short\nbicolor",
        "Short dense\ntwo-tone",
        "Wavy\ntipped",
        "Wiry\nsolid",
        "Curly\nsolid",
        "Fluffy double\ntipped",
        "Long golden\ntwo-tone",
        "Ultra-short\nbicolor",
    ]
    col_titles = ["Input", "Ground-truth 3D", "Ours"]
    col_w = src.width // 3
    body_h = row_h - strip_h

    for panel_idx in range(2):
        canvas_h = top + 4 * body_h + 3 * gap
        canvas = Image.new("RGB", (gutter + src.width, canvas_h), "white")
        draw = ImageDraw.Draw(canvas)
        header_font = font(25, bold=True)
        label_font = font(22, bold=True)

        for c, title in enumerate(col_titles):
            draw_centered(
                draw,
                (gutter + c * col_w + col_w / 2, 17),
                title,
                header_font,
            )
        draw.line((gutter, top - 9, canvas.width, top - 9), fill=(215, 215, 215), width=2)

        for local_row in range(4):
            source_row = panel_idx * 4 + local_row
            y0 = source_row * row_h + strip_h
            tile = src.crop((0, y0, src.width, (source_row + 1) * row_h))
            y = top + local_row * (body_h + gap)
            canvas.paste(tile, (gutter, y))
            draw.multiline_text(
                (18, y + body_h / 2),
                labels[source_row],
                font=label_font,
                fill=(38, 38, 38),
                anchor="lm",
                spacing=3,
            )
            draw.rectangle(
                (gutter, y, canvas.width - 1, y + body_h - 1),
                outline=(226, 226, 226),
                width=2,
            )

        save_jpeg(canvas, f"visual_gate_{'A' if panel_idx == 0 else 'B'}.jpg")


def build_whole_animal_panels() -> None:
    global_src = Image.open(GLOBAL_SRC).convert("RGB")
    global_mid = global_src.height // 2
    save_jpeg(global_src.crop((0, 0, global_src.width, global_mid)), "global_turntable_A.jpg")
    save_jpeg(
        global_src.crop((0, global_mid, global_src.width, global_src.height)),
        "global_turntable_B.jpg",
    )

    nf_src = Image.open(NF_SRC).convert("RGB")
    # Four equal result rows are followed by a shared column-label strip. Keep
    # that strip under each two-animal panel.
    row_split = round(nf_src.height * 1084 / 2200)
    labels_start = round(nf_src.height * 2076 / 2200)
    labels = nf_src.crop((0, labels_start, nf_src.width, nf_src.height))
    upper = nf_src.crop((0, 0, nf_src.width, row_split))
    lower = nf_src.crop((0, row_split, nf_src.width, labels_start))
    for name, body in (("nf_compare_A.jpg", upper), ("nf_compare_B.jpg", lower)):
        panel = Image.new("RGB", (nf_src.width, body.height + labels.height), "white")
        panel.paste(body, (0, 0))
        panel.paste(labels, (0, body.height))
        save_jpeg(panel, name)


def main() -> None:
    build_capability_panels()
    build_visual_gate_panels()
    build_whole_animal_panels()
    print(f"Wrote supplement panels to {OUT}")


if __name__ == "__main__":
    main()
