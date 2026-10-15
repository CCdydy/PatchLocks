#!/usr/bin/env python3
"""Build Figure 3: observed patch -> predicted 3D -> final-groom detail.

All panels are square and arranged in one row for a single-column figure. The
last panel is cropped directly from the final beagle groom close-up, so the
figure shows fur texture rather than animal anatomy. The two rendered panels
receive the same light-champagne display transform; geometry is untouched.
"""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont


ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "v2_results/paper_cover_patch/beagle_p855"
META = RUN / "metadata.json"
PRED = RUN / "v16_pred_patch_top_champagne.png"
FINAL = (
    ROOT
    / "v2_results/refine_v16_fluffy/details_champagne/beagle/beagle_detail_v3.png"
)
OUT = Path(os.environ.get("PATCH_OUT", ROOT / "outputs/figures/PATCH_RESULT_SINGLE_COLUMN.png"))

TILE = 500
GAP = 18
HEADER = 48
WHITE = (255, 255, 255)
INK = (26, 26, 26)
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def repo_path(path: str) -> Path:
    """metadata.json stores absolute paths from the original machine
    (/home/rog/Desktop/diffusionF/...); resolve them against this repository."""
    p = Path(path)
    for anchor in ("patchlocks", "v2_results"):
        if p.is_absolute() and anchor in p.parts:
            return ROOT.joinpath(*p.parts[p.parts.index(anchor):])
    return ROOT / p


def observed_patch() -> Image.Image:
    meta = json.loads(META.read_text(encoding="utf-8"))
    record = next(view for view in meta["views"] if view["view_id"] == "0003")
    pack = np.load(repo_path(meta["source_pack"]), allow_pickle=True)
    patch_id = int(meta["patch_id"])
    view_idx = int(np.where(pack["view_ids"] == "0003")[0][0])
    source = Image.open(repo_path(record["source_image"])).convert("RGB")

    uv = pack["uv"][patch_id, view_idx].astype(np.float32)
    visible = (pack["vis"][patch_id, view_idx] > 0.5) & (
        pack["valid"][patch_id] > 0.5
    )
    px = (uv[..., 0] + 1.0) * 0.5 * source.width
    py = (uv[..., 1] + 1.0) * 0.5 * source.height
    xs, ys = px[visible], py[visible]
    cx, cy = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())
    side = 1.18 * max(xs.max() - xs.min(), ys.max() - ys.min())
    return source.crop(
        (round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2))
    )


def square(image: Image.Image, size: int = TILE) -> Image.Image:
    image = image.convert("RGB")
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize(
        (size, size), Image.Resampling.LANCZOS
    )


def focus(image: Image.Image, cx: float, cy: float, fraction: float) -> Image.Image:
    image = image.convert("RGB")
    side = round(min(image.size) * fraction)
    x = round(image.width * cx)
    y = round(image.height * cy)
    return image.crop((x - side // 2, y - side // 2, x + side // 2, y + side // 2))


def light_champagne(image: Image.Image) -> Image.Image:
    """Apply one restrained, geometry-preserving display transform."""
    image = ImageEnhance.Brightness(image.convert("RGB")).enhance(1.20)
    image = ImageEnhance.Color(image).enhance(0.78)
    return ImageEnhance.Contrast(image).enhance(0.94)


def final_fur_crop() -> Image.Image:
    image = Image.open(FINAL).convert("RGB")
    # Central torso only: no silhouette, limbs, face or background.
    return focus(image, 0.55, 0.36, 0.36)


def main() -> None:
    panels = [
        square(focus(observed_patch(), 0.64, 0.50, 0.68)),
        light_champagne(square(focus(
            Image.open(PRED).rotate(90, resample=Image.Resampling.BICUBIC), 0.50, 0.40, 0.66
        ))),
        light_champagne(square(final_fur_crop())),
    ]
    labels = ["Observed patch", "Predicted 3D", "Final groom"]
    width = 3 * TILE + 2 * GAP
    canvas = Image.new("RGB", (width, HEADER + TILE), WHITE)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(FONT), 30)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = index * (TILE + GAP)
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (TILE - (box[2] - box[0])) / 2, 5), label, font=font, fill=INK)
        canvas.paste(panel, (x, HEADER))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print(f"wrote {OUT} {canvas.size}")


if __name__ == "__main__":
    main()
