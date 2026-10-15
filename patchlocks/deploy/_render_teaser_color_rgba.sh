#!/usr/bin/env bash
# Render the locked v16 reconstructions for the paper teaser.
# Geometry/groom is unchanged.  Every animal uses the same locked melanin Hair
# BSDF so appearance cannot hide or exaggerate cross-animal fur geometry.
set -euo pipefail

ROOT="${ROOT:-/home/rog/Desktop/diffusionF}"
cd "$ROOT"
source patchlocks/deploy/paper_render_style.env

BLENDER="patchlocks/third_party/blender-4.5.10-linux-x64/blender"
SRC="v2_results/refine_v16_fluffy"
OUTROOT="$SRC/teaser_color_rgba"
mkdir -p "$OUTROOT"

declare -A MESH
MESH[beagle]="patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/beagle_dog/furless_reshaped.obj"
MESH[whiteTiger]="patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/whiteTiger/furless_reshaped.obj"
MESH[panda]="patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/panda/furless.obj"
MESH[cat]="patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/cat/furless_reshaped.obj"

ANIMALS="${ANIMALS:-beagle whiteTiger panda cat}"
RENDER_FULL="${RENDER_FULL:-1}"
DETAIL_RES="${DETAIL_RES:-1600}"
DETAIL_SAMPLES="${DETAIL_SAMPLES:-64}"
DETAIL_CAMDIST="${DETAIL_CAMDIST:-0.42}"
for animal in $ANIMALS; do
  out="$OUTROOT/$animal"
  mkdir -p "$out"
  if [[ "$RENDER_FULL" == "1" ]]; then
    "$BLENDER" -b --factory-startup \
      --python patchlocks/deploy/render_dog_fur_pro.py -- \
      --fur "$SRC/$animal/grow/surfpack_v2fur.npz" \
      --mesh "${MESH[$animal]}" \
      --out "$out" --views 1 --res 1800 --samples 64 --up y --camdist 2.05 \
      --az_deg 270 --el_deg 9 --out_name side_color.png \
      --param melanin --melanin "$PAPER_MELANIN" --melanin_red "$PAPER_MELANIN_RED" \
      --roughness "$PAPER_ROUGHNESS" --radial_rough "$PAPER_RADIAL_ROUGH" \
      --rand_rough 0.20 --rand_color 0.08 --radius_mul "$PAPER_RADIUS_MUL" \
      --skin "$PAPER_SKIN" --exposure "$PAPER_EXPOSURE" --view agx \
      --look "$PAPER_LOOK" --paper_light --camera_relative_light --transparent \
      > "$out/render.log" 2>&1
    echo "teaser full-body RGBA done: $animal"
  fi

  # True high-resolution torso close-up.  It keeps the exact full-body camera
  # orientation and lens, changing only distance, so the paper magnifier box
  # has a well-defined geometric correspondence to this render.
  "$BLENDER" -b --factory-startup \
    --python patchlocks/deploy/render_dog_fur_pro.py -- \
    --fur "$SRC/$animal/grow/surfpack_v2fur.npz" \
    --mesh "${MESH[$animal]}" \
    --out "$out" --views 1 --res "$DETAIL_RES" --samples "$DETAIL_SAMPLES" --up y --camdist "$DETAIL_CAMDIST" \
    --az_deg 270 --el_deg 9 --out_name detail_color.png \
    --param melanin --melanin "$PAPER_MELANIN" --melanin_red "$PAPER_MELANIN_RED" \
    --roughness "$PAPER_ROUGHNESS" --radial_rough "$PAPER_RADIAL_ROUGH" \
    --rand_rough 0.20 --rand_color 0.08 --radius_mul "$PAPER_RADIUS_MUL" \
    --skin "$PAPER_SKIN" --exposure "$PAPER_EXPOSURE" --view agx \
    --look "$PAPER_LOOK" --paper_light --camera_relative_light --transparent \
    > "$out/render_detail.log" 2>&1
  echo "teaser detail RGBA done: $animal"
done

echo "TEASER_COLOR_RGBA_DONE -> $OUTROOT"
