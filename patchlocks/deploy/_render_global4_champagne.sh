#!/usr/bin/env bash
# fig:qual re-render: 4 turntable views per animal from the FINAL v16 groom,
# in the SAME locked champagne melanin material as the teaser (geometry unchanged;
# camera pulled in 2.3 -> 2.05 so the animal fills more of the cell).
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; cd "$ROOT"
source patchlocks/deploy/paper_render_style.env
BLENDER="patchlocks/third_party/blender-4.5.10-linux-x64/blender"
SRC="${SRC:-v2_results/refine_v16_fluffy}"
OUTROOT="${OUTROOT:-$SRC/global4_champagne}"; mkdir -p "$OUTROOT"
ANIMALS="${ANIMALS:-beagle whiteTiger panda cat}"

declare -A MESH
MESH[beagle]=patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/beagle_dog/furless_reshaped.obj
MESH[whiteTiger]=patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/whiteTiger/furless_reshaped.obj
MESH[panda]=patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/panda/furless.obj
MESH[cat]=patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/cat/furless_reshaped.obj

for a in $ANIMALS; do
  OUT="$OUTROOT/$a"; mkdir -p "$OUT"
  "$BLENDER" -b --factory-startup --python patchlocks/deploy/render_dog_fur_pro.py -- \
    --fur "$SRC/$a/grow/surfpack_v2fur.npz" --mesh "${MESH[$a]}" --out "$OUT" \
    --views 4 --res 1280 --samples 32 --up y --camdist 2.05 \
    --param melanin --melanin "$PAPER_MELANIN" --melanin_red "$PAPER_MELANIN_RED" \
    --roughness "$PAPER_ROUGHNESS" --radial_rough "$PAPER_RADIAL_ROUGH" --rand_rough 0.20 --rand_color 0.08 \
    --radius_mul "$PAPER_RADIUS_MUL" --skin "$PAPER_SKIN" --exposure "$PAPER_EXPOSURE" --view agx \
    --look "$PAPER_LOOK" --paper_light --transparent \
    > "$OUT/global4.log" 2>&1
  echo "done $a"
done
echo "GLOBAL4_CHAMPAGNE_DONE -> $OUTROOT"
