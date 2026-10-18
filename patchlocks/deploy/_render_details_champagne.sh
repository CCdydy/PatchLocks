#!/usr/bin/env bash
# fig:detail re-render: the SAME two close-up cameras (views 1,3, camdist 0.5)
# per animal, in the locked champagne melanin material — unifies the figure
# family (teaser / qual / nfcompare / detail all share one display material).
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; cd "$ROOT"
source patchlocks/deploy/paper_render_style.env
BLENDER="patchlocks/third_party/blender-4.5.10-linux-x64/blender"
SRC="${SRC:-v2_results/refine_v16_fluffy}"
OUTROOT="${OUTROOT:-outputs/renders/details_champagne}"   # never write into the (linked) archive
mkdir -p "$OUTROOT"
ANIMALS="${ANIMALS:-beagle whiteTiger panda cat}"
DETAIL_VIEWS="${DETAIL_VIEWS:-1 3}"

declare -A MESH
MESH[beagle]=patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/beagle_dog/furless_reshaped.obj
MESH[whiteTiger]=patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/whiteTiger/furless_reshaped.obj
MESH[panda]=patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/panda/furless.obj
MESH[cat]=patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/cat/furless_reshaped.obj

# Other animals (docs/NEW_ANIMAL.md): mesh and vertical axis come from patchlocks/deploy/inputs/<animal>/
mesh_of() { [[ -n "${MESH[$1]:-}" ]] && echo "${MESH[$1]}" || echo "patchlocks/deploy/inputs/$1/furless.obj"; }
up_of() { local f="patchlocks/deploy/inputs/$1/up.txt"; [[ -f "$f" ]] && tr -d '[:space:]' < "$f" || echo y; }

for a in $ANIMALS; do
  OUT="$OUTROOT/$a"; mkdir -p "$OUT"
  for v in $DETAIL_VIEWS; do
    "$BLENDER" -b --factory-startup --python patchlocks/deploy/render_dog_fur_pro.py -- \
      --fur "$SRC/$a/grow/surfpack_v2fur.npz" --mesh "$(mesh_of "$a")" --out "$OUT" \
      --only_view $v --views 4 --res 1536 --samples 24 --up="$(up_of "$a")" --camdist 0.5 \
      --param melanin --melanin "$PAPER_MELANIN" --melanin_red "$PAPER_MELANIN_RED" \
      --roughness "$PAPER_ROUGHNESS" --radial_rough "$PAPER_RADIAL_ROUGH" --rand_rough 0.20 --rand_color 0.08 \
      --radius_mul "$PAPER_RADIUS_MUL" --skin "$PAPER_SKIN" --exposure "$PAPER_EXPOSURE" --view agx \
      --look "$PAPER_LOOK" --paper_light --transparent \
      --out_name "${a}_detail_v${v}.png" > "$OUT/detail_v$v.log" 2>&1
  done
  echo "done $a"
done
echo "DETAILS_CHAMPAGNE_DONE"
