#!/usr/bin/env bash
# fig:nfcompare re-render: hi-res view-1 of BOTH sides in the SAME locked
# champagne melanin material (identical treatment; only the strand sets differ).
# Replaces the washed flat material that hid fiber detail on both sides.
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; cd "$ROOT"
source patchlocks/deploy/paper_render_style.env
BLENDER="patchlocks/third_party/blender-4.5.10-linux-x64/blender"
NF="patchlocks/deploy/nf_official_render"
OURS="v2_results/refine_v16_fluffy"

declare -A MESH
MESH[beagle]=patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/beagle_dog/furless_reshaped.obj
MESH[whiteTiger]=patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/whiteTiger/furless_reshaped.obj
MESH[panda]=patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/panda/furless.obj
MESH[cat]=patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/cat/furless_reshaped.obj

render_one () {  # $1=fur npz  $2=mesh  $3=outdir
  mkdir -p "$3"
  "$BLENDER" -b --factory-startup --python patchlocks/deploy/render_dog_fur_pro.py -- \
    --fur "$1" --mesh "$2" --out "$3" \
    --only_view 1 --views 4 --res 2560 --samples 24 --up y --camdist 2.05 \
    --param melanin --melanin "$PAPER_MELANIN" --melanin_red "$PAPER_MELANIN_RED" \
    --roughness "$PAPER_ROUGHNESS" --radial_rough "$PAPER_RADIAL_ROUGH" --rand_rough 0.20 --rand_color 0.08 \
    --radius_mul "$PAPER_RADIUS_MUL" --length_mul "${4:-1.0}" --skin "$PAPER_SKIN" --exposure "$PAPER_EXPOSURE" --view agx \
    --look "$PAPER_LOOK" --paper_light --transparent \
    > "$3/render_champ_v1.log" 2>&1
}

for a in beagle whiteTiger panda cat; do
  render_one "$NF/nf500k_$a.npz"                 "${MESH[$a]}" "$NF/render_${a}_champ" 1.0
  echo "done NF $a"
  ours_length_mul=1.0
  if [[ "$a" == "panda" ]]; then ours_length_mul="${PANDA_OURS_LENGTH_MUL:-1.20}"; fi
  render_one "$OURS/$a/grow/surfpack_v2fur.npz"  "${MESH[$a]}" "$NF/render_${a}_champ_ours" "$ours_length_mul"
  echo "done OURS $a"
done
echo "NF_CHAMPAGNE_DONE"
