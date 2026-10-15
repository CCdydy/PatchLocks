#!/usr/bin/env bash
set -euo pipefail

# Build the full V11 long/curly/fluffy dataset.
# This script only builds data.  It does not launch diffusion training.
#
# Default:
#   1600 generated 3D patches
#   4 rendered views
#   rotaug naug=1 -> 2 single-view copies per rendered view
#   about 12.8k single-view training dirs

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"
BL="${BL:-patchlocks/third_party/blender-4.5.10-linux-x64/blender}"
OUT="${OUT:-patchlocks/datasets/v11_longfur}"
NTOT="${NTOT:-1600}"
ROT_NAUG="${ROT_NAUG:-1}"
ROT_MAX_DEG="${ROT_MAX_DEG:-35}"
RENDER_RES="${RENDER_RES:-256}"
RENDER_SAMPLES="${RENDER_SAMPLES:-56}"
CODEC="${CODEC:-patchlocks/models/fur_strand_codec_v6upright.pt}"
LATENT="${LATENT:-target_latent64_v6codec.npz}"

cd "$ROOT"
mkdir -p "$OUT/patches" "$OUT/renders_mv" "$OUT/renders_sv_rotaug" "$OUT/logs"

COATS=(ultra_short short_dense short_double_standoff wiry wavy curly long_golden long_silky fluffy_double)
COUNTS=(80 112 96 96 208 256 272 208 272)  # sums to 1600, matches V11 COAT_P.
SCHEMES=(solid two_tone bicolor tabby tipped)

sum=0
for c in "${COUNTS[@]}"; do sum=$((sum + c)); done
if [[ "$sum" -ne "$NTOT" ]]; then
  echo "ERROR: COUNTS sum $sum != NTOT $NTOT. Adjust counts or set NTOT=1600." >&2
  exit 2
fi

echo "== V11 full dataset =="
echo "OUT=$OUT NTOT=$NTOT rotaug=$ROT_NAUG render_res=$RENDER_RES"

idx=0
for ci in "${!COATS[@]}"; do
  coat="${COATS[$ci]}"
  count="${COUNTS[$ci]}"
  for j in $(seq 0 $((count - 1))); do
    scheme="${SCHEMES[$((j % ${#SCHEMES[@]}))]}"
    dst="$OUT/patches/rd_$(printf '%04d' "$idx")_${coat}_${scheme}.npz"
    if [[ ! -f "$dst" ]]; then
      "$PY" patchlocks/v11_longfur/gen_realdog_v11.py \
        --seed $((210000 + idx)) \
        --coat "$coat" \
        --color_scheme "$scheme" \
        --out "$dst" >/dev/null
    fi
    idx=$((idx + 1))
  done
done
echo "generated patches: $(find "$OUT/patches" -maxdepth 1 -name '*.npz' | wc -l)"

echo "== render multi-view =="
THR=$(( $(nproc) / 2 )); [[ "$THR" -lt 1 ]] && THR=1
for shard in 0 1; do
  "$BL" -b -t "$THR" --factory-startup --python patchlocks/data_gen/photoreal_render_loop_v8.py -- \
    --patches "$OUT/patches" \
    --out "$OUT/renders_mv" \
    --views 4 \
    --res "$RENDER_RES" \
    --samples "$RENDER_SAMPLES" \
    --shard "$shard/2" \
    --seed0 $((310000 + shard * 100000)) \
    > "$OUT/logs/render_${shard}.log" 2>&1 &
done
wait
echo "render dirs: $(find "$OUT/renders_mv" -mindepth 1 -maxdepth 1 -type d | wc -l)"
# MV_ONLY=1: stop here. The paper trains on renders_mv; the single-view rotaug copies below
# belong to an earlier single-view model. Multi-view targets are baked by bake_mv_targets.py.
if [[ "${MV_ONLY:-0}" == "1" ]]; then echo "MV_ONLY=1: done after multi-view renders"; exit 0; fi

echo "== explode single-view with physical in-plane rotaug =="
rm -rf "$OUT/renders_sv_rotaug"
"$PY" patchlocks/data_gen/explode_singleview_rotaug.py \
  "$OUT/renders_mv" \
  "$OUT/renders_sv_rotaug" \
  --naug "$ROT_NAUG" \
  --max_deg "$ROT_MAX_DEG" \
  --seed 11 | tee "$OUT/logs/explode_rotaug.log"

echo "== bake codec z64 target on rotaug single-view dirs =="
"$PY" patchlocks/models/bake_latent_batch.py \
  "$OUT/renders_sv_rotaug" 64 \
  --codec "$CODEC" \
  --out-name "$LATENT" | tee "$OUT/logs/bake_latent.log"

echo "== build field_gt on rotaug single-view dirs =="
"$PY" patchlocks/data_gen/build_field_gt.py "$OUT/renders_sv_rotaug" | tee "$OUT/logs/build_field_gt.log"

echo "DONE V11 full dataset"
echo "renders for training: $OUT/renders_sv_rotaug"
echo "latent name: $LATENT"
