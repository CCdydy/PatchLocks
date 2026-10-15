#!/usr/bin/env bash
# Frozen independent test set B: 160 patches, train-matched coat proportions, disjoint seeds.
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT"
PY="${PY:-python}"
OUT="${OUT:-outputs/testsets/testB_patches}"   # the frozen set used in the paper is patchlocks/experiments/rebuttal_2027/testB_patches; never regenerate it in place
mkdir -p "$OUT"
SCHEMES=(solid two_tone bicolor tabby tipped)
# coat:count  (COUNTS/10 from build_full_dataset.sh, rounded to 160 total)
declare -A COUNTS=( [ultra_short]=8 [short_dense]=11 [short_double_standoff]=10 [wiry]=10 \
                    [wavy]=21 [curly]=26 [long_golden]=27 [long_silky]=21 [fluffy_double]=26 )
idx=0
for coat in ultra_short short_dense short_double_standoff wiry wavy curly long_golden long_silky fluffy_double; do
  n=${COUNTS[$coat]}
  for ((j=0;j<n;j++)); do
    seed=$((610000+idx)); scheme=${SCHEMES[$((idx%5))]}
    name=$(printf "rd_%04d_%s_%s" $idx "$coat" "$scheme")
    $PY patchlocks/v11_longfur/gen_realdog_v11.py --seed $seed --coat "$coat" \
        --color_scheme "$scheme" --out "$OUT/$name.npz" >/dev/null
    idx=$((idx+1))
  done
  echo "coat $coat done ($idx total)"
done
echo "TESTB_PATCHES_DONE n=$idx"
