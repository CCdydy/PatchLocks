#!/usr/bin/env bash
# S4: does reconstruction cost grow with the final strand count?
# The FIT (Step2) operates on per-patch variables and a fixed proxy sample -> independent of strand count.
# The GROW instantiates strands -> scales. Measure the grow at several budgets, reusing all upstream artifacts.
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT"
PY="${PY:-python}"
SRC=v2_results/refine_v16_4animals/beagle
OUT=v2_results/s4_scaling; mkdir -p $OUT
set -a; source patchlocks/deploy/refine_annotations/beagle.env; set +a
for N in 300000 600000 1200000 2400000; do
  S=$(date +%s.%N)
  $PY patchlocks/deploy/grow_v2_region_prior.py \
    --prior "$SRC/v2_region_prior.annotated.npz" \
    --strand_profile_npz "$SRC/canon_swatch_reliable_straight.npz" \
    --radius_profile_npz "$SRC/canon_swatch_reliable_straight.npz" \
    --out "$OUT/n$N" --target_arclen 0.01269999984651804 --target_strands $N \
    --scale_contract patch_window_footprint --scale_contract_density_exp 0.5 \
    --scale_contract_length_exp 0 --scale_contract_radius_exp 0.0 \
    --length_mode region_smooth --length_region_blend 0.75 \
    --region_detail_policy "${REGION_DETAIL_POLICY:-generic}" \
    --patch_params "$SRC/step2/patch_params.npz" --patch_param_vars flow,length \
    >/dev/null 2>&1
  E=$(date +%s.%N)
  R=$($PY -c "import numpy as np;d=np.load('$OUT/n$N/surfpack_v2fur.npz',mmap_mode='r');print(d['strands'].shape[0])")
  echo "  target=$N  actual_roots=$R  grow_time=$(echo "$E - $S" | bc)s"
done
