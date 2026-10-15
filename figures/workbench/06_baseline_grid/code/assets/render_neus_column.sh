#!/usr/bin/env bash
# Render the NeuS baseline column (NF's released neus.obj) for all 4 animals,
# 4 turntable views each, in the SAME frame as our fur results (clay surface).
set -euo pipefail
ROOT=/home/rog/Desktop/diffusionF; cd "$ROOT"
BL=patchlocks/third_party/blender-4.5.10-linux-x64/blender
PY=/home/rog/miniconda3/envs/difflocks/bin/python
NFB=patchlocks/datasets/neuralfur/official_results/extracted/neuralfur
declare -A DIR=( [beagle]=beagle_dog [whiteTiger]=whiteTiger [panda]=panda [cat]=cat )

for a in beagle whiteTiger panda cat; do
  read CTR SIZE < <($PY -c "import json;p=json.load(open('_baselines/frame_params.json'))['$a'];print('%f,%f,%f'%tuple(p['ctr']),p['size'])")
  for v in 0 1 2 3; do
    $BL -b --factory-startup --python _baselines/render_surface.py -- \
      --mesh "$NFB/${DIR[$a]}/neus.obj" --out "_baselines/surf_render/${a}_neus" \
      --center "$CTR" --size "$SIZE" --camdist 2.05 --only_view $v --views 4 --res 1280 \
      > "_baselines/surf_render/${a}_neus/render_v$v.log" 2>&1
  done
  echo "done NeuS $a"
done
echo "NEUS_COLUMN_DONE"
