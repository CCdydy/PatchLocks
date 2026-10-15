#!/usr/bin/env bash
# GenZoo baseline: feed-forward image -> SMAL+ mesh for the 4 animals, then
# render each mesh in the same frame as our fur (surface column of the grid).
# Needs (from MPI, in _baselines/genzoo/data/): genzoo_1M.ckpt, smal_plus.pkl.
set -euo pipefail
ROOT=/home/rog/Desktop/diffusionF; cd "$ROOT/_baselines/genzoo"
PY=/home/rog/miniconda3/envs/difflocks/bin/python
BL="$ROOT/patchlocks/third_party/blender-4.5.10-linux-x64/blender"

for f in data/genzoo_1M.ckpt data/smal_plus.pkl; do
  [ -f "$f" ] || { echo "MISSING $f -- download from GenZoo site first"; exit 1; }
done

# 1. feed-forward regression -> per-image obj (GenZoo writes output/obj/<name>.obj)
$PY inference_genzoo.py eg_input --checkpoint data/genzoo_1M.ckpt --output output
echo "=== GenZoo inference done ==="; ls output/obj/

# 2. render each GenZoo mesh into our per-animal frame (clay surface column)
for a in beagle whiteTiger panda cat; do
  read CTR SIZE < <($PY -c "import json;p=json.load(open('$ROOT/_baselines/frame_params.json'))['$a'];print('%f,%f,%f'%tuple(p['ctr']),p['size'])")
  mkdir -p "$ROOT/_baselines/surf_render/${a}_genzoo"
  for v in 0 1 2 3; do
    $BL -b --factory-startup --python "$ROOT/_baselines/render_surface.py" -- \
      --mesh "output/obj/${a}.obj" --out "$ROOT/_baselines/surf_render/${a}_genzoo" \
      --autoframe --up -y --camdist 2.05 --only_view $v --views 4 --res 1280 \
      --clay 0.70,0.68,0.62 \
      > "$ROOT/_baselines/surf_render/${a}_genzoo/render_v$v.log" 2>&1
  done
  echo "rendered GenZoo $a"
done
echo "GENZOO_DONE"
