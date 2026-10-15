#!/usr/bin/env bash
# Stage 4 — reconstruct the animals from their multi-view captures (the paper's whole-animal results).
#
#   bash pipeline/04_reconstruct_animals.sh                       # beagle whiteTiger panda cat
#   ANIMALS=beagle bash pipeline/04_reconstruct_animals.sh
#   GEO_DIR=outputs/train/v16_e2a_cw15 bash pipeline/04_reconstruct_animals.sh   # a retrained model
#
# Per animal (patchlocks/deploy/run_v2_character_mainline.sh):
#   Step 1   diffusion: multi-view patch predictions            infer_v14_surfpack_proposals.py
#   Step 1b  per-patch density / region prior                   build_v2_region_prior.py
#   Step 1c  direction-free fur character                       extract_fur_character.py
#   Step 1d  canonical character swatch                         build_canon_swatch.py
#   Step 2   multi-view orientation + length fit (12 views)     patch_graph_opt_geo.py
#   Step 3   surface growth, 1.2M-root budget                   grow_v2_region_prior.py
#   Render   QA turntable + close-ups                           render_dog_fur_pro.py (Blender)
# Each animal's recipe lives in patchlocks/deploy/refine_annotations/<animal>.env.
# ~15 min per animal + ~20 min of close-up rendering (RTX PRO 6000).
# Output: outputs/refine_v16/<animal>/grow/surfpack_v2fur.npz (+ render/, details/, side_full/).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export OUTROOT="${OUTROOT:-outputs/refine_v16}"
export ANIMALS="${ANIMALS:-beagle whiteTiger panda cat}"
require "${MODEL_PATH:-${GEO_DIR:-patchlocks/datasets/v16/e2a_cw15}/v14_best.pt}"
for a in $ANIMALS; do require "patchlocks/deploy/neuralfur_${a}_official"; done

step "reconstruct: $ANIMALS -> $OUTROOT"
bash patchlocks/deploy/run_locked_3animals.sh
