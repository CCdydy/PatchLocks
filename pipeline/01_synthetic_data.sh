#!/usr/bin/env bash
# Stage 1 (optional) — regenerate the synthetic training corpus.
#
#   bash pipeline/01_synthetic_data.sh            # -> outputs/data/v11_longfur/renders_mv
#
# The released corpus (data/manifest.txt: patchlocks/datasets/v11_longfur/renders_mv) is what the
# paper model was trained on; this stage rebuilds it from scratch:
#   1. 1,600 procedural fur patches, nine coat archetypes, seeds 210000+ (deterministic)
#   2. 4-view Blender renders, 256 px, 56 samples (Cycles GPU; not bit-exact)
#   3. z64 strand latents + field targets with the strand codec
#   4. per-view pose features
# Several hours on one GPU. The frozen test sets are NOT regenerated here (see docs/DATA.md).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
OUT="${OUT:-outputs/data/v11_longfur}"
CODEC="${CODEC:-patchlocks/models/fur_strand_codec_v6upright.pt}"
require "$CODEC"
require patchlocks/third_party/blender-4.5.10-linux-x64/blender

step "1-2. generate patches and render 4 views -> $OUT"
OUT="$OUT" MV_ONLY=1 bash patchlocks/v11_longfur/build_full_dataset.sh

step "3. bake z64 latents and field targets"
"$PY" patchlocks/data_gen/bake_mv_targets.py "$OUT/renders_mv" \
  --codec "$CODEC" --out-name target_latent64_v6codec.npz --H 64

step "4. pose features"
"$PY" patchlocks/data_gen/prepare_v13_mv_dataset.py --renders "$OUT/renders_mv" \
  --out "$OUT/v13_mv_prep" --latent-name target_latent64_v6codec.npz --res 64 --write-pose

echo "done: $OUT/renders_mv  (train on it with DATA=$OUT/renders_mv bash pipeline/02_train_diffusion.sh)"
