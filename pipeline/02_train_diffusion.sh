#!/usr/bin/env bash
# Stage 2 (optional) — train the paper diffusion model (v16: multi-view EDM + clump-channel weighting).
#
#   bash pipeline/02_train_diffusion.sh           # -> outputs/train/v16_e2a_cw15/v14_best.pt
#
# Settings equal the saved arguments of the paper checkpoint (seed 0, CLUMP_CW=1.5, no bucket
# sampler, early stopping on the composite score; the paper checkpoint is the iteration-15000
# best). cuDNN determinism is not enforced, so a retrained model is close to, not bit-identical
# with, the released one. To use it downstream:
#   GEO_DIR=outputs/train/v16_e2a_cw15 bash pipeline/04_reconstruct_animals.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export DATA="${DATA:-patchlocks/datasets/v11_longfur/renders_mv}"
export OUT="${OUT:-outputs/train/v16_e2a_cw15}"
require "$DATA"
require patchlocks/models/fur_strand_codec_v6upright.pt

step "train v16 on $DATA -> $OUT"
CLUMP_CW=1.5 CLUMP_BUCKET=0 SEED=0 bash patchlocks/v16/launch_train_v16.sh
