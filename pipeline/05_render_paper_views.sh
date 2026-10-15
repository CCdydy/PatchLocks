#!/usr/bin/env bash
# Stage 5 — render grooms with the paper's display material ("champagne" Hair BSDF) and cameras.
#
#   bash pipeline/05_render_paper_views.sh                    # grooms from stage 4 (outputs/refine_v16)
#   SRC=v2_results/refine_v16_fluffy bash pipeline/05_render_paper_views.sh   # the archived paper grooms
#
# Writes 4-view turntables and two close-ups per animal to outputs/renders/<SRC name>/.
# Rendering an archived groom with these settings reproduces the paper renders pixel for pixel.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export SRC="${SRC:-outputs/refine_v16}"
export ANIMALS="${ANIMALS:-beagle whiteTiger panda cat}"
for a in $ANIMALS; do require "$SRC/$a/grow/surfpack_v2fur.npz" "run stage 4 first, or set SRC=v2_results/refine_v16_fluffy"; done
OUT="outputs/renders/$(basename "$SRC")"

step "4-view turntables -> $OUT/global4_champagne"
OUTROOT="$OUT/global4_champagne" bash patchlocks/deploy/_render_global4_champagne.sh

step "close-ups -> $OUT/details_champagne"
OUTROOT="$OUT/details_champagne" bash patchlocks/deploy/_render_details_champagne.sh
