#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# LOCKED whole-animal pipeline for the 3 paper animals (beagle, whiteTiger, panda).
#
# Purpose: the diffusion is the ONLY thing that changes. When the new, stronger
# diffusion is trained, point the SWAP POINT below at it and run this script.
# Everything downstream (proposals -> region prior -> character -> swatch ->
# Step2 -> grow -> render) regenerates from the new model. The per-animal
# "characteristics" (region annotation incl. the white-tiger fix, local
# grow/tail/seam policy, density budget) are LOCKED and reused unchanged.
#
# What regenerates on swap (diffusion-derived): proposals_maps/, v2_region_prior,
#   character_prior, canon_swatch, step2/patch_params, grow/surfpack.
# What stays LOCKED (diffusion-independent): refine_annotations/<animal>.env
#   (region-detail/tail/seam policy) + <animal>.json (+ whiteTiger fixed labels),
#   surface patch packs, geopacks, meshes, multi-view data, TARGET_STRANDS.
# For v16 swaps, the same per-animal .env files also carry the approved
# character-consumption scale (V16_TARGET_ARCLEN / V16_STRAND_PROFILE_LIFT_SCALE).
# It is applied only when USE_V16_FINAL_RECIPE=1, or automatically when GEO_DIR
# / MODEL_PATH points at a v16 model. V10 reference runs remain unchanged.
# Set V16_RECIPE_PROFILE=v10_volume to recover the older V10-like outer coat
# volume while still using the v16 prior.
#
# Approved reference (V10 diffusion, refine_v4_light, 2026-06-28):
#   beagle      1,105,838 roots  arc_med 0.02511
#   whiteTiger  1,095,712 roots  arc_med 0.04070
#   panda       1,128,000 roots  arc_med 0.05007
# A correct swap should land near these root counts; arc_med will move with the
# new model's length character (that is the point of a stronger diffusion).
# =============================================================================

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"
PY="${PY:-python}"
BLENDER="${BLENDER:-patchlocks/third_party/blender-4.5.10-linux-x64/blender}"
WRAP="patchlocks/deploy/run_v2_character_mainline.sh"

# ============ DIFFUSION SWAP POINT — the only thing to change ============
# After training: set GEO_DIR + MODEL_PATH to the new model. The backend is inferred
# for locked model families so v16/MVFuser checkpoints do not accidentally run
# through the older V10 infer path.
# (Defaults below are the v16 paper checkpoint, so a bare run reproduces the paper grooms.
#  For the older V10 reference run: GEO_DIR=patchlocks/v2_locked/V10_PATCH_STABLE.)
export GEO_DIR="${GEO_DIR:-patchlocks/datasets/v16/e2a_cw15}"         # <-- new model dir
export MODEL_PATH="${MODEL_PATH:-$GEO_DIR/v14_best.pt}"               # <-- new checkpoint
if [[ -z "${DIFFUSION_BACKEND:-}" ]]; then
  if [[ "$GEO_DIR" == *"v16"* || "$MODEL_PATH" == *"v16"* ]]; then
    export DIFFUSION_BACKEND="v16"
  elif [[ "$GEO_DIR" == *"V10_PATCH_STABLE"* ]]; then
    export DIFFUSION_BACKEND="v10"
  else
    export DIFFUSION_BACKEND="custom"
  fi
else
  export DIFFUSION_BACKEND
fi
# DIFFUSION_BACKEND: v16 | v14 | v11_2b | v10 | custom
export USE_V16_FINAL_RECIPE="${USE_V16_FINAL_RECIPE:-auto}"      # auto | 0 | 1
export V16_RECIPE_PROFILE="${V16_RECIPE_PROFILE:-final}"          # final | v10_volume
# ========================================================================

OUTROOT="${OUTROOT:-$ROOT/outputs/refine_v16}"                      # never clobbers the archived paper grooms
export TARGET_STRANDS="${TARGET_STRANDS:-1200000}"                    # LOCKED paper density budget (NOT 7.9M)
ANIMALS="${ANIMALS:-beagle whiteTiger panda cat}"

# ---- per-animal INPUTS (structural, diffusion-independent) ----
declare -A PACK MESH GEOPACK DATA_DIR
PACK[beagle]=patchlocks/deploy/neuralfur_beagle_official/realviews/surfpack_beagle_f210_geopatch/surface_patch_dataset.npz
PACK[whiteTiger]=patchlocks/deploy/neuralfur_whiteTiger_official/realviews/surfpack_whiteTiger_f5_geopatch/surface_patch_dataset.npz
PACK[panda]=patchlocks/deploy/neuralfur_panda_official/panda_v2_surfpack.npz/surface_patch_dataset.npz
PACK[cat]=patchlocks/deploy/neuralfur_cat_official/realviews/surfpack_cat_f0_geopatch/surface_patch_dataset.npz

MESH[beagle]=patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/beagle_dog/furless_reshaped.obj
MESH[whiteTiger]=patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/whiteTiger/furless_reshaped.obj
MESH[panda]=patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/panda/furless.obj
MESH[cat]=patchlocks/datasets/neuralfur/official_results/extracted/neuralfur/cat/furless_reshaped.obj

GEOPACK[beagle]=patchlocks/deploy/neuralfur_beagle_geopatch_fine/geodesic_patchpack.npz
GEOPACK[whiteTiger]=patchlocks/deploy/neuralfur_whiteTiger_geopatch_fine/geodesic_patchpack.npz
GEOPACK[panda]=patchlocks/deploy/neuralfur_panda_geopatch_fine/geodesic_patchpack.npz
GEOPACK[cat]=patchlocks/deploy/neuralfur_cat_geopatch_fine/geodesic_patchpack.npz

# Step2 multi-view data (image-flow, honest no-oracle route). The wrapper defaults
# STEP2_FLOW_TARGET=image and auto-loads cameras from <DATA_DIR>/3d_gaussian_splatting/
# GS_/cameras/30000_matrices.pkl. All 3 animals therefore use real image-measured
# orientation, NOT a direction oracle.
DATA_DIR[beagle]=patchlocks/deploy/neuralfur_beagle_official/realviews/nf_data_f210
DATA_DIR[whiteTiger]=patchlocks/deploy/neuralfur_whiteTiger_official/realviews/nf_data_roaringwalk_f5
# panda: honest image-flow from the ARTEMIS multi-view capture (NOT the old face_flow oracle).
# This dir has images_2/, masks_2/{body,hair}/, orientations_2/{angles,vars}/ and the
# 30000_matrices.pkl camera dict — exactly what Step2's load_view_np + load_matrices expect.
DATA_DIR[panda]=patchlocks/datasets/neuralfur/extracted/data/Artemis/panda_processed_GH2/walk
# cat: honest image-flow from the multi-view capture (nf_data_walk_final_f0 has
# images_2/, orientations_2/ and 3d_gaussian_splatting/.../30000_matrices.pkl).
DATA_DIR[cat]=patchlocks/deploy/neuralfur_cat_official/realviews/nf_data_walk_final_f0

for a in $ANIMALS; do
  echo "######################## $a ########################"
  # RECIPE_SUFFIX selects a recipe variant, e.g. RECIPE_SUFFIX=_paper_smooth -> panda_paper_smooth.env
  ENVF="patchlocks/deploy/refine_annotations/${a}${RECIPE_SUFFIX:-}.env"
  [[ -f "$ENVF" ]] || { echo "ERROR: missing $ENVF"; exit 2; }
  # subshell so each animal's locked .env (region-detail/tail policy) cannot leak
  (
    set -a
    source "$ENVF"                       # locked per-animal grow policy + INDIVIDUAL_ANNOTATION_JSON
    use_v16_recipe="$USE_V16_FINAL_RECIPE"
    if [[ "$use_v16_recipe" == "auto" ]]; then
      use_v16_recipe=0
      if [[ "$GEO_DIR" == *"v16"* || "$MODEL_PATH" == *"v16"* ]]; then
        use_v16_recipe=1
      fi
    fi
    if [[ "$use_v16_recipe" == "1" ]]; then
      case "$V16_RECIPE_PROFILE" in
        final)
          if [[ -n "${V16_TARGET_ARCLEN:-}" ]]; then
            TARGET_ARCLEN="$V16_TARGET_ARCLEN"
          fi
          if [[ -n "${V16_STRAND_PROFILE_LIFT_SCALE:-}" ]]; then
            STRAND_PROFILE_LIFT_SCALE="$V16_STRAND_PROFILE_LIFT_SCALE"
          fi
          ;;
        v10_volume)
          if [[ -n "${V16_V10_VOLUME_TARGET_ARCLEN:-}" ]]; then
            TARGET_ARCLEN="$V16_V10_VOLUME_TARGET_ARCLEN"
          elif [[ -n "${V16_TARGET_ARCLEN:-}" ]]; then
            TARGET_ARCLEN="$V16_TARGET_ARCLEN"
          fi
          if [[ -n "${V16_V10_VOLUME_LIFT_SCALE:-}" ]]; then
            STRAND_PROFILE_LIFT_SCALE="$V16_V10_VOLUME_LIFT_SCALE"
          elif [[ -n "${V16_STRAND_PROFILE_LIFT_SCALE:-}" ]]; then
            STRAND_PROFILE_LIFT_SCALE="$V16_STRAND_PROFILE_LIFT_SCALE"
          fi
          ;;
        *)
          echo "ERROR: unknown V16_RECIPE_PROFILE=$V16_RECIPE_PROFILE (expected final|v10_volume)" >&2
          exit 2
          ;;
      esac
      # v16 length is set by target_arclen ALONE. The footprint scale-contract must
      # not also multiply length, or the final median = target * footprint(~3x).
      SCALE_CONTRACT_LENGTH_EXP="${SCALE_CONTRACT_LENGTH_EXP:-0}"
      echo "  v16 recipe[$V16_RECIPE_PROFILE]: target_arclen=${TARGET_ARCLEN:-unset} lift=${STRAND_PROFILE_LIFT_SCALE:-unset} length_exp=${SCALE_CONTRACT_LENGTH_EXP}"
    else
      echo "  v16 final recipe: disabled"
    fi
    set +a
    # Pass per-animal inputs as inline command-prefix env vars: bash cannot export
    # associative arrays to the child, so reassigning PACK=... in the shell would
    # reach the WRAP empty. The prefix form reads the array element here and exports
    # a plain scalar to the child.
    OUTDIR_A="$OUTROOT/$a${RECIPE_SUFFIX:-}"
    mkdir -p "$OUTDIR_A"
    PACK="${PACK[$a]}" OUTDIR="$OUTDIR_A" GEOPACK="${GEOPACK[$a]}" \
      DATA_DIR="${DATA_DIR[$a]}" MESH="${MESH[$a]}" bash "$WRAP"
  )
  echo "  -> $OUTROOT/$a${RECIPE_SUFFIX:-}/grow/surfpack_v2fur.npz"
done

# ---- close-up DETAIL renders (the paper output spec) ----
DET="$OUTROOT/details"; mkdir -p "$DET"
for a in $ANIMALS; do
  OUT="$DET/$a${RECIPE_SUFFIX:-}"; mkdir -p "$OUT"
  for v in 1 3; do
    "$BLENDER" -b --factory-startup --python patchlocks/deploy/render_dog_fur_pro.py -- \
      --fur "$OUTROOT/$a${RECIPE_SUFFIX:-}/grow/surfpack_v2fur.npz" --mesh "${MESH[$a]}" --out "$OUT" \
      --only_view $v --views 4 --res 1536 --samples 128 --up y --camdist 0.5 \
      --flat 0.78,0.78,0.74 --spec 0.22 --skin 0.66,0.66,0.66 --transparent \
      --out_name "${a}_detail_v${v}.png" > "$OUT/detail_v$v.log" 2>&1
  done
  echo "  rendered detail: $a"
done

# ---- full-body side renders (paper/QA overview; kept separate from close-up detail) ----
SIDE="$OUTROOT/side_full"; mkdir -p "$SIDE"
for a in $ANIMALS; do
  OUT="$SIDE/$a${RECIPE_SUFFIX:-}"; mkdir -p "$OUT"
  for v in 1 3; do
    "$BLENDER" -b --factory-startup --python patchlocks/deploy/render_dog_fur_pro.py -- \
      --fur "$OUTROOT/$a${RECIPE_SUFFIX:-}/grow/surfpack_v2fur.npz" --mesh "${MESH[$a]}" --out "$OUT" \
      --only_view $v --views 4 --res "${FULL_SIDE_RES:-1536}" --samples "${FULL_SIDE_SAMPLES:-40}" \
      --up y --camdist "${FULL_SIDE_CAMDIST:-2.3}" \
      --flat 0.78,0.78,0.74 --spec 0.22 --skin 0.66,0.66,0.66 --transparent \
      --out_name "${a}_side_full_v${v}.png" > "$OUT/side_v$v.log" 2>&1
  done
  echo "  rendered full sides: $a"
done
echo "LOCKED_3ANIMALS_DONE -> $OUTROOT"
