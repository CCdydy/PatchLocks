#!/usr/bin/env bash
set -euo pipefail

# V2-A character mainline entrypoint (the locked whole-animal route, end-to-end).
#
# Three-way split:
#   diffusion (learned prior) = "what fur quality is here"  (character)
#   V1-style grow             = "how to grow it across the surface"  (distribution)
#   Step2 (per-animal MV fit) = "direction flow to fit photos"
#
# Steps:
#   1   stable V10 -> per-patch 3D fur proposals (+maps)        infer_v10_surfpack_proposals.py
#   1b  per-patch / per-region density+footprint prior          build_v2_region_prior.py
#   1c  decode 3D, canonicalize, extract direction-free scalars extract_fur_character.py
#   1d  synthesize one clean V1-style representative swatch      build_canon_swatch.py
#   2   multi-view direction fit (NO oracle; length is opt-in)   patch_graph_opt_geo.py
#   3   V1-style grow: PoU roots, density, swatch, scale, Step2  grow_v2_region_prior.py
#   R   visual QA render                                         render_dog_fur_pro.py
#
# Required:
#   PACK=<surface_patch_dataset.npz>   OUTDIR=<run_dir>
# Optional (enable Step2 / visual QA render):
#   GEOPACK=<geodesic_patchpack.npz>   DATA_DIR=<multi-view nf_data>   VIEWS=<csv>   MESH=<furless.obj>
# Step2 and render auto-run only when their inputs are provided; otherwise this
# produces a geometry-only init (no multi-view direction fit).

ROOT="${ROOT:-/home/rog/Desktop/diffusionF}"
PY="${PY:-/home/rog/miniconda3/envs/difflocks/bin/python}"
PACK="${PACK:-}"
OUTDIR="${OUTDIR:-}"

if [[ -z "$PACK" || -z "$OUTDIR" ]]; then
  echo "usage: PACK=<surface_patch_dataset.npz> OUTDIR=<run_dir> $0" >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$OUTDIR"

DIFFUSION_BACKEND="${DIFFUSION_BACKEND:-}"
if [[ -z "$DIFFUSION_BACKEND" ]]; then
  if [[ "${GEO_DIR:-}" == *"v16"* || "${MODEL_PATH:-}" == *"v16"* ]]; then
    DIFFUSION_BACKEND="v16"
  elif [[ "${GEO_DIR:-}" == *"v14"* || "${MODEL_PATH:-}" == *"v14"* ]]; then
    DIFFUSION_BACKEND="v14"
  elif [[ "${GEO_DIR:-}" == *"V10_PATCH_STABLE"* ]]; then
    DIFFUSION_BACKEND="v10"
  elif [[ -n "${GEO_DIR:-}" || -n "${MODEL_PATH:-}" ]]; then
    DIFFUSION_BACKEND="custom"
  else
    DIFFUSION_BACKEND="v10"
  fi
fi

# INFER_SCRIPT = which Step1 inference to run; PROPOSAL_NAME = the proposals filename it writes.
# V14-family (multi-view MVFuser, full-mode z64) backends use infer_v14_surfpack_proposals.py and write
# v14_surfpack_proposals.npz; older single-view backends use infer_v10 and write v10_surfpack_proposals.npz.
INFER_SCRIPT="patchlocks/deploy/infer_v10_surfpack_proposals.py"
PROPOSAL_NAME="v10_surfpack_proposals.npz"
case "$DIFFUSION_BACKEND" in
  v16)
    # V14-family MULTI-VIEW model (MVFuser pose/raydir tokens, full-mode z64, sigma_max=80, CFG, EMA).
    GEO_DIR="${GEO_DIR:-patchlocks/datasets/v16/e2a_cw15}"
    MODEL_PATH="${MODEL_PATH:-$GEO_DIR/v14_best.pt}"
    DIFFUSION_LABEL="V16 clump-lite multi-view character diffusion"
    INFER_SCRIPT="patchlocks/deploy/infer_v14_surfpack_proposals.py"
    PROPOSAL_NAME="v14_surfpack_proposals.npz"
    ;;
  v14)
    GEO_DIR="${GEO_DIR:-patchlocks/datasets/v14/e1_full_sm80}"
    MODEL_PATH="${MODEL_PATH:-$GEO_DIR/v14_best.pt}"
    DIFFUSION_LABEL="V14 multi-view character diffusion"
    INFER_SCRIPT="patchlocks/deploy/infer_v14_surfpack_proposals.py"
    PROPOSAL_NAME="v14_surfpack_proposals.npz"
    ;;
  v11_2b)
    GEO_DIR="${GEO_DIR:-patchlocks/datasets/stage3_v11_2b_paramloss}"
    MODEL_PATH="${MODEL_PATH:-$GEO_DIR/stage3_model_best.pt}"
    DIFFUSION_LABEL="V11.2b@10k character diffusion"
    ;;
  v10)
    GEO_DIR="${GEO_DIR:-patchlocks/datasets/V10_PATCH_STABLE}"
    MODEL_PATH="${MODEL_PATH:-$GEO_DIR/stage3_model.pt}"
    DIFFUSION_LABEL="stable V10 character diffusion"
    ;;
  custom)
    GEO_DIR="${GEO_DIR:-}"
    MODEL_PATH="${MODEL_PATH:-${GEO_DIR:+$GEO_DIR/stage3_model.pt}}"
    DIFFUSION_LABEL="custom character diffusion"
    # custom may be a V14-family ckpt; allow override of the infer script + proposals name.
    INFER_SCRIPT="${INFER_SCRIPT_OVERRIDE:-$INFER_SCRIPT}"
    PROPOSAL_NAME="${PROPOSAL_NAME_OVERRIDE:-$PROPOSAL_NAME}"
    ;;
  *)
    echo "ERROR: unknown DIFFUSION_BACKEND=$DIFFUSION_BACKEND (expected v16, v14, v11_2b, v10, custom)" >&2
    exit 2
    ;;
esac

if [[ -z "$GEO_DIR" || -z "$MODEL_PATH" ]]; then
  echo "ERROR: GEO_DIR and MODEL_PATH must be set for DIFFUSION_BACKEND=$DIFFUSION_BACKEND" >&2
  exit 2
fi
STEP1_DIR="${STEP1_DIR:-$OUTDIR/proposals_maps}"
PROPOSAL_NPZ="$STEP1_DIR/$PROPOSAL_NAME"   # v10_ or v14_ proposals depending on backend (set above)
REGION_PRIOR="${REGION_PRIOR:-$OUTDIR/v2_region_prior.npz}"
CHARACTER_PRIOR="${CHARACTER_PRIOR:-$OUTDIR/character_prior.npz}"
CHARACTER_REGION_PRIOR="${CHARACTER_REGION_PRIOR:-$OUTDIR/v2_region_prior_character_field.npz}"
SWATCH="${SWATCH:-$OUTDIR/canon_swatch_reliable_straight.npz}"
GROW_OUT="${GROW_OUT:-$OUTDIR/grow}"
PATCH_PARAMS="${PATCH_PARAMS:-}"
GEOPACK="${GEOPACK:-}"
MATRICES="${MATRICES:-}"
# Step2 (multi-view fit) inputs — Step2 runs only if DATA_DIR+GEOPACK given and PATCH_PARAMS unset.
DATA_DIR="${DATA_DIR:-}"
VIEWS="${VIEWS:-0000,0003,0006,0009,0012,0015,0018,0021,0024,0027,0030,0033}"
STEP2_OUT="${STEP2_OUT:-$OUTDIR/step2}"
# Visual QA render inputs — render runs only if MESH given.
MESH="${MESH:-}"
BLENDER="${BLENDER:-patchlocks/third_party/blender-4.5.10-linux-x64/blender}"
RENDER_OUT="${RENDER_OUT:-$OUTDIR/render}"

for f in "$MODEL_PATH" "$GEO_DIR/b3_fuser.pt"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing diffusion file: $f" >&2
    exit 2
  fi
done

# Stable V10 identity check. V11.2b/custom runs are explicitly model-path based
# and are not checked against the old V10 hashes.
if [[ "$DIFFUSION_BACKEND" == "v10" ]]; then
  expected_model="5c23ce4345a14511499af5b83789f60e64e3d688d32443e812a9ae96e75ff3ab"
  expected_b3="56335f4046cdd3514a20813ad4beea31aa7c02d4a4f79ce9b486c2d55ac5e674"
  got_model="$(sha256sum "$MODEL_PATH" | awk '{print $1}')"
  got_b3="$(sha256sum "$GEO_DIR/b3_fuser.pt" | awk '{print $1}')"
  if [[ "$got_model" != "$expected_model" || "$got_b3" != "$expected_b3" ]]; then
    echo "ERROR: $GEO_DIR / $MODEL_PATH is not the locked V10_PATCH_STABLE checkpoint." >&2
    exit 3
  fi
fi

echo "== Step1: $DIFFUSION_LABEL proposals with maps ($INFER_SCRIPT) =="
reuse_step1=0
if [[ "${REUSE_STEP1:-0}" == "1" && -f "$PROPOSAL_NPZ" && -f "$STEP1_DIR/density_table.txt" ]]; then
  if "$PY" - "$PROPOSAL_NPZ" "$MODEL_PATH" <<'PY'
import os, sys, numpy as np
npz, model = sys.argv[1], sys.argv[2]
try:
    d = np.load(npz, allow_pickle=True)
    saved = str(d["model"]) if "model" in d.files else ""
except Exception:
    saved = ""
ok = bool(saved) and os.path.realpath(saved) == os.path.realpath(model)
raise SystemExit(0 if ok else 1)
PY
  then
    reuse_step1=1
  else
    echo "existing Step1 proposals were made by a different model; rerun Step1"
  fi
fi
if [[ "$reuse_step1" == "1" ]]; then
  echo "reuse existing Step1 proposals: $PROPOSAL_NPZ"
else
  infer_args=(
    "$INFER_SCRIPT"
    --pack "$PACK"
    --geo_dir "$GEO_DIR"
    --model "$MODEL_PATH"
    --out "$STEP1_DIR"
    --save_maps
    --crop_mode comfort_window
    --comfort_crop_px "${COMFORT_CROP_PX:-256}"
    --view_select scale
    --steps "${STEPS:-40}"
    --seed "${SEED:-0}"
  )
  if [[ -n "$MATRICES" ]]; then
    infer_args+=(--matrices "$MATRICES")
  fi
  "$PY" "${infer_args[@]}"
fi

echo "== Step1b: region prior =="
region_args=(
  patchlocks/deploy/build_v2_region_prior.py
  --table "$STEP1_DIR/density_table.txt" \
  --proposal_npz "$PROPOSAL_NPZ" \
  --pack "$PACK" \
  --out "$REGION_PRIOR"
)
if [[ "${REGION_PRIOR_DROP_Z64_MAP:-1}" == "1" ]]; then
  region_args+=(--drop_z64_map)
fi
if [[ "${REGION_PRIOR_DROP_HEAVY_MAPS:-1}" == "1" ]]; then
  region_args+=(--drop_heavy_maps)
fi
"$PY" "${region_args[@]}"

echo "== Step1c: extract direction-free/color-free fur character =="
char_args=(
  patchlocks/deploy/extract_fur_character.py
  --proposals "$PROPOSAL_NPZ"
  --pack "$PACK"
  --out "$CHARACTER_PRIOR"
  --reliable_pct "${CHARACTER_RELIABLE_PCT:-80}"
  --field_k "${CHARACTER_FIELD_K:-64}"
  --field_iters "${CHARACTER_FIELD_ITERS:-3}"
  --field_reliability_floor "${CHARACTER_FIELD_RELIABILITY_FLOOR:-0.03}"
)
if [[ -n "$GEOPACK" ]]; then
  char_args+=(--geopatch "$GEOPACK")
fi
"$PY" "${char_args[@]}"

if [[ "${APPLY_CHARACTER_FIELD:-0}" == "1" ]]; then
  echo "== Step1c.5: merge smoothed character field into grow prior =="
  "$PY" patchlocks/deploy/apply_character_field_to_prior.py \
    --prior "$REGION_PRIOR" \
    --character "$CHARACTER_PRIOR" \
    --out "$CHARACTER_REGION_PRIOR" \
    --length "${CHARACTER_FIELD_LENGTH_MODE:-replace}" \
    --lift "${CHARACTER_FIELD_LIFT_MODE:-replace}"
  REGION_PRIOR="$CHARACTER_REGION_PRIOR"
else
  echo "== Step1c.5: skipped character-field prior merge =="
  echo "   per-patch character field is kept as diagnostics; Step3 uses region prior + canonical swatch"
fi

if [[ -n "${INDIVIDUAL_ANNOTATION_JSON:-}" ]]; then
  echo "== Step1c.6: apply per-animal region annotation =="
  ANNOTATED_REGION_PRIOR="${ANNOTATED_REGION_PRIOR:-${REGION_PRIOR%.npz}.annotated.npz}"
  "$PY" patchlocks/deploy/apply_individual_region_annotation.py \
    --prior "$REGION_PRIOR" \
    --annotation "$INDIVIDUAL_ANNOTATION_JSON" \
    --out "$ANNOTATED_REGION_PRIOR"
  REGION_PRIOR="$ANNOTATED_REGION_PRIOR"
fi

echo "== Step1d: build reliable canonical swatch =="
swatch_args=(
  patchlocks/deploy/build_canon_swatch.py
  --proposals "$PROPOSAL_NPZ" \
  --pack "$PACK" \
  --character "$CHARACTER_PRIOR" \
  --out "$SWATCH" \
  --reliable_pct "${SWATCH_RELIABLE_PCT:-80}" \
  --min_straightness_pct "${SWATCH_MIN_STRAIGHTNESS_PCT:-75}" \
  --max_curl_pct "${SWATCH_MAX_CURL_PCT:-35}" \
  --max_arc_over_psize_pct "${SWATCH_MAX_ARC_OVER_PSIZE_PCT:-35}" \
  --max_lift_over_arc_pct "${SWATCH_MAX_LIFT_OVER_ARC_PCT:-35}" \
  --max_tip_spread_pct "${SWATCH_MAX_TIP_SPREAD_PCT:-50}" \
  --per_patch "${SWATCH_PER_PATCH:-32}" \
  --swatch_strands "${SWATCH_STRANDS:-20480}"
)
if [[ -n "${SWATCH_MIN_STRAIGHTNESS:-}" ]]; then
  swatch_args+=(--min_straightness "$SWATCH_MIN_STRAIGHTNESS")
fi
if [[ -n "${SWATCH_MAX_CURL:-}" ]]; then
  swatch_args+=(--max_curl "$SWATCH_MAX_CURL")
fi
if [[ -n "${SWATCH_MAX_ARC_OVER_PSIZE:-}" ]]; then
  swatch_args+=(--max_arc_over_psize "$SWATCH_MAX_ARC_OVER_PSIZE")
fi
if [[ -n "${SWATCH_MAX_LIFT_OVER_ARC:-}" ]]; then
  swatch_args+=(--max_lift_over_arc "$SWATCH_MAX_LIFT_OVER_ARC")
fi
if [[ -n "${SWATCH_MAX_TIP_SPREAD:-}" ]]; then
  swatch_args+=(--max_tip_spread "$SWATCH_MAX_TIP_SPREAD")
fi
if [[ "${SWATCH_RAW_PATHS:-0}" == "1" ]]; then
  swatch_args+=(--raw_swatch)
fi
if [[ "${SWATCH_COHERENT_PATHS:-0}" == "1" ]]; then
  swatch_args+=(--coherent_swatch)
fi
"$PY" "${swatch_args[@]}"

if [[ -z "$PATCH_PARAMS" && -n "$DATA_DIR" && -n "$GEOPACK" ]]; then
  echo "== Step2: multi-view direction fit (flow_target=image, NO oracle; length opt-in) =="
  step2_args=(
    patchlocks/deploy/patch_graph_opt_geo.py
    --pack "$GEOPACK" \
    --proposal_npz "$PROPOSAL_NPZ" \
    --data_dir "$DATA_DIR" \
    --views "$VIEWS" \
    --flow_target "${STEP2_FLOW_TARGET:-image}" \
    --opt_vars "${STEP2_OPT_VARS:-flow,length}" \
    --w_anchor "${STEP2_W_ANCHOR:-0.5}" \
    --anchor_conf "${STEP2_ANCHOR_CONF:-dir_coh}" \
    --steps "${STEP2_STEPS:-300}" \
    --render_w "${STEP2_RENDER_W:-768}" \
    --max_roots "${STEP2_MAX_ROOTS:-60000}" \
    --splat_samples "${STEP2_SPLAT_SAMPLES:-5}" \
    --splat_radius "${STEP2_SPLAT_RADIUS:-2.2}" \
    --alpha_scale "${STEP2_ALPHA_SCALE:-0.55}" \
    --auto_region_level "${STEP2_AUTO_REGION_LEVEL:-important}" \
    --out "$STEP2_OUT"
  )
  if [[ "${STEP2_PART_WEIGHTED_LOSS:-1}" == "1" ]]; then
    step2_args+=(--part_weighted_loss)
  fi
  if [[ "${STEP2_PART_PRIORS:-1}" == "1" ]]; then
    step2_args+=(--part_priors)
  fi
  if [[ "${STEP2_DUMP_REGION_VIZ:-1}" == "1" ]]; then
    step2_args+=(--dump_region_viz)
  fi
  "$PY" "${step2_args[@]}"
  PATCH_PARAMS="$STEP2_OUT/patch_params.npz"
fi

if [[ -n "${SWATCH_FIXED:-}" ]]; then SWATCH="$SWATCH_FIXED"; echo "  [ablation] using FIXED swatch: $SWATCH"; fi
echo "== Step3: V1-style grow from canonical swatch =="
grow_args=(
  patchlocks/deploy/grow_v2_region_prior.py
  --prior "$REGION_PRIOR"
  --strand_profile_npz "$SWATCH"
  --radius_profile_npz "$SWATCH"
  --strand_profile_lift_scale "${STRAND_PROFILE_LIFT_SCALE:-1.0}"
  --torso_lift_k "${TORSO_LIFT_K:-0.0}"
  --torso_lift_cap "${TORSO_LIFT_CAP:-0.010}"
  --torso_lift_regions "${TORSO_LIFT_REGIONS:-body}"
  --torso_lift_frac "${TORSO_LIFT_FRAC:-1.0}"
  --torso_lift_mode "${TORSO_LIFT_MODE:-random}"
  --torso_lift_smooth_k "${TORSO_LIFT_SMOOTH_K:-8}"
  --torso_lift_smooth_iters "${TORSO_LIFT_SMOOTH_ITERS:-4}"
  --out "$GROW_OUT"
  --target_arclen "${TARGET_ARCLEN:-0}"
  --target_strands "${TARGET_STRANDS:-0}"
  --scale_contract "${SCALE_CONTRACT:-patch_window_footprint}"
  --scale_contract_length_exp "${SCALE_CONTRACT_LENGTH_EXP:-1.0}"
  --scale_contract_radius_exp "${SCALE_CONTRACT_RADIUS_EXP:-0.0}"
  --scale_contract_density_exp "${SCALE_CONTRACT_DENSITY_EXP:-0.5}"
  --length_mode region_smooth
  --length_region_blend "${LENGTH_REGION_BLEND:-0.75}"
  --region_detail_policy "${REGION_DETAIL_POLICY:-none}"
  --region_detail_tail_scale "${REGION_DETAIL_TAIL_SCALE:-0.68}"
  --region_detail_tail_keep "${REGION_DETAIL_TAIL_KEEP:-0.88}"
  --region_detail_tail_body_cap "${REGION_DETAIL_TAIL_BODY_CAP:-1.12}"
  --region_detail_tail_start_pct "${REGION_DETAIL_TAIL_START_PCT:-35.0}"
  --region_detail_tail_end_pct "${REGION_DETAIL_TAIL_END_PCT:-96.0}"
  --region_detail_tail_uniform_scale "${REGION_DETAIL_TAIL_UNIFORM_SCALE:-1.0}"
  --region_detail_tail_uniform_keep "${REGION_DETAIL_TAIL_UNIFORM_KEEP:-1.0}"
  --region_detail_tail_metric "${REGION_DETAIL_TAIL_METRIC:-axis}"
  --region_detail_tail_taper_pow "${REGION_DETAIL_TAIL_TAPER_POW:-1.0}"
  --region_detail_tail_keep_pow "${REGION_DETAIL_TAIL_KEEP_POW:-1.0}"
  --region_detail_tail_tip_keep "${REGION_DETAIL_TAIL_TIP_KEEP:--1.0}"
  --region_detail_feather "${REGION_DETAIL_FEATHER:-0.0}"
  --region_detail_ear_scale "${REGION_DETAIL_EAR_SCALE:-0.78}"
  --region_detail_ear_keep "${REGION_DETAIL_EAR_KEEP:-0.95}"
  --region_detail_ear_body_cap "${REGION_DETAIL_EAR_BODY_CAP:-0.82}"
  --region_detail_leg_scale "${REGION_DETAIL_LEG_SCALE:-1.0}"
  --region_detail_leg_keep "${REGION_DETAIL_LEG_KEEP:-1.0}"
  --region_detail_leg_body_cap "${REGION_DETAIL_LEG_BODY_CAP:-0.0}"
  --region_detail_leg_feather "${REGION_DETAIL_LEG_FEATHER:-0.0}"
  --region_detail_paw_scale "${REGION_DETAIL_PAW_SCALE:-0.78}"
  --region_detail_paw_keep "${REGION_DETAIL_PAW_KEEP:-1.0}"
  --region_detail_paw_body_cap "${REGION_DETAIL_PAW_BODY_CAP:-0.82}"
  --region_detail_paw_feather "${REGION_DETAIL_PAW_FEATHER:-0.0}"
  --region_detail_face_scale "${REGION_DETAIL_FACE_SCALE:-1.0}"
  --region_detail_face_keep "${REGION_DETAIL_FACE_KEEP:-1.0}"
  --region_detail_face_body_cap "${REGION_DETAIL_FACE_BODY_CAP:-0.0}"
  --tail_length_policy "${TAIL_LENGTH_POLICY:-uniform}"
  --tail_length_regions "${TAIL_LENGTH_REGIONS:-tail,tail_base}"
  --tail_length_body_cap "${TAIL_LENGTH_BODY_CAP:-0.0}"
  --tail_length_tip_scale "${TAIL_LENGTH_TIP_SCALE:-1.0}"
  --tail_length_tip_start_pct "${TAIL_LENGTH_TIP_START_PCT:-30.0}"
  --tail_length_tip_end_pct "${TAIL_LENGTH_TIP_END_PCT:-98.0}"
  --tail_length_tip_pow "${TAIL_LENGTH_TIP_POW:-1.5}"
  --region_flow_policy "${REGION_FLOW_POLICY:-none}"
  --region_flow_smooth_k "${REGION_FLOW_SMOOTH_K:-10}"
  --region_flow_iters "${REGION_FLOW_ITERS:-2}"
  --region_flow_ear_smooth "${REGION_FLOW_EAR_SMOOTH:-0.65}"
  --region_flow_ear_damp "${REGION_FLOW_EAR_DAMP:-0.55}"
  --region_flow_tail_smooth "${REGION_FLOW_TAIL_SMOOTH:-0.55}"
  --region_flow_tail_damp "${REGION_FLOW_TAIL_DAMP:-0.65}"
  --region_flow_paw_smooth "${REGION_FLOW_PAW_SMOOTH:-0.35}"
  --region_flow_paw_damp "${REGION_FLOW_PAW_DAMP:-0.85}"
)
if [[ -n "$PATCH_PARAMS" ]]; then
  grow_args+=(--patch_params "$PATCH_PARAMS")
  grow_args+=(--patch_param_vars "${GROW_PATCH_PARAM_VARS:-${STEP2_OPT_VARS:-flow,length}}")
fi
if [[ "${GROW_NO_POU:-0}" == "1" ]]; then grow_args+=(--no_pou); echo "  [ablation] --no_pou (PoU normalization OFF)"; fi
"$PY" "${grow_args[@]}"
FUR="$GROW_OUT/surfpack_v2fur.npz"

if [[ -n "$MESH" ]]; then
  echo "== Render: hair-BSDF visual QA =="
  "$BLENDER" -b --factory-startup --python patchlocks/deploy/render_dog_fur_pro.py -- \
    --fur "$FUR" --mesh "$MESH" --out "$RENDER_OUT" \
    --views "${RENDER_VIEWS:-6}" --res "${RENDER_RES:-1280}" --samples "${RENDER_SAMPLES:-40}" \
    --up "${RENDER_UP:-y}" --camdist "${RENDER_CAMDIST:-0.5}" \
    --flat "${RENDER_FLAT:-0.78,0.78,0.74}" --spec "${RENDER_SPEC:-0.22}" \
    --skin "${RENDER_SKIN:-0.22,0.16,0.12}"
fi

echo "DONE V2 character mainline"
echo "  proposals:       $PROPOSAL_NPZ"
echo "  region prior:    $REGION_PRIOR"
echo "  character prior: $CHARACTER_PRIOR"
echo "  swatch:          $SWATCH"
[[ -n "$PATCH_PARAMS" ]] && echo "  step2 params:    $PATCH_PARAMS"
echo "  fur:             $FUR"
if [[ -n "$MESH" ]]; then
  echo "  render:          $RENDER_OUT/furview_0.png"
fi
