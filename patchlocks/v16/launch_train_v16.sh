#!/usr/bin/env bash
set -euo pipefail

# V16 Clump-Lite launcher — E1 (V14 full-mode z64 diffusion) + two clump levers (ADD1 CW, ADD2 bucket).
# E1 stays INTACT: full-mode z64 diffusion as the ONLY generator, sigma_max=80, p_mean=-0.4, CFG (cond_drop+cfg),
# EMA, PURE EDM-weighted MSE, same +7 raydir condition, same Heun sampler, composite-score eval + early-stop.
# The ONLY new knobs are CLUMP_CW (ADD1) and CLUMP_BUCKET (ADD2); everything else defaults to the E1 recipe.
# Frequent ckpt save + eval (1000) so the ADD3 clump-coherence diagnostics print often; early-stop on the
# COMPOSITE SCORE (E1 Part F). Does NOT touch V10-V15 files/ckpts.
#
#   E2a (CW only):       CLUMP_CW=2 CLUMP_BUCKET=0 OUT=patchlocks/datasets/v16/e2a_cw2 bash patchlocks/v16/launch_train_v16.sh
#   E2b (bucket only):   CLUMP_CW=0 CLUMP_BUCKET=1 OUT=patchlocks/datasets/v16/e2b_bucket bash patchlocks/v16/launch_train_v16.sh
#   E2c (both):          CLUMP_CW=2 CLUMP_BUCKET=1 OUT=patchlocks/datasets/v16/e2c_both bash patchlocks/v16/launch_train_v16.sh
# Watch: `[eval] score=`, `[eval-v16] clump_corr= ... parallelism ... tangled-rate ...`.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python}"
cd "$ROOT"

DATA="${DATA:-patchlocks/datasets/v11_longfur/renders_mv}"
CODEC="${CODEC:-patchlocks/models/fur_strand_codec_v6upright.pt}"
LATENT="${LATENT:-target_latent64_v6codec.npz}"
POSE="${POSE:-v13_pose_64.npz}"
RES="${RES:-64}"
ITERS="${ITERS:-20000}"
BS="${BS:-16}"
B3_PRE="${B3_PRE:-3000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"     # frequent ckpt save
EVAL_EVERY="${EVAL_EVERY:-1000}"     # frequent eval (ADD3 diagnostics + composite score)
EARLY_STOP="${EARLY_STOP:-3}"        # early-stop on the COMPOSITE SCORE (E1 Part F), not nMAE
MAX_DIRS="${MAX_DIRS:-0}"
SEED="${SEED:-0}"
RESUME="${RESUME:-}"

# ===== E1 recipe knobs (locked to the E1 defaults; override only for E0/repro) =====
DIFF_MODE="${DIFF_MODE:-full}"       # E1: full = diffuse z64 directly, the ONLY generator
SIGMA_MAX="${SIGMA_MAX:-80}"         # E1
SIGMA_MIN="${SIGMA_MIN:-0.002}"
P_MEAN="${P_MEAN:--0.4}"             # E1
P_STD="${P_STD:-1.2}"
COND_DROP="${COND_DROP:-0.1}"        # E1: CFG all-group random drop
DROP_GROUP="${DROP_GROUP:-none}"     # HARD RULE: NO group-dropout. Keep none.
CFG="${CFG:-3}"                      # E1: classifier-free guidance scale
EMA_DECAY="${EMA_DECAY:-0.999}"      # E1
SCORE_EVAL_N="${SCORE_EVAL_N:-240}"

# ===== V16 Clump-Lite levers (the ONLY additions vs E1) =====
CLUMP_CW="${CLUMP_CW:-0}"            # ADD1: clump-sensitive z64 CHANNEL WEIGHTING on the LATENT EDM-MSE. 0=off. Try 1.5/2.0/3.0.
CLUMP_BUCKET="${CLUMP_BUCKET:-0}"   # ADD2: clump-LEVEL quartile balanced sampler. 0=off, 1=on.
CLUMP_CW_T3_MUL="${CLUMP_CW_T3_MUL:-1.25}"      # ADD1: tier-3.0 channels get clump_cw * this (capped at 4)
CLUMP_BUCKET_TANGLE_DEG="${CLUMP_BUCKET_TANGLE_DEG:-30.0}"  # ADD2/ADD3: tangled-rate threshold (deg)

# Conditioning/target toggles (carried from V14/E1)
NO_COLOR="${NO_COLOR:-1}"            # 1 = z64-only target (geometry gate)
VIEW_DROPOUT="${VIEW_DROPOUT:-1}"
SINGLE_VIEW="${SINGLE_VIEW:-0}"

TAG="clumplite_cw${CLUMP_CW}_bk${CLUMP_BUCKET}"; [ "$SINGLE_VIEW" = "1" ] && TAG="${TAG}_sv"
OUT="${OUT:-patchlocks/datasets/v16/v16_${TAG}}"
mkdir -p "$OUT"

ARGS=(
  --renders "$DATA" --res "$RES"
  --codec_path "$CODEC" --latent_name "$LATENT" --pose_name "$POSE"
  --iters "$ITERS" --bs "$BS" --b3_pre_iters "$B3_PRE"
  --save_every "$SAVE_EVERY" --eval_every "$EVAL_EVERY" --early_stop "$EARLY_STOP"
  --diff_mode "$DIFF_MODE"
  --sigma_max "$SIGMA_MAX" --sigma_min "$SIGMA_MIN" --p_mean "$P_MEAN" --p_std "$P_STD"
  --cond_drop "$COND_DROP" --drop_group "$DROP_GROUP" --cfg "$CFG" --ema_decay "$EMA_DECAY"
  --score_eval_n "$SCORE_EVAL_N"
  --clump_cw "$CLUMP_CW" --clump_bucket "$CLUMP_BUCKET"
  --clump_cw_t3_mul "$CLUMP_CW_T3_MUL" --clump_bucket_tangle_deg "$CLUMP_BUCKET_TANGLE_DEG"
  --seed "$SEED"
  --out "$OUT"
)
[ "$NO_COLOR"    = "1" ] && ARGS+=(--no_color)
[ "$VIEW_DROPOUT" = "1" ] && ARGS+=(--view_dropout)
[ "$SINGLE_VIEW" = "1" ] && ARGS+=(--single_view)
[ "$MAX_DIRS"   != "0" ] && ARGS+=(--max_dirs "$MAX_DIRS")
[ -n "$RESUME"         ] && ARGS+=(--resume_stage3 "$RESUME")

echo "[V16] OUT=$OUT TAG=$TAG | E1: diff_mode=$DIFF_MODE sigma_max=$SIGMA_MAX p_mean=$P_MEAN cond_drop=$COND_DROP cfg=$CFG ema_decay=$EMA_DECAY"
echo "[V16] clump levers: CLUMP_CW=$CLUMP_CW CLUMP_BUCKET=$CLUMP_BUCKET (t3_mul=$CLUMP_CW_T3_MUL tangle_deg=$CLUMP_BUCKET_TANGLE_DEG)"
echo "[V16] $PY patchlocks/v16/train_v16_clumplite.py ${ARGS[*]}"
exec "$PY" patchlocks/v16/train_v16_clumplite.py "${ARGS[@]}"
