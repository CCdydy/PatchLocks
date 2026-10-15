#!/usr/bin/env bash
# v6 REDESIGNED dataset overnight pipeline: self-healing booth render (--dr, NO denoise = rainbow guard) ->
# bake H64 latent -> field_gt -> v6 HYBRID train (z64 diffusion + density head + appearance). Resumable:
# render_loop skips dirs that already have rgb_3.png + cameras.json + patch.npz, so re-running continues.
set -u; ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"; cd "$ROOT"
PY="${PY:-python}"
BL=patchlocks/third_party/blender-4.5.10-linux-x64/blender
PATCHES=patchlocks/datasets/v6/patches
REND=patchlocks/datasets/v6/renders
OUT=patchlocks/datasets/stage3_v6
mkdir -p "$REND" _logs/v6
NEED=$(ls "$PATCHES"/*.npz 2>/dev/null | wc -l)
echo "=== V6-OVERNIGHT START $(date '+%m-%d %H:%M') | target=$NEED patches ==="

# ---- self-healing PARALLEL booth render (NSHARD processes over disjoint patch shards; resume on crash). denoise OFF (rainbow guard). ----
NSHARD=2; THR=$(( $(nproc) / NSHARD )); [ "$THR" -lt 1 ] && THR=1   # thread-limit each Blender so NSHARD procs don't oversubscribe cores (was thrashing at load 4x)
for attempt in 1 2 3 4; do
  d=$(ls "$REND"/*/rgb_3.png 2>/dev/null | wc -l)
  [ "$d" -ge "$NEED" ] && break
  echo "=== v6 RENDER attempt $attempt (done=$d/$NEED) ${NSHARD}x parallel (-t $THR) $(date +%H:%M) ==="
  pids=()
  for sh in $(seq 0 $((NSHARD - 1))); do
    $BL -b -t "$THR" --factory-startup --python patchlocks/data_gen/render_loop.py -- \
        --patches "$PATCHES" --out "$REND" --views 4 --res 256 --samples 24 --dr \
        --shard "$sh/$NSHARD" --seed0 $((1000 + sh * 100000)) >> "_logs/v6/render_$sh.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
done
d=$(ls "$REND"/*/rgb_3.png 2>/dev/null | wc -l)
echo "=== v6 RENDER COMPLETE done=$d/$NEED $(date +%H:%M) ==="
[ "$d" -ge $((NEED * 9 / 10)) ] || { echo "v6 RENDER INCOMPLETE ($d/$NEED) -- abort before training"; exit 1; }

# ---- bake H64 latent targets + field_gt ----
echo "=== v6 BAKE latent64 (H64) $(date +%H:%M) ==="
$PY patchlocks/models/bake_latent_batch.py "$REND" 64 > _logs/v6/bake.log 2>&1; rc=$?
[ $rc -eq 0 ] || { echo "BAKE FAILED rc=$rc (see _logs/v6/bake.log)"; exit 1; }
echo "=== v6 build_field_gt $(date +%H:%M) ==="
$PY patchlocks/data_gen/build_field_gt.py "$REND" > _logs/v6/field.log 2>&1; rc=$?
[ $rc -eq 0 ] || { echo "FIELD_GT FAILED rc=$rc (see _logs/v6/field.log)"; exit 1; }
echo "=== V6-DATA-READY renders=$d $(date +%H:%M) ==="

# ---- v6 HYBRID train: Branch B (z64 diffusion) -> Branch A (density head) -> appearance ----
echo "=== v6 GEO z64-diffusion (H64) $(date +%H:%M) ==="
$PY patchlocks/training/stage3_z64_diffusion.py --res 64 --iters 70000 --b3_pre_iters 6000 \
  --hf_w 1.0 --clump_ch_w 3.0 --hf_oversample 3.0 --cond_drop 0.1 --gcond_batch 8 \
  --renders "$REND" --out "$OUT" > _logs/v6/geo.log 2>&1
rc=$?; grep -E "patches|it +[0-9]+000 |DISTRIB|^clump|^dens|^curl|^length|shuffle|saved|Traceback|Error|OutOfMemory" _logs/v6/geo.log | tail -40 || true
[ $rc -eq 0 ] || { echo "GEO z64 FAILED rc=$rc (see _logs/v6/geo.log)"; exit 1; }

echo "=== v6 DENSITY HEAD $(date +%H:%M) ==="
$PY patchlocks/training/stage3_density_head.py --ckpt_dir "$OUT" --renders "$REND" --out "$OUT" --gcond_batch 8 > _logs/v6/dh.log 2>&1
rc=$?; grep -E "massCorr|mean|spatCorr|relL1|saved|Traceback|Error" _logs/v6/dh.log | tail -15 || true
[ $rc -eq 0 ] || { echo "DENSITY HEAD FAILED rc=$rc (see _logs/v6/dh.log)"; exit 1; }

echo "=== v6 APPEARANCE+MATERIAL $(date +%H:%M) ==="
$PY patchlocks/training/stage3_2_appearance.py --ckpt_dir "$OUT" --renders "$REND" --res 64 --iters 9000 --out "$OUT" > _logs/v6/app.log 2>&1
rc=$?; grep -E "frozen|saved|bucket_acc|mat_mae|Traceback|Error" _logs/v6/app.log | tail -15 || true
[ $rc -eq 0 ] || { echo "APPEARANCE FAILED rc=$rc (see _logs/v6/app.log)"; exit 1; }

echo "=== V6-OVERNIGHT-DONE $(date '+%m-%d %H:%M') | model=$OUT ==="
