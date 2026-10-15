#!/usr/bin/env bash
# B post-render chain (CONTRACT.md section B): bake -> pose -> QA -> FREEZE -> eval_3arm -> bootstrap.
# Run AFTER the Blender render of testB_renders completes (160 dirs x rgb_0..3 + cameras.json + patch.npz).
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$ROOT"
PY="${PY:-python}"
R=patchlocks/experiments/rebuttal_2027/testB_renders
E=patchlocks/experiments/rebuttal_2027

echo "== [1/6] render completeness check =="
n_dirs=$(ls -d "$R"/*/ | wc -l)
n_ok=$(for d in "$R"/*/; do [ -f "$d/rgb_3.png" ] && [ -f "$d/cameras.json" ] && [ -f "$d/patch.npz" ] && echo 1; done | wc -l)
echo "dirs=$n_dirs complete=$n_ok"
[ "$n_ok" -ge 150 ] || { echo "FAIL: <150 complete render dirs"; exit 1; }

echo "== [2/6] bake latents + fields (same codec, H=64) =="
$PY patchlocks/data_gen/bake_mv_targets.py "$R" \
  --codec patchlocks/models/fur_strand_codec_v6upright.pt \
  --out-name target_latent64_v6codec.npz --H 64

echo "== [3/6] pose sidecars =="
$PY patchlocks/data_gen/prepare_v13_mv_dataset.py --renders "$R" \
  --out "$E/testB_prep" --res 64 --write-pose

echo "== [4/6] QA: bake coverage =="
$PY - <<'PYEOF'
import glob, numpy as np, json, os
R="patchlocks/experiments/rebuttal_2027/testB_renders"
dirs=sorted(glob.glob(R+"/*"))
ok=miss=0; masks=[]
for d in dirs:
    t=f"{d}/target_latent64_v6codec.npz"; p=f"{d}/v13_pose_64.npz"
    if os.path.exists(t) and os.path.exists(p) and os.path.exists(f"{d}/field_gt_64.npz"):
        m=np.load(t)["mask"]; masks.append(float(np.mean(m>0.5))); ok+=1
    else: miss+=1
print(f"baked+posed OK {ok}/{len(dirs)} (miss {miss}); mask coverage mean {np.mean(masks):.3f} min {np.min(masks):.3f}")
assert ok>=150, "too few complete B dirs"
PYEOF

echo "== [5/6] FREEZE (manifest + marker; no further selection permitted) =="
( cd "$R" && ls -d */ | sort ) > "$E/testB_FROZEN_manifest.txt"
date -Is >> "$E/testB_FROZEN_manifest.txt"
echo "FROZEN: contents listed in testB_FROZEN_manifest.txt; B must not be regenerated or filtered." > "$E/testB_FROZEN"

echo "== [6/6] eval_3arm on B + bootstrap =="
$PY "$E/eval_3arm.py" --mode all --renders "$R" --ckpt_cond_stats --out "$E/eval3arm_testB.json"
$PY "$E/bootstrap_ci.py" --json "$E/eval3arm_testB.json" --out "$E/ci_testB.json"
echo "== B chain DONE =="
