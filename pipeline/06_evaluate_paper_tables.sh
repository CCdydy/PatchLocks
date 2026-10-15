#!/usr/bin/env bash
# Stage 6 — recompute every quantitative result of the paper.
#
#   bash pipeline/06_evaluate_paper_tables.sh            # all tables (~1-1.5 h on one GPU)
#   bash pipeline/06_evaluate_paper_tables.sh capability baselines   # a subset
#
# Each target writes its result next to the archived one: results with a configurable output go
# to outputs/tables/; the four scripts with a fixed output path rewrite the tracked JSON in place,
# so `git diff` shows whether a number moved. Targets:
#   capability     tab:capability (n=80, 1-view vs ours)          patchlocks/v16/eval_v16_vs_v10.json
#   recipe         supp tab:recipe (E1 middle row)                patchlocks/v16/eval_e1_vs_v10_sameprotocol.json
#   baselines      tab:baselines ID (n=160) + EXT (n=80) + CIs    outputs/tables/{eval3arm,ci}_{testB,ood}.json
#   oneview        one-view ablation (0.631 -> 0.720)             outputs/tables/eval_k1_val80.json
#   ceiling        supp tab:ceiling                               outputs/tables/clumpaware_testB.json
#   spearman       character Spearman correlations (CPU)          .../rebuttal_2027/init_positioning.json
#   identifiability supp tab:identifiability (CPU)                .../rebuttal_2027/identifiability.json
#   seams          supp tab:seams (CPU)                           outputs/tables/seam_continuity.json
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
E=patchlocks/experiments/rebuttal_2027
T=outputs/tables; mkdir -p "$T/logs"
TARGETS=("$@"); (( ${#TARGETS[@]} )) || TARGETS=(capability recipe baselines oneview ceiling spearman identifiability seams)

run() { local name="$1"; shift; step "$name"; local t0=$(date +%s)
  if "$@" > "$T/logs/$name.log" 2>&1; then echo "ok ($(( $(date +%s) - t0 )) s), log: $T/logs/$name.log"
  else echo "FAILED, see $T/logs/$name.log" >&2; return 1; fi; }

for t in "${TARGETS[@]}"; do
  case "$t" in
    capability) run capability "$PY" patchlocks/v16/eval_v16_vs_v10.py ;;
    recipe)     run recipe "$PY" patchlocks/v16/eval_e1_vs_v10_sameprotocol.py ;;
    baselines)
      run baselines_testB "$PY" $E/eval_3arm.py --mode all --renders $E/testB_renders --ckpt_cond_stats --out $T/eval3arm_testB.json
      run ci_testB        "$PY" $E/bootstrap_ci.py --json $T/eval3arm_testB.json --out $T/ci_testB.json
      run baselines_ood   "$PY" $E/eval_3arm.py --mode all --renders $E/ood_renders --ckpt_cond_stats --out $T/eval3arm_ood.json
      run ci_ood          "$PY" $E/bootstrap_ci.py --json $T/eval3arm_ood.json --out $T/ci_ood.json ;;
    oneview)
      run val80   "$PY" $E/eval_3arm.py --mode val --out $T/eval3arm_val80.json
      run oneview "$PY" $E/eval_k1.py --k4_json $T/eval3arm_val80.json --out $T/eval_k1_val80.json ;;
    ceiling)    run ceiling "$PY" $E/eval_clumpaware.py --ckpt_cond_stats --out $T/clumpaware_testB.json ;;
    spearman)   run spearman "$PY" $E/eval_init_positioning.py ;;
    identifiability) run identifiability "$PY" $E/eval_identifiability.py ;;
    seams)      run seams "$PY" $E/eval_seam_continuity.py --out $T/seam_continuity.json ;;
    *) echo "unknown target: $t" >&2; exit 2 ;;
  esac
done
step "done — compare with the archived results"
echo "  git diff --stat -- patchlocks/        # in-place JSONs"
echo "  python pipeline/compare_tables.py     # outputs/tables vs archived JSONs"
