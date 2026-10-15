#!/usr/bin/env bash
# Rebuild every scriptable figure of the PatchLocks paper (main + supplement) into
# outputs/figures/, using the same file names and sub-directories as the paper's
# Figures/ folder. CPU only (PIL / numpy / matplotlib); no Blender, no GPU.
#
#   bash figures/build_paper_figures.sh            # all figures
#   bash figures/build_paper_figures.sh teaser nfcompare   # a subset
#
# Targets: teaser nfcompare baselines_qual supp_panels baselinegrid patchfig patchresult
# (pipeline.png is a manual draw.io export -- see figures/README.md.)
# Inputs come from the data linked by scripts/link_data.sh (data/manifest.txt).
# Afterwards compare against the published files with
#   python figures/verify_paper_figures.py --paper-figures <PatchLocks_paper/Figures>
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"                       # builders use repository-relative paths
PY="${PY:-/home/rog/miniconda3/envs/difflocks/bin/python}"
OUT="outputs/figures"
FV="$OUT/final_v16_4animals"
S60="$OUT/_supp60/final_v16_4animals"
mkdir -p "$FV" "$S60" "$OUT/supplement_panels"

run() { echo "+ $*"; "$@"; }

build_teaser() {          # main Fig. 1  fig:teaser
  run "$PY" figures/workbench/02_teaser_four_animals/code/build_teaser_clean.py
}

build_nfcompare() {       # main Fig. 3  fig:nfcompare (beagle + panda; panda "Ours" = length x1.2 render)
  NF_ANIMALS=beagle,panda NF_OURS=panda=render_panda_champ_ours_len120 NF_OURS_OWN_CROP=panda \
  NF_PNG_OPTIMIZE=1 NF_OUT="$FV/NF_COMPARE_2ANIMALS.png" run "$PY" patchlocks/deploy/_build_nfcompare_nf.py
}

build_baselines_qual() {  # main fig:baselines_qual
  QUAL_OUT="$OUT/fig_baselines_qual.png" \
  run "$PY" patchlocks/experiments/rebuttal_2027/paper_draft/compose_qual.py
}

build_supp_panels() {     # supp. fig:qual_full(_more), fig:nfcompare_full(_more), fig:gate_a/b
  Q_OUT="$FV/ALL_GLOBAL_4ANIMALS.png" run "$PY" patchlocks/deploy/_build_qual_nf.py
  NF_OUT="$FV/NF_COMPARE_4ANIMALS.png" run "$PY" patchlocks/deploy/_build_nfcompare_nf.py
  SUPP_GLOBAL="$FV/ALL_GLOBAL_4ANIMALS.png" SUPP_NF="$FV/NF_COMPARE_4ANIMALS.png" \
  SUPP_OUT="$OUT/supplement_panels" \
  run "$PY" figures/workbench/11_supplement_panels/code/build_supplement_panels.py
}

build_baselinegrid() {    # supp. fig:baselinegrid (included at 60 % via _supp60/)
  BG_OUT="$FV/BASELINE_GRID_NF.png" run "$PY" patchlocks/deploy/_build_baseline_grid_nf.py
  run "$PY" figures/make_supp60.py "$FV/BASELINE_GRID_NF.png" "$S60/BASELINE_GRID_NF.png"
}

build_patchfig() {        # supp. fig:patch (included at 60 % via _supp60/); loads ~5 GB of surface packs
  PF_OUT="$FV/ALL_PATCHES_AND_CHARACTER_TIGHT.png" run "$PY" patchlocks/deploy/_build_patchfig_dark.py
  run "$PY" figures/make_supp60.py "$FV/ALL_PATCHES_AND_CHARACTER_TIGHT.png" \
    "$S60/ALL_PATCHES_AND_CHARACTER_TIGHT.png"
}

build_patchresult() {     # supp. fig:patch_result
  PATCH_OUT="$OUT/PATCH_RESULT_SINGLE_COLUMN.png" \
  run "$PY" figures/workbench/05_detail_closeups/code/build_patch_result_single.py
}

ALL=(teaser nfcompare baselines_qual supp_panels baselinegrid patchfig patchresult)
TARGETS=("$@")
(( ${#TARGETS[@]} )) || TARGETS=("${ALL[@]}")
for t in "${TARGETS[@]}"; do
  if [[ " ${ALL[*]} " != *" $t "* ]]; then echo "unknown target: $t (choose from: ${ALL[*]})" >&2; exit 2; fi
  echo "== $t"
  "build_$t"
done
echo "done -> $REPO/$OUT"
