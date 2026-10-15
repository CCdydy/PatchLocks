#!/usr/bin/env bash
# Stage 7 — rebuild the paper figures from renders and grooms (CPU only, ~1 min).
#
#   bash pipeline/07_build_paper_figures.sh                          # -> outputs/figures/
#   PAPER_FIGURES=/path/to/paper/Figures bash pipeline/07_build_paper_figures.sh   # + verify
#
# All 12 script-built figures are byte-identical to the published ones; the pipeline overview is
# drawn in draw.io (figures/pipeline/). Details: figures/README.md.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
step "build figures -> outputs/figures"
bash figures/build_paper_figures.sh "$@"
if [[ -n "${PAPER_FIGURES:-}" ]]; then
  step "verify against $PAPER_FIGURES"
  "$PY" figures/verify_paper_figures.py --paper-figures "$PAPER_FIGURES"
fi
