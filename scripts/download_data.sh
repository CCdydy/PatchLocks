#!/usr/bin/env bash
# Download our data from Hugging Face (zengzzy/PatchLocks-data) into a data root and link it.
#
#   bash scripts/download_data.sh /path/to/DATA_ROOT                 # everything (~28 GB)
#   bash scripts/download_data.sh /path/to/DATA_ROOT testsets        # or only some subsets
#
# Subsets: training  synthetic training corpus (13 GB)
#          testsets  frozen test sets for Table 2 (3.4 GB)
#          animals   the paper's reconstructions of the four animals + their renders (10 GB)
#          figures   remaining figure inputs (0.2 GB)
# The dataset uses the data-root layout, so the download is linked into this repository with
# scripts/link_data.sh afterwards. Re-running resumes and skips finished files.
# Third-party inputs (NeuralFur / ARTEMIS captures and the patch packs preprocessed from them) are
# not on Hugging Face; see docs/USAGE.md. Needs: pip install huggingface_hub
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${1:?usage: download_data.sh DATA_ROOT [training|testsets|animals|figures ...]}"
shift || true
PY="${PY:-python}"

"$PY" - "$ROOT" "$@" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download

E = "patchlocks/experiments/rebuttal_2027"
SUBSETS = {
    "training": ["patchlocks/datasets/v11_longfur/renders_mv/**"],
    "testsets": [f"{E}/{d}/**" for d in ("testB_patches", "testB_renders", "testB_prep",
                                          "ood_patches", "ood_renders", "ood_prep")],
    "animals":  ["v2_results/refine_v16_fluffy/**", "v2_results/refine_v16_4animals/**"],
    "figures":  [f"{E}/grid5_B/**", "patchlocks/datasets/v16/v16_visual_gate_agx/**",
                 "v2_results/refine_v16_paper_smooth/**", "v2_results/paper_cover_patch/**",
                 "patchlocks/deploy/nf_official_render/**"],
}
root, names = sys.argv[1], sys.argv[2:]
unknown = [n for n in names if n not in SUBSETS]
if unknown:
    sys.exit(f"unknown subset(s): {unknown}; choose from {list(SUBSETS)}")
patterns = [p for n in names for p in SUBSETS[n]] or None
snapshot_download("zengzzy/PatchLocks-data", repo_type="dataset", local_dir=root,
                  allow_patterns=patterns, ignore_patterns=["README.md", ".gitattributes"])
print("downloaded to", root)
PYEOF

# Link what was downloaded; third-party entries are reported as missing until you add them.
bash "$REPO/scripts/link_data.sh" "$ROOT" || true
