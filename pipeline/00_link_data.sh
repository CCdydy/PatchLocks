#!/usr/bin/env bash
# Stage 0 — make weights and data available at the paths the code expects.
#
#   bash pipeline/00_link_data.sh /path/to/DATA_ROOT [weights|ours|derived|thirdparty ...]
#
# Symlinks every entry of data/manifest.txt from DATA_ROOT (same relative layout).
# The linked data is treated as read-only: every later stage writes to outputs/.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
bash scripts/link_data.sh "$@"
