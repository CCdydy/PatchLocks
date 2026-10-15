#!/usr/bin/env bash
# Symlink every entry of data/manifest.txt from a data root into this repository.
#
#   bash scripts/link_data.sh /path/to/DATA_ROOT [category ...]
#
# DATA_ROOT must use the same relative layout as the manifest (for example an extracted
# release bundle, or the original paper transfer package). Optional category filters
# (weights | ours | derived | thirdparty) restrict which entries are linked.
# Existing paths in the repository are never overwritten; missing sources are reported.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:?usage: link_data.sh DATA_ROOT [category ...]}"
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
shift || true
CATS=("$@")

linked=0; skipped=0; missing=0
while read -r cat rel _; do
  [[ -z "$cat" || "$cat" == \#* ]] && continue
  if (( ${#CATS[@]} )) && [[ ! " ${CATS[*]} " == *" $cat "* ]]; then continue; fi
  src="$DATA_ROOT/$rel"; dst="$REPO/$rel"
  if [[ ! -e "$src" ]]; then
    echo "MISSING  $rel"; missing=$((missing + 1)); continue
  fi
  if [[ -e "$dst" || -L "$dst" ]]; then
    skipped=$((skipped + 1)); continue
  fi
  mkdir -p "$(dirname "$dst")"
  ln -s "$src" "$dst"
  linked=$((linked + 1))
done < "$REPO/data/manifest.txt"

echo "linked $linked, already present $skipped, missing $missing"
[[ $missing -eq 0 ]]
