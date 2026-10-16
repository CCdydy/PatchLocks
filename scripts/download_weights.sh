#!/usr/bin/env bash
# Download the model weights from Hugging Face (zengzzy/PatchLocks) and verify them.
#
#   bash scripts/download_weights.sh
#
# Each checkpoint lands at the repository path the code expects (data/manifest.txt, category
# "weights") and is checked against scripts/weights_SHA256SUMS.txt. Existing files are kept.
# Needs curl (or wget) and sha256sum; 549 MB in total.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${BASE:-https://huggingface.co/zengzzy/PatchLocks/resolve/main}"
SUMS="$REPO/scripts/weights_SHA256SUMS.txt"
cd "$REPO"

fetch() {
  if command -v curl >/dev/null; then curl -fL --retry 3 -o "$2" "$1"; else wget -O "$2" "$1"; fi
}
while read -r _ path; do
  if [[ -e "$path" ]]; then echo "present: $path"; continue; fi
  echo "downloading $path"
  mkdir -p "$(dirname "$path")"
  fetch "$BASE/$path" "$path.part" && mv "$path.part" "$path"
done < "$SUMS"
sha256sum -c "$SUMS"
echo "weights ready."
