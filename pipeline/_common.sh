# Sourced by every pipeline/NN_*.sh script: run from the repository root with strict bash.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export ROOT
export PY="${PY:-python}"

step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

require() {  # require <path> [hint]: fail early with a pointer to the data manifest
  if [[ ! -e "$1" ]]; then
    echo "missing: $1" >&2
    echo "${2:-link the data first: bash pipeline/00_link_data.sh /path/to/DATA_ROOT (see data/manifest.txt)}" >&2
    exit 2
  fi
}
