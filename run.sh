#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

usage() {
    cat <<'EOF'
Usage:
  ./run.sh install
  ./run.sh adapt
  ./run.sh train
  ./run.sh detect
  ./run.sh all

Optional environment variable overrides:
  PYTHON=python3
  RAW_INPUT=data/raw/cicddos2019_dataset.csv
  ADAPTED_OUTPUT=data/raw/adapted_cic2019.csv
  DETECT_FILE=data/raw/adapted_cic2019.csv
  ROWS=50000
  CHUNK_SIZE=5000
  OUTPUT=results.csv
  VERBOSE=1

Examples:
  ./run.sh adapt
  ROWS=50000 CHUNK_SIZE=10000 ./run.sh detect
  RAW_INPUT=data/raw/custom.csv ADAPTED_OUTPUT=data/raw/custom_adapted.csv ./run.sh adapt
  ./run.sh all
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

TARGET="$1"
shift || true

case "$TARGET" in
    help|-h|--help)
        usage
        ;;
    install|adapt|train|detect|all|clean-results)
        make "$TARGET" "$@"
        ;;
    *)
        echo "Unknown command: $TARGET" >&2
        echo "" >&2
        usage
        exit 1
        ;;
esac
