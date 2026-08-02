#!/usr/bin/env bash
set -euo pipefail

STAGE2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for arg in "$@"; do
  case "${arg}" in
    --fold|--fold=*)
      echo "Error: --fold is managed by train_all_folds.sh and must not be provided." >&2
      exit 2
      ;;
  esac
done

for fold in 0 1 2 3 4; do
  echo "Starting Stage-2 fold ${fold}/4"
  bash "${STAGE2_DIR}/train_efficientnet.sh" --fold "${fold}" "$@"
done
