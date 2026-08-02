#!/usr/bin/env bash
set -euo pipefail

STAGE2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${STAGE2_DIR}/../.." && pwd)"
STAGE2_PYTHON="${STAGE2_PYTHON:-/home/lsw/miniconda3/envs/ortho-yolo26/bin/python}"

mkdir -p "${STAGE2_DIR}/.matplotlib"
export MPLCONFIGDIR="${STAGE2_DIR}/.matplotlib"
cd "${PROJECT_ROOT}"
exec "${STAGE2_PYTHON}" -m src.stage2.prepare_data "$@"
