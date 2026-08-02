#!/usr/bin/env bash
set -euo pipefail

STAGE1_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${STAGE1_DIR}/../.." && pwd)"
STAGE1_PYTHON="${STAGE1_PYTHON:-/home/lsw/miniconda3/envs/ortho-yolo26/bin/python}"

mkdir -p "${STAGE1_DIR}/.matplotlib"
export MPLCONFIGDIR="${STAGE1_DIR}/.matplotlib"
cd "${PROJECT_ROOT}"
exec "${STAGE1_PYTHON}" -m src.stage1.evaluate_efficientnet "$@"
