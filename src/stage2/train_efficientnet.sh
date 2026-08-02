#!/usr/bin/env bash
set -euo pipefail

STAGE2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${STAGE2_DIR}/../.." && pwd)"
STAGE2_PYTHON="${STAGE2_PYTHON:-python3}"

mkdir -p "${STAGE2_DIR}/.matplotlib"
export MPLCONFIGDIR="${STAGE2_DIR}/.matplotlib"
cd "${PROJECT_ROOT}"
STAGE2_NPROC_PER_NODE="${STAGE2_NPROC_PER_NODE:-1}"
if [[ "${STAGE2_NPROC_PER_NODE}" == "1" ]]; then
  exec "${STAGE2_PYTHON}" -m src.stage2.train_efficientnet "$@"
fi
exec "${STAGE2_PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${STAGE2_NPROC_PER_NODE}" -m src.stage2.train_efficientnet "$@"
