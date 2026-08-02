#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}}"
CONDA_ROOT="${CONDA_ROOT:-/paddle/miniconda3}"
CONDA_BIN="${CONDA_BIN:-${CONDA_ROOT}/bin/conda}"
ENV_DIR="${ENV_DIR:-${PROJECT_ROOT}/.conda_env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"

if [[ ! -d "${PROJECT_ROOT}/src/stage1" || ! -d "${PROJECT_ROOT}/src/stage2" ]]; then
  echo "Error: PROJECT_ROOT does not look like mSASSS_pipeline: ${PROJECT_ROOT}" >&2
  exit 1
fi

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Error: conda executable not found: ${CONDA_BIN}" >&2
  exit 1
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Conda:        ${CONDA_BIN}"
echo "Environment:  ${ENV_DIR}"
echo "Python:       ${PYTHON_VERSION}"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${CONDA_BIN}" create -y -p "${ENV_DIR}" "python=${PYTHON_VERSION}" pip
else
  echo "Environment already exists, reusing: ${ENV_DIR}"
fi

PYTHON_BIN="${ENV_DIR}/bin/python"
"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel

if [[ -n "${TORCH_INDEX_URL}" ]]; then
  "${PYTHON_BIN}" -m pip install --index-url "${TORCH_INDEX_URL}" torch torchvision
else
  "${PYTHON_BIN}" -m pip install torch torchvision
fi

"${PYTHON_BIN}" -m pip install \
  numpy \
  opencv-python-headless \
  PyYAML \
  Pillow \
  matplotlib \
  ultralytics

"${PYTHON_BIN}" -m pip check

PROJECT_ROOT="${PROJECT_ROOT}" "${PYTHON_BIN}" - <<'PY'
import os
import sys
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"])
sys.path.insert(0, str(project_root))

import cv2  # noqa: F401
import matplotlib  # noqa: F401
import numpy  # noqa: F401
import PIL  # noqa: F401
import torch  # noqa: F401
import torchvision  # noqa: F401
import ultralytics  # noqa: F401
import yaml  # noqa: F401

from src.stage1.efficientnet import EfficientNetKeypointModel  # noqa: F401
from src.stage1.yolo_strategy import normalize_input_shape  # noqa: F401
from src.stage2.model import VUOrdinalEfficientNet  # noqa: F401

print("Smoke test passed.")
PY

cat <<EOF

Conda environment is ready.

Activate it with:
  source "${CONDA_ROOT}/bin/activate" "${ENV_DIR}"

Run existing scripts with this interpreter, for example:
  STAGE1_PYTHON="${ENV_DIR}/bin/python" bash "${PROJECT_ROOT}/src/stage1/prepare_data.sh" --check-only
  STAGE1_PYTHON="${ENV_DIR}/bin/python" bash "${PROJECT_ROOT}/src/stage1/train_efficientnet.sh" --help
  STAGE2_PYTHON="${ENV_DIR}/bin/python" bash "${PROJECT_ROOT}/src/stage2/train_efficientnet.sh" --help

For a specific PyTorch wheel source, rerun with TORCH_INDEX_URL set, for example:
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121" bash "${SCRIPT_DIR}/create_venv.sh"
EOF
