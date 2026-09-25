#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TERRAIN_ENV="${ROOT_DIR}/.venv_terrain_cuda"
TERRAIN_PYTHON="${TERRAIN_ENV}/bin/python"
BOOTSTRAP_PYTHON="${ROOT_DIR}/.venv_isaaclab/bin/python"
python3.12 -m venv --without-pip "${TERRAIN_ENV}"
"${BOOTSTRAP_PYTHON}" -m pip --python "${TERRAIN_PYTHON}" install \
  torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
"${BOOTSTRAP_PYTHON}" -m pip --python "${TERRAIN_PYTHON}" install \
  -r "${ROOT_DIR}/requirements-terrain-cuda.txt" -e "${ROOT_DIR}/source/lunar_rover_tasks"
"${BOOTSTRAP_PYTHON}" -m pip --python "${TERRAIN_PYTHON}" check
"${TERRAIN_PYTHON}" -c 'import torch; assert torch.cuda.is_available(); assert "sm_75" in torch.cuda.get_arch_list(); print(torch.__version__, torch.cuda.get_device_name())'
