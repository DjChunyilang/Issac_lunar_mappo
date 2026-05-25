#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv_isaaclab/bin/python"

"${PYTHON}" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0 torchvision==0.25.0
"${PYTHON}" -m pip install --extra-index-url https://pypi.nvidia.com --pre "isaacsim[all,extscache]==6.0.0"

mkdir -p "${ROOT_DIR}/external"
if [[ ! -d "${ROOT_DIR}/external/IsaacLab/.git" ]]; then
  git clone --branch v3.0.0-beta https://github.com/isaac-sim/IsaacLab.git "${ROOT_DIR}/external/IsaacLab"
fi

cd "${ROOT_DIR}/external/IsaacLab"
"${PYTHON}" -m pip install "setuptools<82" cmake
"${PYTHON}" -m pip install --extra-index-url https://pypi.nvidia.com \
  -e "${ROOT_DIR}/external/IsaacLab/source/isaaclab" \
  -e "${ROOT_DIR}/external/IsaacLab/source/isaaclab_assets" \
  -e "${ROOT_DIR}/external/IsaacLab/source/isaaclab_tasks" \
  -e "${ROOT_DIR}/external/IsaacLab/source/isaaclab_rl[skrl]"
"${PYTHON}" -m pip install -e "${ROOT_DIR}/source/lunar_rover_tasks"
