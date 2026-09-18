#!/usr/bin/env bash
# Install only the proxy training stack; Isaac Sim is a separate optional stack.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROXY_ENV="${ROOT_DIR}/.venv_isaaclab"
BASE_PYTHON="${PROXY_BASE_PYTHON:-python3.12}"
PROXY_PYTHON="${PROXY_ENV}/bin/python"

if [[ ! -x "${PROXY_PYTHON}" ]]; then
  "${BASE_PYTHON}" -m venv "${PROXY_ENV}"
fi
"${PROXY_PYTHON}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
"${PROXY_PYTHON}" -m pip install --upgrade pip setuptools wheel
TORCH_INDEX="${PROXY_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
TORCH_SPEC="${PROXY_TORCH_SPEC:-torch==2.10.0}"
"${PROXY_PYTHON}" -m pip install --index-url "${TORCH_INDEX}" "${TORCH_SPEC}"
"${PROXY_PYTHON}" -m pip install -r "${ROOT_DIR}/requirements-proxy.txt"
"${PROXY_PYTHON}" -m pip install --no-deps -e "${ROOT_DIR}/source/lunar_rover_tasks"
"${PROXY_PYTHON}" -m pip check
