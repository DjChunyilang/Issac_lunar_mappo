#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLANNER_PYTHON="${ROOT_DIR}/.venv_terrain_cuda/bin/python"
BOOTSTRAP_PYTHON="${ROOT_DIR}/.venv_isaaclab/bin/python"
if [[ ! -x "${PLANNER_PYTHON}" ]]; then
  bash "${ROOT_DIR}/scripts/install_terrain_cuda.sh"
fi
"${PLANNER_PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA planner requires a working GPU environment"'
"${BOOTSTRAP_PYTHON}" -m pip --python "${PLANNER_PYTHON}" install \
  -r "${ROOT_DIR}/requirements-planner-cuda.txt"
"${BOOTSTRAP_PYTHON}" -m pip --python "${PLANNER_PYTHON}" check
"${PLANNER_PYTHON}" - <<'PY'
import osqp
import torch
assert osqp.__version__ == "1.0.4"
assert torch.arange(16., device="cuda").square().sum().item() == 1240.
print("Planner CUDA ready:", torch.__version__, torch.cuda.get_device_name(0), "OSQP", osqp.__version__)
PY
