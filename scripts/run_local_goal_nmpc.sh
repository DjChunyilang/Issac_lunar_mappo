#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ACADOS_SOURCE_DIR="$repo_root/.venv_isaaclab/acados"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$repo_root"
exec "$repo_root/.venv_isaaclab/bin/python" scripts/train_local_goal_nmpc.py "$@"
