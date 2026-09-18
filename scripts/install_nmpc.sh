#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
proxy_python="$repo_root/.venv_isaaclab/bin/python"
acados_root="$repo_root/.venv_isaaclab/acados"
acados_commit="59d93e17d2985fdd73fc58b8a83ed8f83a024171"
test -x "$proxy_python"
"$proxy_python" -m pip install -r "$repo_root/requirements-nmpc.txt"
if [ ! -d "$acados_root" ]; then
  git -c http.version=HTTP/1.1 clone --depth 1 --branch v0.5.5 https://github.com/acados/acados.git "$acados_root"
fi
if [ "$(git -C "$acados_root" rev-parse HEAD)" != "$acados_commit" ]; then
  echo "Existing acados checkout does not match the pinned commit; refusing overwrite." >&2
  exit 1
fi
git -C "$acados_root" -c http.version=HTTP/1.1 submodule update --init --depth 1 external/blasfeo external/hpipm
"$repo_root/.venv_isaaclab/bin/cmake" -S "$acados_root" -B "$acados_root/build" \
  -DCMAKE_BUILD_TYPE=Release -DACADOS_WITH_OPENMP=OFF -DBLASFEO_TARGET=X64_AUTOMATIC \
  -DCMAKE_INSTALL_PREFIX="$acados_root" -DCMAKE_POLICY_VERSION_MINIMUM=3.5
"$repo_root/.venv_isaaclab/bin/cmake" --build "$acados_root/build" -j4
"$repo_root/.venv_isaaclab/bin/cmake" --install "$acados_root/build"
"$proxy_python" -m pip install -e "$acados_root/interfaces/acados_template"
ACADOS_SOURCE_DIR="$acados_root" "$proxy_python" -c \
  'import os; from acados_template.utils import get_tera,get_tera_exec_path; get_tera(force_download=True) if not os.access(get_tera_exec_path(),os.X_OK) else None'
"$proxy_python" -m pip check
