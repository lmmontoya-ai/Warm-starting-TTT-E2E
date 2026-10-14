#!/usr/bin/env bash
# Prepare a fresh Linux GPU host (any provider) for this repo:
#   1. install the system packages the scripts call (skip with --skip-apt);
#   2. install uv if it is missing;
#   3. install Python 3.12 and sync the locked project environment;
#   4. check that JAX sees the GPU backend and the expected number of devices.
#
# Usage:
#   scripts/bootstrap_gpu_host.sh [--repo-root DIR] [--expected-gpus N] [--skip-apt]
#                                 [--venv DIR --requirements FILE]
#
#   --repo-root DIR       repository checkout (default: the parent of this script's directory)
#   --expected-gpus N     fail unless JAX reports exactly N devices
#                         (default: the number of GPUs listed by nvidia-smi, if available)
#   --skip-apt            do not install system packages (no root or sudo needed)
#   --venv DIR            build a pip venv from --requirements instead of `uv sync --frozen`;
#                         use only for pinned legacy runtimes (see configs/research/vast_runtime_profiles.yaml)
#   --requirements FILE   requirements file for --venv, relative to the repo root
#
# The AWS CLI is not installed here; install it yourself if you use
# scripts/28_fetch_s3_dataset.py to pull data from your own S3-compatible bucket.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPECTED_GPUS=""
SKIP_APT=0
VENV_DIR=""
REQ_FILE=""

usage() {
  sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --expected-gpus) EXPECTED_GPUS="$2"; shift 2 ;;
    --skip-apt) SKIP_APT=1; shift ;;
    --venv) VENV_DIR="$2"; shift 2 ;;
    --requirements) REQ_FILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Expected a repo root containing pyproject.toml, got: ${REPO_ROOT}" >&2
  exit 1
fi
if [[ -n "${VENV_DIR}" && -z "${REQ_FILE}" ]] || [[ -z "${VENV_DIR}" && -n "${REQ_FILE}" ]]; then
  echo "--venv and --requirements must be given together." >&2
  exit 2
fi
if [[ -n "${REQ_FILE}" && ! -f "${REPO_ROOT}/${REQ_FILE}" ]]; then
  echo "Missing requirements file: ${REPO_ROOT}/${REQ_FILE}" >&2
  exit 1
fi

if [[ "${SKIP_APT}" -eq 0 ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    if [[ "$(id -u)" -eq 0 ]]; then
      SUDO=""
    else
      SUDO="sudo"
    fi
    export DEBIAN_FRONTEND=noninteractive
    ${SUDO} apt-get update
    ${SUDO} apt-get install -y build-essential ca-certificates curl git jq rsync unzip
  else
    echo "apt-get not found; skipping system packages (install curl and git yourself)." >&2
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"
export UV_LINK_MODE=copy

# CUDA libraries shipped with the driver image; harmless when the paths do not exist.
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ -z "${EXPECTED_GPUS}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  EXPECTED_GPUS="$(nvidia-smi -L | grep -c '^GPU ' || true)"
fi
export BOOTSTRAP_EXPECTED_GPUS="${EXPECTED_GPUS}"

cd "${REPO_ROOT}"
uv python install 3.12

if [[ -n "${VENV_DIR}" ]]; then
  uv venv --python 3.12 "${VENV_DIR}"
  uv pip sync --python "${VENV_DIR}/bin/python" "${REQ_FILE}"
  uv pip install --python "${VENV_DIR}/bin/python" --no-deps -e .
  PYTHON_CMD=("${VENV_DIR}/bin/python")
else
  uv sync --frozen
  PYTHON_CMD=(uv run --frozen python)
fi

"${PYTHON_CMD[@]}" - <<'PY'
import os

import jax
import orbax.checkpoint as ocp

print(f"jax={jax.__version__} orbax={ocp.__version__}")
print(f"backend={jax.default_backend()} device_count={jax.device_count()}")
for index, device in enumerate(jax.devices()):
    print(f"device[{index}]={device}")
if jax.default_backend() != "gpu":
    raise SystemExit("JAX did not select the GPU backend; check the NVIDIA driver and CUDA wheels.")
expected = os.environ.get("BOOTSTRAP_EXPECTED_GPUS", "").strip()
if expected and jax.device_count() != int(expected):
    raise SystemExit(f"Expected {expected} JAX devices, got {jax.device_count()}.")
PY

echo "GPU host ready: ${REPO_ROOT}"
