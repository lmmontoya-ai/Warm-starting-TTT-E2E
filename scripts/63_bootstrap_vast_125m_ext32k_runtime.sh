#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-$PWD}"
VENV_DIR="${2:-.venv-ext32k}"
REQ_FILE="${3:-requirements/125m_ext32k_runtime.txt}"

if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Expected repo root with pyproject.toml, got: ${REPO_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/${REQ_FILE}" ]]; then
  echo "Missing requirements file: ${REPO_ROOT}/${REQ_FILE}" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
  awscli \
  build-essential \
  ca-certificates \
  curl \
  git \
  rsync \
  unzip

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="${HOME}/.local/bin:${PATH}"
export UV_LINK_MODE=copy
export HF_HUB_ENABLE_HF_TRANSFER=1
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "${REPO_ROOT}"
uv python install 3.12
uv venv --python 3.12 "${VENV_DIR}"
uv pip sync --python "${VENV_DIR}/bin/python" "${REQ_FILE}"
uv pip install --python "${VENV_DIR}/bin/python" --no-deps -e .

echo "Pinned 32K extension runtime bootstrapped in ${REPO_ROOT}/${VENV_DIR}"
