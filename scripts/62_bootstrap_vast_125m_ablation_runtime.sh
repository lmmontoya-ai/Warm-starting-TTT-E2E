#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-$PWD}"

if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Expected repo root with pyproject.toml, got: ${REPO_ROOT}" >&2
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
uv sync --frozen

echo "Pinned Vast ablation runtime bootstrapped in ${REPO_ROOT}"
