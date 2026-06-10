#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-$PWD}"

if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Expected repo root with pyproject.toml, got: ${REPO_ROOT}" >&2
  exit 1
fi

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

export DEBIAN_FRONTEND=noninteractive
${SUDO} apt-get update
${SUDO} apt-get install -y \
  awscli \
  build-essential \
  ca-certificates \
  curl \
  git \
  jq \
  rsync \
  unzip

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="${HOME}/.local/bin:${PATH}"
export UV_LINK_MODE=copy
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "${REPO_ROOT}"
uv python install 3.12
uv sync --frozen

uv run --exact python - <<'PY'
import jax
import orbax.checkpoint as ocp

devices = jax.devices()
print(f"jax={jax.__version__} orbax={ocp.__version__}")
print(f"backend={jax.default_backend()} device_count={jax.device_count()} local_device_count={jax.local_device_count()}")
for index, device in enumerate(devices):
    print(f"device[{index}]={device}")
if jax.default_backend() != "gpu":
    raise SystemExit("Expected JAX GPU backend.")
if jax.device_count() != 2:
    raise SystemExit(f"Expected exactly 2 JAX devices for revision_v2_prime_h100_2x, got {jax.device_count()}.")
PY

echo "Revision-v2 Prime 2xH100 runtime bootstrapped in ${REPO_ROOT}"
