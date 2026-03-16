from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


def _parse_version_tuple(raw: str | None) -> tuple[int, ...] | None:
    if not raw:
        return None
    parts: list[int] = []
    for item in raw.split("."):
        item = item.strip()
        if not item or not item.isdigit():
            break
        parts.append(int(item))
    return tuple(parts) if parts else None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _query_driver_version() -> str | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            return line
    return None


def _probe_library_load(env: dict[str, str], library: str) -> tuple[bool, str | None]:
    probe = (
        "import ctypes, sys\n"
        f"ctypes.CDLL({library!r})\n"
        "sys.stdout.write('ok\\n')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return True, None
    error = (proc.stderr or proc.stdout or "").strip() or f"returncode={proc.returncode}"
    return False, error


def _candidate_nvrtc_dirs() -> list[str]:
    candidates = ["/opt/conda/lib", "/usr/local/cuda/lib64", "/usr/local/nvidia/lib64", "/usr/local/nvidia/lib"]
    resolved: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = Path(candidate)
        if not path.exists():
            continue
        if not any(path.glob("libnvrtc.so*")):
            continue
        value = str(path.resolve())
        if value in seen:
            continue
        resolved.append(value)
        seen.add(value)
    return resolved


def _prepend_ld_library_path(env: dict[str, str], directory: str) -> dict[str, str]:
    updated = env.copy()
    current = updated.get("LD_LIBRARY_PATH", "").strip()
    if current:
        parts = [item for item in current.split(":") if item]
        if directory not in parts:
            updated["LD_LIBRARY_PATH"] = ":".join([directory, *parts])
    else:
        updated["LD_LIBRARY_PATH"] = directory
    return updated


def _minimum_driver_for_cuda_runtime(cuda_runtime_version: str | None) -> str | None:
    version = _parse_version_tuple(cuda_runtime_version)
    if not version:
        return None
    if version[:2] >= (12, 9):
        return "575.51.03"
    return None


@dataclass(frozen=True)
class CudaPreflight:
    status: str
    issues: list[str]
    driver_version: str | None
    minimum_driver_version: str | None
    driver_meets_packaged_runtime: bool | None
    cuda_runtime_package_version: str | None
    cuda_nvcc_package_version: str | None
    jax_version: str | None
    jaxlib_version: str | None
    libnvrtc_loadable_default: bool
    libnvrtc_load_error_default: str | None
    libnvrtc_loadable_effective: bool
    libnvrtc_load_error_effective: str | None
    ld_library_path_default: str
    ld_library_path_effective: str
    ld_library_path_patch: list[str]
    candidate_nvrtc_dirs: list[str]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def prepare_cuda_runtime_env(base_env: dict[str, str] | None = None) -> tuple[dict[str, str], CudaPreflight]:
    env = os.environ.copy()
    if base_env:
        env.update(base_env)

    driver_version = _query_driver_version()
    cuda_runtime_package_version = _package_version("nvidia-cuda-runtime-cu12")
    cuda_nvcc_package_version = _package_version("nvidia-cuda-nvcc-cu12")
    minimum_driver_version = _minimum_driver_for_cuda_runtime(
        cuda_runtime_package_version or cuda_nvcc_package_version
    )

    driver_meets_packaged_runtime: bool | None = None
    if driver_version and minimum_driver_version:
        driver_tuple = _parse_version_tuple(driver_version)
        minimum_tuple = _parse_version_tuple(minimum_driver_version)
        if driver_tuple and minimum_tuple:
            driver_meets_packaged_runtime = driver_tuple >= minimum_tuple

    libnvrtc_loadable_default, libnvrtc_load_error_default = _probe_library_load(env, "libnvrtc.so.12")
    candidate_nvrtc_dirs = _candidate_nvrtc_dirs()
    ld_library_path_patch: list[str] = []
    effective_env = env.copy()
    libnvrtc_loadable_effective = libnvrtc_loadable_default
    libnvrtc_load_error_effective = libnvrtc_load_error_default

    if not libnvrtc_loadable_default:
        for candidate in candidate_nvrtc_dirs:
            trial_env = _prepend_ld_library_path(effective_env, candidate)
            load_ok, load_error = _probe_library_load(trial_env, "libnvrtc.so.12")
            if not load_ok:
                continue
            effective_env = trial_env
            ld_library_path_patch.append(candidate)
            libnvrtc_loadable_effective = True
            libnvrtc_load_error_effective = None
            break

    issues: list[str] = []
    if not libnvrtc_loadable_default:
        issues.append("libnvrtc_missing_default_loader")
    if ld_library_path_patch:
        issues.append("ld_library_path_patched")
    if not libnvrtc_loadable_effective:
        issues.append("libnvrtc_unloadable")
    if driver_meets_packaged_runtime is False:
        issues.append("driver_below_packaged_cuda_runtime")

    status = "ok"
    if "libnvrtc_unloadable" in issues or "driver_below_packaged_cuda_runtime" in issues:
        status = "incompatible_runtime"

    return (
        effective_env,
        CudaPreflight(
            status=status,
            issues=issues,
            driver_version=driver_version,
            minimum_driver_version=minimum_driver_version,
            driver_meets_packaged_runtime=driver_meets_packaged_runtime,
            cuda_runtime_package_version=cuda_runtime_package_version,
            cuda_nvcc_package_version=cuda_nvcc_package_version,
            jax_version=_package_version("jax"),
            jaxlib_version=_package_version("jaxlib"),
            libnvrtc_loadable_default=libnvrtc_loadable_default,
            libnvrtc_load_error_default=libnvrtc_load_error_default,
            libnvrtc_loadable_effective=libnvrtc_loadable_effective,
            libnvrtc_load_error_effective=libnvrtc_load_error_effective,
            ld_library_path_default=env.get("LD_LIBRARY_PATH", ""),
            ld_library_path_effective=effective_env.get("LD_LIBRARY_PATH", ""),
            ld_library_path_patch=ld_library_path_patch,
            candidate_nvrtc_dirs=candidate_nvrtc_dirs,
        ),
    )
