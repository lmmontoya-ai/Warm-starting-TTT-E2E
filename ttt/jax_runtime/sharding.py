"""Sharding/device helpers for native JAX runtime."""

from __future__ import annotations

import jax


def local_device_summary() -> dict[str, int | str]:
    backend = jax.default_backend()
    return {
        "backend": backend,
        "process_count": int(jax.process_count()),
        "process_index": int(jax.process_index()),
        "device_count": int(jax.device_count()),
        "local_device_count": int(jax.local_device_count()),
    }


def pick_compute_device() -> str:
    return str(jax.default_backend())
