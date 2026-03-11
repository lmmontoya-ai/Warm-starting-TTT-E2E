"""Single-process sharding helpers for the parity runtime."""

from __future__ import annotations

import jax


def local_device_summary() -> dict[str, object]:
    return {
        "backend": jax.default_backend(),
        "device_count": int(jax.device_count()),
        "local_device_count": int(jax.local_device_count()),
        "process_count": int(jax.process_count()),
        "process_index": int(jax.process_index()),
    }
