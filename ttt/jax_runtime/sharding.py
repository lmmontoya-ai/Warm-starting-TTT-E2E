"""Single-process local-device sharding helpers for the parity runtime."""

from __future__ import annotations

import equinox as eqx
import jax


def local_device_summary() -> dict[str, object]:
    return {
        "backend": jax.default_backend(),
        "device_count": int(jax.device_count()),
        "local_device_count": int(jax.local_device_count()),
        "process_count": int(jax.process_count()),
        "process_index": int(jax.process_index()),
    }


def ensure_local_batch_divisible(global_batch_size: int) -> int:
    local_device_count = int(jax.local_device_count())
    if local_device_count <= 0:
        raise ValueError("No local JAX devices are available.")
    if global_batch_size % local_device_count != 0:
        raise ValueError(
            f"global_batch_size={global_batch_size} must be divisible by "
            f"local_device_count={local_device_count}"
        )
    return local_device_count


def reshape_batch_for_local_devices(batch):
    local_device_count = ensure_local_batch_divisible(int(batch.input_ids.shape[0]))
    per_device_batch = int(batch.input_ids.shape[0]) // local_device_count
    return jax.tree.map(
        lambda x: x.reshape((local_device_count, per_device_batch) + x.shape[1:]),
        batch,
    )


def replicate_pytree(tree):
    devices = jax.local_devices()
    return jax.tree.map(
        lambda x: jax.device_put_replicated(x, devices) if eqx.is_array(x) else x,
        tree,
    )


def unreplicate_pytree(tree):
    return jax.tree.map(
        lambda x: jax.device_get(x[0]) if eqx.is_array(x) else x,
        tree,
    )
