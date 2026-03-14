"""Minimal JAX utilities shared by the parity runtime."""

from __future__ import annotations

import os
import random
from collections.abc import Callable, Hashable
from functools import partial
from typing import Any, Literal, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from einops import rearrange
from jaxtyping import PRNGKeyArray, PyTree

from ttt.config import JaxDistributedConfig

T = TypeVar("T", bound=PyTree)


def master_log(logger, *args, level: int = 20, **kwargs) -> None:
    if jax.process_index() == 0:
        logger.log(level, *args, **kwargs)


def initialize_distibuted(distributed_config: JaxDistributedConfig) -> None:
    if distributed_config.backend:
        os.environ["JAX_PLATFORM_NAME"] = distributed_config.backend
        if distributed_config.backend == "cpu":
            core_count = distributed_config.num_devices or os.cpu_count()
            if core_count:
                os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={core_count}"

    if not distributed_config.distributed:
        return

    local_device_ids = None
    if distributed_config.local_device_ids:
        local_device_ids = [int(x) for x in distributed_config.local_device_ids.split(",")]
    jax.distributed.initialize(
        coordinator_address=distributed_config.coordinator_address,
        num_processes=distributed_config.num_processes,
        process_id=distributed_config.process_id,
        local_device_ids=local_device_ids,
    )


def set_random_seed(seed: int) -> PRNGKeyArray:
    np.random.seed(seed)
    random.seed(seed)
    return jrandom.PRNGKey(seed)


def get_float_dtype_by_name(dtype: str):
    match dtype:
        case "bf16" | "bfloat16":
            return jnp.bfloat16
        case "fp16" | "float16":
            return jnp.float16
        case "fp32" | "float32":
            return jnp.float32
        case "fp64" | "float64":
            return jnp.float64
        case _:
            raise ValueError(f"Unknown dtype: {dtype}")


def tree_slice(tree: T, i: int) -> T:
    return jax.tree.map(lambda x: x[i], tree)


def tree_rearrange(tree: T, pattern: str, **axes_lengths) -> T:
    return jax.tree.map(lambda x: rearrange(x, pattern, **axes_lengths), tree)


def clone_pytree(tree: T) -> T:
    return jax.tree.map(lambda x: jnp.array(x, copy=True) if hasattr(x, "shape") else x, tree)


def canonicalize_dtype(*args, dtype=None, inexact: bool = True):
    if dtype is None:
        filtered = [jnp.asarray(x) for x in args if x is not None]
        dtype = jnp.result_type(*filtered)
        if inexact and not jnp.issubdtype(dtype, jnp.inexact):
            dtype = jnp.promote_types(jnp.float32, dtype)
    if inexact and not jnp.issubdtype(dtype, jnp.inexact):
        raise ValueError(f"Dtype must be inexact: {dtype}")
    return dtype


def promote_dtype(*args, dtype=None, inexact: bool = True) -> list[Any]:
    dtype = canonicalize_dtype(*args, dtype=dtype, inexact=inexact)
    return [jnp.asarray(x, dtype) if x is not None else None for x in args]


def get_gradient_checkpoint_policy(
    name: Literal[
        "everything_saveable",
        "nothing_saveable",
        "checkpoint_dots",
        "checkpoint_dots_with_no_batch_dims",
    ]
    | Callable[..., bool]
    | str
):
    if not isinstance(name, str):
        return name
    match name:
        case "everything_saveable":
            return jax.checkpoint_policies.everything_saveable
        case "nothing_saveable":
            return jax.checkpoint_policies.nothing_saveable
        case "checkpoint_dots":
            return jax.checkpoint_policies.checkpoint_dots
        case "checkpoint_dots_with_no_batch_dims":
            return jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
        case "":
            return None
        case _:
            raise ValueError(f"Unknown gradient checkpoint policy: {name}")


def remat_bwd(
    fun: Callable[..., Any],
    *,
    prevent_cse: bool = True,
    static_argnums: int | tuple[int, ...] = (),
    policy: Callable[..., bool] | None = None,
):
    @jax.custom_vjp
    def standard_fn(*args):
        return fun(*args)

    @partial(jax.remat, prevent_cse=prevent_cse, policy=policy, static_argnums=static_argnums)
    def fwd(*args):
        out, vjp_fn = jax.vjp(fun, *args)
        return out, vjp_fn

    @partial(jax.remat, prevent_cse=prevent_cse, policy=policy, static_argnums=static_argnums)
    def bwd(vjp_fn, grads):
        return vjp_fn(grads)

    standard_fn.defvjp(fwd, bwd)
    return jax.remat(
        standard_fn,
        prevent_cse=prevent_cse,
        policy=policy,
        static_argnums=static_argnums,
    )


def maybe_remat(
    fun: Callable[..., Any],
    *,
    prevent_cse: bool = True,
    static_argnums: int | tuple[int, ...] = (),
    policy: str | Callable[..., bool] = "",
):
    resolved_policy = get_gradient_checkpoint_policy(policy)
    if resolved_policy is None:
        return fun
    return jax.remat(
        fun,
        prevent_cse=prevent_cse,
        static_argnums=static_argnums,
        policy=resolved_policy,
    )


def maybe_remat_bwd(
    fun: Callable[..., Any],
    *,
    prevent_cse: bool = True,
    static_argnums: int | tuple[int, ...] = (),
    policy: str | Callable[..., bool] = "",
):
    resolved_policy = get_gradient_checkpoint_policy(policy)
    if resolved_policy is None:
        return fun
    return remat_bwd(
        fun,
        prevent_cse=prevent_cse,
        static_argnums=static_argnums,
        policy=resolved_policy,
    )


def maybe_double_remat(
    fun: Callable[..., Any],
    *,
    prevent_cse: bool = True,
    policy_remat: str | Callable[..., bool] = "",
    policy_remat_bwd: str | Callable[..., bool] = "",
    static_argnums: int | tuple[int, ...] = (),
):
    return maybe_remat_bwd(
        maybe_remat(
            fun,
            prevent_cse=prevent_cse,
            static_argnums=static_argnums,
            policy=policy_remat,
        ),
        prevent_cse=prevent_cse,
        static_argnums=static_argnums,
        policy=policy_remat_bwd,
    )


def scan_or_loop(f, init, xs, use_loop: bool = False):
    if not use_loop:
        return jax.lax.scan(f, init, xs)

    carry = init
    ys = []
    n = jax.tree.leaves(xs)[0].shape[0]
    for i in range(n):
        x = tree_slice(xs, i)
        carry, y = f(carry, x)
        ys.append(y)
    return carry, jax.tree.map(
        lambda *parts: None if all(part is None for part in parts) else jnp.stack(parts),
        *ys,
    )


def welfords_online_mean(fun, batch):
    num_loops = jax.tree.leaves(batch)[0].shape[0]
    if num_loops == 1:
        return fun(jax.tree.map(lambda x: x[0], batch))

    first_batch_slice = jax.tree.map(lambda x: x[0], batch)
    acc_init = jax.tree.map(lambda x: jnp.zeros_like(x), jax.eval_shape(fun, first_batch_slice))
    count_init = jnp.array(0, dtype=jnp.int32)

    def update_online_mean(carry, batch_slice):
        acc_carry, count = carry
        value = fun(batch_slice)
        next_count = count + 1
        acc_carry = jax.tree.map(
            lambda acc, current: acc + (current - acc) / next_count.astype(current.dtype),
            acc_carry,
            value,
        )
        return (acc_carry, next_count), None

    (acc_result, _count), _ = jax.lax.scan(update_online_mean, (acc_init, count_init), batch)
    return acc_result


def vmap_mean(fun, batch, *, axis_name: Hashable):
    vmap_dim_size = jax.tree.flatten(batch)[0][0].shape[0]
    if vmap_dim_size == 1:
        single_microbatch = tree_rearrange(batch, "1 ... -> ...")
        return fun(single_microbatch)

    @partial(jax.vmap, in_axes=(0,), out_axes=None, axis_name=axis_name)
    def vmapped_fn(x):
        return jax.lax.pmean(fun(x), axis_name=axis_name)

    return vmapped_fn(batch)


def global_norm_safe(updates):
    leaves = jax.tree.leaves(updates)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(
        sum(
            jnp.sum(jnp.square(leaf.astype(jnp.float32)))
            for leaf in leaves
            if leaf is not None
        )
    )


def scan_remat_chunk(f, carry, xs, *, remat_n_loops: int, unroll: bool):
    num_loops = jax.tree.leaves(xs)[0].shape[0]
    if remat_n_loops <= 0 or remat_n_loops >= num_loops:
        return scan_or_loop(f, carry, xs, use_loop=unroll)

    n_chunks = num_loops // remat_n_loops
    grouped = tree_rearrange(
        xs,
        "(chunk inner) ... -> chunk inner ...",
        chunk=n_chunks,
        inner=remat_n_loops,
    )

    @partial(
        jax.remat,
        prevent_cse=False,
        policy=get_gradient_checkpoint_policy("nothing_saveable"),
    )
    def chunk_f(chunk_carry, chunk_xs):
        return scan_or_loop(f, chunk_carry, chunk_xs, use_loop=unroll)

    carry, result = scan_or_loop(chunk_f, carry, grouped, use_loop=unroll)
    result = tree_rearrange(result, "chunk inner ... -> (chunk inner) ...")
    return carry, result
