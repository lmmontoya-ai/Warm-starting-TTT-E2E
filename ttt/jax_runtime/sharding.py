"""Mesh- and state-sharding helpers for the parity runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import equinox as eqx
import jax
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxtyping import PyTree

from ttt.config import Config
from ttt.utils.jax_utils import tree_rearrange

from .model.transformer import MetaModel

T = TypeVar("T", bound=PyTree)


def local_device_summary() -> dict[str, object]:
    return {
        "backend": jax.default_backend(),
        "device_count": int(jax.device_count()),
        "local_device_count": int(jax.local_device_count()),
        "process_count": int(jax.process_count()),
        "process_index": int(jax.process_index()),
    }


def prepare_data_parallelism(cfg: Config, global_dev_num: int | None = None) -> int:
    if global_dev_num is None:
        global_dev_num = int(jax.device_count())
    if cfg.training.n_data_parallel is None:
        if global_dev_num % int(cfg.training.n_state_parallel) != 0:
            raise ValueError(
                "Number of devices must be divisible by state parallelism: "
                f"{global_dev_num} !% {cfg.training.n_state_parallel}"
            )
        cfg.training.n_data_parallel = global_dev_num // int(cfg.training.n_state_parallel)
    if int(cfg.training.n_data_parallel) * int(cfg.training.n_state_parallel) != global_dev_num:
        raise ValueError(
            "Data/state parallelism does not match the number of devices: "
            f"{cfg.training.n_data_parallel} * {cfg.training.n_state_parallel} != {global_dev_num}"
        )
    return int(cfg.training.n_data_parallel)


def shard_fn(
    tree: T,
    mesh: jax.sharding.Mesh,
    where_spec_pairs: list[tuple[Callable[[MetaModel], tuple[Any, ...]], P]],
) -> T:
    for where, spec in where_spec_pairs:
        tree = eqx.tree_at(
            where,
            tree,
            replace_fn=lambda x: jax.lax.with_sharding_constraint(x, NamedSharding(mesh, spec)),
            is_leaf=lambda x: x is None,
        )
    return tree


class ModelSharding:
    def __init__(self, cfg: Config, mesh: jax.sharding.Mesh | None = None):
        self.config = cfg
        self.mesh = mesh
        if self.mesh is None:
            n_data_parallel = prepare_data_parallelism(cfg)
            self.mesh = jax.make_mesh(
                axis_shapes=(n_data_parallel, int(cfg.training.n_state_parallel)),
                axis_names=("data", "state"),
            )

    def data_sharding(self) -> NamedSharding:
        return NamedSharding(self.mesh, P("data"))

    def replicated_sharding(self) -> NamedSharding:
        return NamedSharding(self.mesh, P())

    def shard_params(self, model: MetaModel) -> MetaModel:
        shard_cfg = [
            (lambda m: (m.language_model.model.ln_f,), P("state")),
            (
                lambda m: (
                    m.language_model.model.wte,
                    m.language_model.model.h.blocks.seq_norm,
                    m.language_model.model.h.blocks.ffn_norm,
                    m.language_model.lm_head,
                ),
                P(None, "state"),
            ),
            (
                lambda m: (
                    m.language_model.model.h.blocks.seq_modeling_block.wq,
                    m.language_model.model.h.blocks.seq_modeling_block.wk,
                    m.language_model.model.h.blocks.seq_modeling_block.wv,
                    m.language_model.model.h.blocks.feed_forward.w1,
                    m.language_model.model.h.blocks.feed_forward.w3,
                ),
                P(None, "state", None),
            ),
            (
                lambda m: (
                    m.language_model.model.h.blocks.feed_forward.w2,
                    m.language_model.model.h.blocks.seq_modeling_block.wo,
                ),
                P(None, None, "state"),
            ),
        ]

        if self.config.model.prime:
            shard_cfg.extend(
                [
                    (
                        lambda m: (m.language_model.model.h.prime_storage.ffn_prime_norm,),
                        P(None, "state"),
                    ),
                    (
                        lambda m: (
                            m.language_model.model.h.prime_storage.feed_forward_prime.w1,
                            m.language_model.model.h.prime_storage.feed_forward_prime.w3,
                        ),
                        P(None, "state", None),
                    ),
                    (
                        lambda m: (m.language_model.model.h.prime_storage.feed_forward_prime.w2,),
                        P(None, None, "state"),
                    ),
                ]
            )

        return shard_fn(model, self.mesh, shard_cfg)


def put_replicated(tree: T, sharding: NamedSharding) -> T:
    return jax.tree.map(
        lambda x: jax.device_put(x, sharding) if eqx.is_array(x) else x,
        tree,
    )


def make_global_array(
    arr,
    *,
    data_sharding: NamedSharding,
    global_batch_size: int,
):
    return jax.make_array_from_process_local_data(
        sharding=data_sharding,
        local_data=arr,
        global_shape=(global_batch_size, *arr.shape[1:]),
    )


def to_data_parallel_batch(
    batch,
    *,
    data_sharding: NamedSharding,
    global_batch_size: int,
    n_data_parallel: int,
):
    batch = jax.tree.map(
        lambda x: make_global_array(
            x,
            data_sharding=data_sharding,
            global_batch_size=global_batch_size,
        ),
        batch,
    )
    return tree_rearrange(
        batch,
        "(data_parallel batch) ... -> data_parallel batch ...",
        data_parallel=n_data_parallel,
    )
