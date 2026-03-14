"""Reference-shaped train/eval loop helpers for the parity runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
from equinox import nn
from optax import GradientTransformation, OptState

from ttt.config import Config
from ttt.dataloader import dummy_dataset, lm_dataset
from ttt.utils.filter_utils import filter_apply_updates, get_filter_spec
from ttt.utils.jax_utils import global_norm_safe, tree_rearrange, vmap_mean, welfords_online_mean

from .model.data import Batch
from .model.transformer import MetaModel

M = MetaModel.MetricType


def _to_sharded_batch(batch, *, data_sharding, global_batch_size: int, n_data_parallel: int):
    def load_to_sharded_array(arr):
        return jax.make_array_from_process_local_data(
            sharding=data_sharding,
            local_data=arr,
            global_shape=(global_batch_size, *arr.shape[1:]),
        )

    batch = jax.tree.map(load_to_sharded_array, batch)
    return tree_rearrange(
        batch,
        "(data_parallel batch) ... -> data_parallel batch ...",
        data_parallel=n_data_parallel,
    )


def make_train_step(cfg: Config, optimizer: GradientTransformation):
    @eqx.filter_jit(donate="all-except-first")
    @eqx.filter_vmap(in_axes=(None, None, None, 0), out_axes=None, axis_name="data_parallel")
    def train_on_sequence(
        state: nn.State,
        meta_model: MetaModel,
        opt_state: OptState,
        batch: Batch,
    ):
        if batch.shape[0] % int(cfg.training.accum_steps) != 0:
            raise ValueError(
                "Gradient accumulation steps should divide the per-device batch size, "
                f"got {batch.shape[0]} !% {cfg.training.accum_steps}"
            )

        batch_local = tree_rearrange(
            batch,
            "(accum batch) ... -> accum batch ...",
            accum=max(1, int(cfg.training.accum_steps)),
        )
        outer_spec = get_filter_spec(meta_model, cfg.training.spec_outer, "outer parameters")
        trainable, frozen = eqx.partition(meta_model, outer_spec)

        def loss_fn(trainable_model, microbatch):
            full_model = eqx.combine(trainable_model, frozen)
            return vmap_mean(lambda seq: full_model.loss_for_sequence(seq, state), microbatch, axis_name="batch")

        grad_fn = lambda microbatch: eqx.filter_value_and_grad(loss_fn, has_aux=True)(trainable, microbatch)
        (loss, metrics), grads = welfords_online_mean(grad_fn, batch_local)
        avg_loss, avg_metrics, avg_grads = jax.lax.pmean((loss, metrics, grads), axis_name="data_parallel")
        avg_outer_grad_norm = global_norm_safe(avg_grads)
        avg_metrics[M.outer_grad_norm] = avg_outer_grad_norm
        updates, next_opt_state = optimizer.update(avg_grads, opt_state, trainable)
        next_model = eqx.combine(filter_apply_updates(trainable, updates), frozen)
        return next_model, next_opt_state, avg_loss, avg_metrics

    return train_on_sequence


@eqx.filter_jit
@eqx.filter_vmap(axis_name="data_parallel", in_axes=(None, 0, None), out_axes=None)
def eval_step_fn(meta_model: MetaModel, batch: Batch, state: nn.State):
    loss, metrics = vmap_mean(lambda seq: meta_model.loss_for_sequence(seq, state), batch, axis_name="batch")
    _avg_loss, avg_metrics = jax.lax.pmean((loss, metrics), axis_name="data_parallel")
    return avg_metrics


@dataclass
class EvalSummary:
    loss_scalar: float
    loss_ce: float
    token_nll_curve: np.ndarray
    metric_arrays: dict[str, np.ndarray]
    batches: int
    tokens: int
    elapsed_seconds: float


class Evaluator:
    def __init__(self, *, cfg: Config, data_sharding, n_data_parallel: int):
        self.cfg = cfg
        self.data_sharding = data_sharding
        self.n_data_parallel = n_data_parallel
        self.global_batch_size = int(cfg.training.eval_batch_size)
        self.dataset = (
            dummy_dataset(
                seq_len=cfg.training.seq_length,
                global_batch_size=self.global_batch_size,
                bos_token_id=cfg.model.bos_token_id,
                eos_token_id=cfg.model.eos_token_id,
                repeat=False,
            )
            if cfg.training.dummy_dataset
            else lm_dataset(
                path=cfg.training.dataset_path,
                split=cfg.training.eval_split,
                seq_len=cfg.training.seq_length,
                global_batch_size=self.global_batch_size,
                bos_token_id=cfg.model.bos_token_id,
                eos_token_id=cfg.model.eos_token_id,
                seed=cfg.training.data_seed,
                repeat=False,
                shuffle=False,
            )
        )

    def iter_batches(self):
        return iter(
            self.dataset.to_iter_dataset(
                grain.ReadOptions(
                    num_threads=max(1, int(self.cfg.training.loader_workers)),
                    prefetch_buffer_size=32,
                )
            )
        )

    def evaluate(self, *, model: MetaModel, state: nn.State, artifacts_dir: Path):
        max_batches = max(1, int(self.cfg.training.jax_eval_batches))
        metric_history: list[dict[MetaModel.MetricType, np.ndarray]] = []
        started = jax.device_get(jnp.asarray(0.0))
        import time

        started = time.time()
        tokens = 0
        for idx, batch in enumerate(self.iter_batches()):
            if idx >= max_batches:
                break
            batch = _to_sharded_batch(
                batch,
                data_sharding=self.data_sharding,
                global_batch_size=self.global_batch_size,
                n_data_parallel=self.n_data_parallel,
            )
            metrics = eval_step_fn(model, batch, state)
            metric_history.append(jax.tree.map(np.asarray, jax.device_get(metrics)))
            tokens += int(self.cfg.training.eval_batch_size) * int(self.cfg.training.seq_length)

        if not metric_history:
            raise ValueError("jax_eval produced no batches")

        aggregated = jax.tree.map(lambda *parts: np.mean(np.stack(parts, axis=0), axis=0), *metric_history)
        token_nll_curve = aggregated[M.token_nll_loss]
        np.save(artifacts_dir / "per_position_nll.npy", token_nll_curve)
        np.save(artifacts_dir / "loss_curve.npy", aggregated[M.loss])

        metric_arrays = {
            "loss_curve.npy": aggregated[M.loss],
            "token_nll_curve.npy": token_nll_curve,
        }
        np.save(artifacts_dir / "token_nll_curve.npy", token_nll_curve)
        elapsed = max(float(time.time() - started), 1e-9)
        return EvalSummary(
            loss_scalar=float(np.mean(aggregated[M.loss])),
            loss_ce=float(np.mean(aggregated[M.loss])),
            token_nll_curve=token_nll_curve,
            metric_arrays=metric_arrays,
            batches=len(metric_history),
            tokens=tokens,
            elapsed_seconds=elapsed,
        )
