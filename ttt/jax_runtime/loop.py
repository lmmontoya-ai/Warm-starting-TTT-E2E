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
from ttt.utils.filter_utils import filter_apply_updates
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


def _aggregate_metric_history(
    metric_history: list[dict[MetaModel.MetricType, np.ndarray]],
) -> dict[MetaModel.MetricType, np.ndarray]:
    # Eval metrics flow through BF16-heavy paths. Cast to float32 before the
    # final reduction so paper-facing scalars are not snapped to coarse bins.
    return jax.tree.map(
        lambda *parts: np.asarray(np.stack(parts, axis=0), dtype=np.float32).mean(axis=0, dtype=np.float32),
        *metric_history,
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
        loss_fn = lambda model, microbatch: vmap_mean(
            lambda seq: MetaModel.loss_for_sequence(model, seq, state),
            microbatch,
            axis_name="batch",
        )

        grad_fn = lambda microbatch: eqx.filter_value_and_grad(loss_fn, has_aux=True)(meta_model, microbatch)
        (loss, metrics), grads = welfords_online_mean(grad_fn, batch_local)
        avg_loss, avg_metrics, avg_grads = jax.lax.pmean((loss, metrics, grads), axis_name="data_parallel")
        avg_grads = avg_grads.trainable_parameters()
        avg_outer_grad_norm = global_norm_safe(avg_grads)
        avg_metrics[M.outer_grad_norm] = avg_outer_grad_norm
        updates, next_opt_state = optimizer.update(avg_grads, opt_state, meta_model.trainable_parameters())
        next_model = filter_apply_updates(meta_model, updates)
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

        aggregated = _aggregate_metric_history(metric_history)
        token_nll_curve = np.asarray(aggregated[M.token_nll_loss], dtype=np.float32)
        loss_curve = np.asarray(aggregated[M.loss], dtype=np.float32)
        np.save(artifacts_dir / "per_position_nll.npy", token_nll_curve)
        np.save(artifacts_dir / "loss_curve.npy", loss_curve)

        metric_arrays = {
            "loss_curve.npy": loss_curve,
            "token_nll_curve.npy": token_nll_curve,
        }
        np.save(artifacts_dir / "token_nll_curve.npy", token_nll_curve)
        elapsed = max(float(time.time() - started), 1e-9)
        return EvalSummary(
            loss_scalar=float(np.mean(loss_curve, dtype=np.float32)),
            loss_ce=float(np.mean(loss_curve, dtype=np.float32)),
            token_nll_curve=token_nll_curve,
            metric_arrays=metric_arrays,
            batches=len(metric_history),
            tokens=tokens,
            elapsed_seconds=elapsed,
        )
