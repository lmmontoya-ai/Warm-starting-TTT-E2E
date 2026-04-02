#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from ttt.config import Config, register_configs
from ttt.jax_runtime.checkpoint import resolve_restore_payload, unify_dict_with_eqx_module
from ttt.jax_runtime.model.data import Batch
from ttt.jax_runtime.model.transformer import BlockCollectionSplit, MetaModel
from ttt.jax_runtime.sharding import ModelSharding, put_replicated
from ttt.research.tracking import write_eval_manifest
from ttt.research.types import EvalResult, utc_now_iso
from ttt.utils.filter_utils import get_filter_spec
from ttt.utils.jax_utils import clone_pytree, initialize_distibuted, scan_remat_chunk, set_random_seed, tree_rearrange


register_configs()

M = MetaModel.MetricType


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    return [int(part) for part in _parse_csv(raw)]


def _parse_float_csv(raw: str) -> list[float]:
    return [float(part) for part in _parse_csv(raw)]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: set[str] = set()
    for row in rows:
        fields.update(row.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _sanitize_optimizer_cfg(raw_cfg: Any) -> None:
    if not OmegaConf.is_config(raw_cfg):
        return
    training = raw_cfg.get("training")
    if training is None:
        return
    for key in ("optimizer_inner", "optimizer_outer"):
        opt = training.get(key)
        if opt is None:
            continue
        opt_type = str(opt.get("optimizer_type", "sgd"))
        defaults: dict[str, Any] = {
            "init_lr": 0.0,
            "end_lr": 0.0,
            "lr": 0.0,
            "lr_warmup_steps": 0,
            "lr_decay_steps": 0,
            "b1": 0.0,
            "b2": 0.0,
            "clip_gradient": 0.0,
            "weight_decay": 0.0,
            "bf16_momentum": False,
        }
        if opt_type == "adamw":
            defaults.update(
                {
                    "b1": 0.9,
                    "b2": 0.95,
                    "clip_gradient": 1.0,
                    "weight_decay": 0.1,
                    "bf16_momentum": False,
                    "init_lr": 0.0,
                    "end_lr": 1e-5,
                }
            )
        for field, default_value in defaults.items():
            value = opt.get(field, None)
            if value in (None, "???", "MISSING"):
                opt[field] = default_value


def _discover_runs(exp_dir: Path, paper_run_id: str, stage_filter: set[str], run_filter: set[str]) -> list[dict[str, Any]]:
    root = exp_dir / paper_run_id
    if not root.exists():
        raise FileNotFoundError(f"Missing paper run root: {root}")

    refs: list[dict[str, Any]] = []
    for manifest_path in sorted(root.rglob("run_manifest.json")):
        run_dir = manifest_path.parent
        run_manifest = _load_json(manifest_path)
        stage_id = str(run_manifest.get("stage_id", "")).strip()
        run_id = str(run_manifest.get("run_id", run_dir.name)).strip() or run_dir.name
        exp_name = str(run_manifest.get("exp_name", run_id)).strip() or run_id
        if stage_filter and stage_id not in stage_filter:
            continue
        if run_filter and run_id not in run_filter and exp_name not in run_filter:
            continue
        refs.append(
            {
                "stage_id": stage_id,
                "run_id": run_id,
                "exp_name": exp_name,
                "run_dir": run_dir,
            }
        )
    return refs


def _load_cfg(path: Path) -> Config:
    raw_cfg = OmegaConf.load(path)
    _sanitize_optimizer_cfg(raw_cfg)
    merged = OmegaConf.merge(OmegaConf.structured(Config), raw_cfg)
    cfg_obj = OmegaConf.to_object(merged)
    if not isinstance(cfg_obj, Config):
        raise TypeError(f"Expected Config object from {path}")
    return cfg_obj


def _restore_model(model: MetaModel, restored_weights):
    dynamic = model.weights()
    restored_dynamic, _, _ = unify_dict_with_eqx_module(restored_weights, dynamic)
    _, static = eqx.partition(model, eqx.is_inexact_array)
    return eqx.combine(restored_dynamic, static)


def _to_local_batch(tokens: np.ndarray, *, bos_token_id: int) -> Batch:
    input_ids = np.asarray(tokens[..., :-1], dtype=np.int32)
    target_tokens = np.asarray(tokens[..., 1:], dtype=np.int32)
    loss_masks = np.ones_like(target_tokens, dtype=np.int32)
    loss_masks[target_tokens == bos_token_id] = 0
    position_ids = np.broadcast_to(np.arange(input_ids.shape[-1], dtype=np.int32), input_ids.shape)
    return Batch(
        input_ids=input_ids,
        target_tokens=target_tokens,
        loss_masks=loss_masks,
        attention_mask=None,
        position_ids=position_ids,
    )


def _to_sharded_batch(batch: Batch, *, data_sharding, global_batch_size: int, n_data_parallel: int) -> Batch:
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


def _random_token_excluding(rng: random.Random, vocab_size: int, excluded: set[int]) -> int:
    while True:
        token = rng.randrange(vocab_size)
        if token not in excluded:
            return token


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    accs = [float(row["niah_accuracy"]) for row in rows if row.get("status") == "ok"]
    if not accs:
        return {}
    metrics: dict[str, float] = {
        "niah_accuracy_mean": float(mean(accs)),
    }
    by_length: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        ctx = int(row["context_length"])
        by_length[ctx].append(float(row["niah_accuracy"]))
    for ctx, values in sorted(by_length.items()):
        metrics[f"niah_by_length_{ctx}"] = float(mean(values))
    return metrics


def _meta_final_logits(meta_model: MetaModel, seq: Batch, state) -> jnp.ndarray:
    cfg = meta_model.config
    if str(cfg.training.train_mode) != "meta":
        raise NotImplementedError(
            f"JAX NIAH appendix eval currently supports only meta checkpoints, got train_mode={cfg.training.train_mode}"
        )

    block_collection = meta_model.language_model.model.h.blocks
    prime_storage = meta_model.language_model.model.h.prime_storage if cfg.model.prime else None
    model = eqx.tree_at(
        lambda m: m.language_model.model.h,
        meta_model,
        BlockCollectionSplit(cfg.model, block_collection, prime_storage, key=jax.random.PRNGKey(0)),
    )

    state_prefix_suffix = state.substate(block_collection)
    state_prefix, state_suffix = BlockCollectionSplit.split_state(state_prefix_suffix, cfg.model.suffix_len)
    state_all = clone_pytree(state)
    tokens_per_chunk = int(cfg.model.mini_batch_size)
    if int(cfg.training.seq_length) % tokens_per_chunk != 0:
        raise ValueError(
            f"seq_length {cfg.training.seq_length} must be divisible by mini_batch_size {tokens_per_chunk}"
        )

    model = jax.tree.map(lambda p: p.astype(meta_model.state_dtype), model)
    inner_opt_state = model.inner_optimizer(state_all).init(model.inner_parameters())
    xt_embed = model.language_model.wte_call(seq.input_ids)
    prefix_output = model.language_model.prefix_call(
        model.language_model.model.h.prefix_blocks,
        xt_embed,
        state_prefix,
        seq,
    ).last_hidden_state

    seq_chunks = tree_rearrange(seq, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)
    prefix_chunks = tree_rearrange(prefix_output, "(chunk token) ... -> chunk token ...", token=tokens_per_chunk)

    def process_suffix_chunk(carry, inputs):
        model_inner, opt_state_inner, state_tuple = carry
        suffix_chunk, prefix_chunk = inputs
        spec_inner = get_filter_spec(model_inner, cfg.training.spec_inner, "inner parameters")
        inner_params, _ = eqx.partition(model_inner, spec_inner)
        _, outer_params = eqx.partition(model, spec_inner)
        model_inner = eqx.combine(inner_params, outer_params)
        new_model, new_opt_state, new_state_tuple, _metrics = MetaModel.inner_loop_step(
            model_inner,
            opt_state_inner,
            state_tuple,
            suffix_chunk,
            prefix_chunk,
        )
        return (new_model, new_opt_state, new_state_tuple), None

    carry = (model, inner_opt_state, (state_all, state_suffix))
    if seq_chunks.input_ids.shape[0] > 1:
        prefix_inputs = jax.tree.map(lambda x: x[:-1], seq_chunks)
        carry, _ = scan_remat_chunk(
            eqx.filter_checkpoint(process_suffix_chunk, prevent_cse=False),
            carry,
            (prefix_inputs, prefix_chunks[:-1]),
            remat_n_loops=cfg.training.inner_remat_freq,
            unroll=cfg.model.unroll_inner_scan,
        )

    model_final, _, (_state_all_final, state_suffix_final) = carry
    spec_inner = get_filter_spec(model_final, cfg.training.spec_inner, "inner parameters")
    inner_params, _ = eqx.partition(model_final, spec_inner)
    _, outer_params = eqx.partition(model, spec_inner)
    model_final = eqx.combine(inner_params, outer_params)

    final_seq = jax.tree.map(lambda x: x[-1], seq_chunks)
    final_prefix = prefix_chunks[-1]
    outputs = model_final.language_model.suffix_call(final_prefix, state_suffix_final, final_seq)
    return outputs.logits[-1]


def _run_one_niah_eval(*, run_ref: dict[str, Any], args: argparse.Namespace, checkpoint_root: Path, repo_root: Path) -> EvalResult:
    cfg = _load_cfg(run_ref["run_dir"] / "resolved_config.yaml")
    cfg.training.runtime_mode = cfg.training.RuntimeMode.jax_eval
    cfg.training.load_part = cfg.training.LoadPart.params
    cfg.training.resume_checkpoint_format = "orbax"
    checkpoint_path = (checkpoint_root / args.exp_folder / run_ref["exp_name"]).resolve()
    if args.checkpoint_step is not None:
        checkpoint_path = checkpoint_path / str(int(args.checkpoint_step))
    cfg.training.resume_checkpoint_path = str(checkpoint_path)
    cfg.training.eval_batch_size = int(args.batch_size)

    score_batch_size = int(args.batch_size)
    if score_batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    eval_dir = run_ref["run_dir"] / args.eval_subdir
    eval_dir.mkdir(parents=True, exist_ok=True)
    raw_json = eval_dir / "eval_raw.json"
    raw_csv = eval_dir / "eval_raw.csv"
    manifest_path = eval_dir / "eval_manifest.json"
    created = utc_now_iso()

    initialize_distibuted(cfg.backend)
    cfg.model.seq_len = max(_parse_int_csv(args.contexts))
    key = set_random_seed(int(cfg.training.model_seed))
    model_sharding = ModelSharding(cfg)
    data_sharding = model_sharding.data_sharding()
    replicated_sharding = model_sharding.replicated_sharding()
    n_data_parallel = int(cfg.training.n_data_parallel)
    if score_batch_size % n_data_parallel != 0:
        raise ValueError(
            f"--batch-size ({score_batch_size}) must be divisible by n_data_parallel ({n_data_parallel})"
        )

    def create_host_model_and_state():
        return eqx.nn.make_with_state(MetaModel)(cfg, key=key)

    @eqx.filter_jit
    def create_sharded_model_and_state():
        model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=key)
        state = put_replicated(state, replicated_sharding)
        model = model_sharding.shard_params(model)
        return model, state

    @eqx.filter_jit
    def shard_restored_model(model: MetaModel):
        return model_sharding.shard_params(model)

    @eqx.filter_jit
    @eqx.filter_vmap(in_axes=(None, 0, None), out_axes=0)
    def score_final_token_logits(meta_model: MetaModel, batch: Batch, state):
        return jax.vmap(lambda seq: _meta_final_logits(meta_model, seq, state))(batch)

    started = time.time()
    rows: list[dict[str, Any]] = []

    host_model, host_state = create_host_model_and_state()
    restore_payload = None
    restore_error: Exception | None = None
    try:
        with model_sharding.mesh:
            model, state = create_sharded_model_and_state()
            restore_payload = resolve_restore_payload(
                cfg=cfg,
                current_checkpoint_dir=checkpoint_path if checkpoint_path.name.isdigit() else checkpoint_path.parent,
                targets={"model_weights": model.weights()},
            )
            if restore_payload is None:
                raise FileNotFoundError(f"Missing restore payload for {run_ref['exp_name']} at {checkpoint_path}")
            model = _restore_model(model, restore_payload.model_weights)
            model = shard_restored_model(model)
            state = put_replicated(
                state.set(model.step_index, jax.numpy.array(restore_payload.step, dtype=jax.numpy.int32)),
                replicated_sharding,
            )
    except Exception as exc:
        restore_error = exc
        restore_payload = resolve_restore_payload(
            cfg=cfg,
            current_checkpoint_dir=checkpoint_path if checkpoint_path.name.isdigit() else checkpoint_path.parent,
            targets={"model_weights": host_model.weights()},
        )
        if restore_payload is None:
            raise FileNotFoundError(f"Missing restore payload for {run_ref['exp_name']} at {checkpoint_path}") from exc
        host_model = _restore_model(host_model, restore_payload.model_weights)

    with model_sharding.mesh:
        if restore_error is not None:
            model = shard_restored_model(host_model)
            state = put_replicated(host_state, replicated_sharding)
            state = put_replicated(
                state.set(model.step_index, jax.numpy.array(restore_payload.step, dtype=jax.numpy.int32)),
                replicated_sharding,
            )

        for context_length in _parse_int_csv(args.contexts):
            if context_length <= 0:
                raise ValueError("Context lengths must be positive")
            for pos in _parse_float_csv(args.positions):
                pos_idx = int(round((context_length - 1) * pos))
                pos_idx = min(max(pos_idx, 0), context_length - 1)
                rng = random.Random(int(args.seed) + context_length * 1000 + int(pos * 1000))

                example_meta: list[dict[str, Any]] = []
                token_windows: list[list[int]] = []

                for example_idx in range(int(args.examples)):
                    needle = rng.randrange(int(cfg.model.vocab_size))
                    context = [
                        _random_token_excluding(rng, int(cfg.model.vocab_size), {needle})
                        for _ in range(context_length)
                    ]
                    context[pos_idx] = needle
                    choice_set = [needle]
                    seen = {needle}
                    while len(choice_set) < int(args.candidates):
                        cand = _random_token_excluding(rng, int(cfg.model.vocab_size), seen)
                        choice_set.append(cand)
                        seen.add(cand)
                    rng.shuffle(choice_set)
                    placeholder = _random_token_excluding(rng, int(cfg.model.vocab_size), {needle})
                    token_windows.append([*context, int(placeholder)])
                    example_meta.append(
                        {
                            "example_idx": example_idx,
                            "needle": int(needle),
                            "candidates": [int(x) for x in choice_set],
                        }
                    )

                pad_token = token_windows[-1][-1]
                correct = 0
                for start_idx in range(0, len(token_windows), score_batch_size):
                    chunk_tokens = token_windows[start_idx : start_idx + score_batch_size]
                    chunk_meta = example_meta[start_idx : start_idx + score_batch_size]
                    real_count = len(chunk_tokens)
                    if real_count < score_batch_size:
                        chunk_tokens = list(chunk_tokens)
                        chunk_meta = list(chunk_meta)
                        while len(chunk_tokens) < score_batch_size:
                            chunk_tokens.append([*chunk_tokens[-1][:-1], pad_token])
                            chunk_meta.append({"example_idx": -1, "needle": -1, "candidates": [int(pad_token)]})

                    local_batch = _to_local_batch(np.asarray(chunk_tokens, dtype=np.int32), bos_token_id=int(cfg.model.bos_token_id))
                    sharded_batch = _to_sharded_batch(
                        local_batch,
                        data_sharding=data_sharding,
                        global_batch_size=score_batch_size,
                        n_data_parallel=n_data_parallel,
                    )
                    logits = np.asarray(jax.device_get(score_final_token_logits(model, sharded_batch, state)), dtype=np.float32)
                    logits = logits.reshape(score_batch_size, logits.shape[-1])
                    for idx in range(real_count):
                        meta = chunk_meta[idx]
                        choice_set = [int(x) for x in meta["candidates"]]
                        candidate_logits = np.asarray([logits[idx, cand] for cand in choice_set], dtype=np.float32)
                        pred = choice_set[int(np.argmax(candidate_logits))]
                        if pred == int(meta["needle"]):
                            correct += 1

                rows.append(
                    {
                        "record_type": "niah_proxy",
                        "status": "ok",
                        "paper_run_id": args.paper_run_id,
                        "stage_id": run_ref["stage_id"],
                        "run_id": run_ref["run_id"],
                        "exp_name": run_ref["exp_name"],
                        "eval_id": args.eval_id,
                        "checkpoint_step": int(restore_payload.step),
                        "context_length": int(context_length),
                        "position_fraction": float(pos),
                        "position_index": int(pos_idx),
                        "examples": int(args.examples),
                        "candidates": int(args.candidates),
                        "niah_accuracy": float(correct / max(1, int(args.examples))),
                    }
                )

    metrics = _summarize_rows(rows)
    finished = utc_now_iso()
    _write_json(
        raw_json,
        {
            "schema_version": "1.0",
            "paper_run_id": args.paper_run_id,
            "stage_id": run_ref["stage_id"],
            "run_id": run_ref["run_id"],
            "exp_name": run_ref["exp_name"],
            "eval_id": args.eval_id,
            "created_at_utc": created,
            "finished_at_utc": finished,
            "metrics": metrics,
            "rows": rows,
            "elapsed_seconds": float(time.time() - started),
        },
    )
    _write_csv(raw_csv, rows)

    result = EvalResult(
        run_id=run_ref["run_id"],
        stage_id=run_ref["stage_id"],
        eval_id=args.eval_id,
        status="succeeded",
        created_at_utc=created,
        finished_at_utc=finished,
        eval_manifest_path=str(manifest_path),
        raw_json_path=str(raw_json),
        raw_csv_path=str(raw_csv),
        metrics=metrics,
        artifacts={
            "eval_dir": str(eval_dir),
            "elapsed_seconds": str(float(time.time() - started)),
        },
    )
    write_eval_manifest(manifest_path, result, repo_root=repo_root)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JAX-native NIAH proxy eval on parity checkpoints.")
    parser.add_argument("--paper-run-id", required=True)
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--exp-folder", required=True)
    parser.add_argument("--stages", default="")
    parser.add_argument("--runs", default="")
    parser.add_argument("--contexts", default="8192,32768")
    parser.add_argument("--positions", default="0.1,0.5,0.9")
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--eval-id", default="jax_niah_proxy")
    parser.add_argument("--eval-subdir", default="niah_jax")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    stage_filter = set(_parse_csv(args.stages))
    run_filter = set(_parse_csv(args.runs))

    run_refs = _discover_runs(exp_dir, args.paper_run_id, stage_filter, run_filter)
    if not run_refs:
        raise FileNotFoundError("No runs found for JAX NIAH eval.")

    if args.summary_json is None:
        summary_json = (
            Path("./reports/paper") / args.paper_run_id / "eval" / "niah_jax_summary.json"
        ).resolve()
    else:
        summary_json = args.summary_json.expanduser().resolve()

    if args.summary_csv is None:
        summary_csv = (
            Path("./reports/paper") / args.paper_run_id / "eval" / "niah_jax_summary.csv"
        ).resolve()
    else:
        summary_csv = args.summary_csv.expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[1]
    rows: list[dict[str, Any]] = []
    failed = 0

    for run_ref in run_refs:
        try:
            result = _run_one_niah_eval(
                run_ref=run_ref,
                args=args,
                checkpoint_root=checkpoint_root,
                repo_root=repo_root,
            )
            row = {
                "paper_run_id": args.paper_run_id,
                "stage_id": run_ref["stage_id"],
                "run_id": run_ref["run_id"],
                "exp_name": run_ref["exp_name"],
                "status": result.status,
                "eval_id": args.eval_id,
            }
            row.update(result.metrics)
            rows.append(row)
        except Exception as exc:
            failed += 1
            rows.append(
                {
                    "paper_run_id": args.paper_run_id,
                    "stage_id": run_ref["stage_id"],
                    "run_id": run_ref["run_id"],
                    "exp_name": run_ref["exp_name"],
                    "status": "failed",
                    "eval_id": args.eval_id,
                    "error_message": str(exc),
                }
            )

    _write_json(
        summary_json,
        {
            "schema_version": "1.0",
            "paper_run_id": args.paper_run_id,
            "eval_id": args.eval_id,
            "n_runs": len(rows),
            "n_failed": failed,
            "rows": rows,
        },
    )
    _write_csv(summary_csv, rows)
    print(f"Wrote NIAH summary JSON: {summary_json}")
    print(f"Wrote NIAH summary CSV:  {summary_csv}")
    if failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
