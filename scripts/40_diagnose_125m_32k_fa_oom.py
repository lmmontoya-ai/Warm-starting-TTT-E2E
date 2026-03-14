#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import equinox as eqx
import grain.python as grain
import jax
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ttt.config import Config, register_configs
from ttt.dataloader import dummy_dataset, lm_dataset
from ttt.jax_runtime.checkpoint import resolve_restore_payload, unify_dict_with_eqx_module
from ttt.jax_runtime.loop import make_train_step
from ttt.jax_runtime.model.transformer import MetaModel
from ttt.jax_runtime.optimizers import make_optimizer
from ttt.jax_runtime.sharding import ModelSharding, local_device_summary, put_replicated, to_data_parallel_batch
from ttt.utils.jax_utils import initialize_distibuted, set_random_seed


def _utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _stringify_compiler_ir(ir: Any) -> str:
    for attr in ("as_hlo_text", "as_text", "dump"):
        candidate = getattr(ir, attr, None)
        if callable(candidate):
            try:
                value = candidate()
            except TypeError:
                continue
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value)
    return str(ir)


def _keypath(path) -> str:
    return jax.tree_util.keystr(path)


def _leaf_local_nbytes(value: Any) -> int | None:
    if not eqx.is_array(value):
        return None
    if hasattr(value, "addressable_shards") and value.addressable_shards:
        total = 0
        for shard in value.addressable_shards:
            shard_value = shard.data
            total += int(np.prod(getattr(shard_value, "shape", (0,)))) * np.dtype(shard_value.dtype).itemsize
        return total
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        return None
    return int(np.prod(shape)) * np.dtype(dtype).itemsize


def _tree_summary(tree: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_local_nbytes = 0
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        row: dict[str, Any] = {"path": _keypath(path)}
        if eqx.is_array(value):
            row["shape"] = list(value.shape)
            row["dtype"] = str(value.dtype)
            row["sharding"] = str(getattr(value, "sharding", ""))
            local_nbytes = _leaf_local_nbytes(value)
            row["local_nbytes"] = local_nbytes
            if local_nbytes is not None:
                total_local_nbytes += local_nbytes
        else:
            row["type"] = type(value).__name__
        rows.append(row)
    return {"total_local_nbytes": total_local_nbytes, "rows": rows}


def _block_tree(tree):
    return jax.tree.map(
        lambda x: jax.block_until_ready(x) if eqx.is_array(x) else x,
        tree,
    )


def _make_train_dataset(cfg: Config):
    if cfg.training.dummy_dataset:
        return dummy_dataset(
            seq_len=cfg.training.seq_length,
            global_batch_size=cfg.training.global_batch_size,
            bos_token_id=cfg.model.bos_token_id,
            eos_token_id=cfg.model.eos_token_id,
            repeat=True,
        )
    return lm_dataset(
        path=cfg.training.dataset_path,
        split=cfg.training.data_split,
        seq_len=cfg.training.seq_length,
        global_batch_size=cfg.training.global_batch_size,
        bos_token_id=cfg.model.bos_token_id,
        eos_token_id=cfg.model.eos_token_id,
        seed=cfg.training.data_seed,
        repeat=True,
        shuffle=False,
    )


def _restore_model(model: MetaModel, restored_weights):
    dynamic = model.weights()
    restored_dynamic, _ = unify_dict_with_eqx_module(restored_weights, dynamic)
    _, static = eqx.partition(model, eqx.is_inexact_array)
    return eqx.combine(restored_dynamic, static)


def _compose_cfg(args: argparse.Namespace) -> Config:
    register_configs()
    configs_dir = Path(__file__).resolve().parents[1] / "configs"
    overrides = [
        f"+deploy={args.deploy}",
        f"+experiment={args.experiment}",
        "training.runtime_mode=jax_train",
        "training.log_wandb=false",
        "training.wandb_entity=none",
        "training.wandb_project=none",
        "training.wandb_key=env",
        f"training.exp_dir={args.exp_dir}",
        f"training.exp_folder={args.exp_folder}",
        f"training.exp_name={args.exp_name}",
        f"training.stage_id={args.stage_id}",
        f"training.run_id={args.run_id}",
        f"training.total_steps={args.total_steps}",
        f"training.save_milestone_freq={args.save_milestone_freq}",
        f"training.load_part={args.load_part}",
        f"training.data_split={args.data_split}",
        f"training.loader_workers={args.loader_workers}",
        f"training.data_seed={args.data_seed}",
        f"training.model_seed={args.model_seed}",
        f"training.checkpoint_path={args.checkpoint_root}",
        f"deploy_paths.checkpoint={args.checkpoint_root}",
        f"deploy_paths.data.books3={args.books_root}",
    ]
    if args.dclm_root:
        overrides.append(f"deploy_paths.data.dclm_filter_8k={args.dclm_root}")
    if args.resume_checkpoint_path:
        overrides.append(f"training.resume_checkpoint_path={args.resume_checkpoint_path}")
        overrides.append(f"training.resume_checkpoint_format={args.resume_checkpoint_format}")
    if args.resume_exp_name:
        overrides.append(f"training.resume_exp_name={args.resume_exp_name}")
    if args.resume_step is not None:
        overrides.append(f"training.resume_step={args.resume_step}")
    if args.global_batch_size is not None:
        overrides.append(f"training.global_batch_size={args.global_batch_size}")
    if args.seq_length is not None:
        overrides.append(f"training.seq_length={args.seq_length}")
    if args.accum_steps is not None:
        overrides.append(f"training.accum_steps={args.accum_steps}")
    if args.n_state_parallel is not None:
        overrides.append(f"training.n_state_parallel={args.n_state_parallel}")
    if args.n_data_parallel is not None:
        overrides.append(f"training.n_data_parallel={args.n_data_parallel}")
    if args.dummy_dataset:
        overrides.append("training.dummy_dataset=true")
    overrides.extend(args.override)

    with initialize_config_dir(version_base=None, config_dir=str(configs_dir)):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile-memory diagnosis harness for the local 125M 32K FA extension path. "
            "Builds the exact train_step graph and records config, sharding, bytes, IR, "
            "and compile/execute results under artifacts/oom_diagnosis."
        )
    )
    parser.add_argument("--deploy", default="interactive")
    parser.add_argument("--experiment", default="125m/extension/ext-125m-fa-32K")
    parser.add_argument("--artifact-root", type=Path, default=Path("./artifacts/oom_diagnosis"))
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--exp-dir", default="./experiments")
    parser.add_argument("--exp-folder", default="oom_diagnosis")
    parser.add_argument("--exp-name", default="ext-125m-fa-32K-local-diag")
    parser.add_argument("--stage-id", default="S0_125M_DIAG")
    parser.add_argument("--run-id", default="ext-125m-fa-32K-local-diag")
    parser.add_argument("--checkpoint-root", default="./checkpoints")
    parser.add_argument("--books-root", required=True)
    parser.add_argument("--dclm-root", default="")
    parser.add_argument("--resume-checkpoint-path", default="")
    parser.add_argument("--resume-checkpoint-format", default="orbax")
    parser.add_argument("--resume-exp-name", default="")
    parser.add_argument("--resume-step", type=int, default=None)
    parser.add_argument("--load-part", default="params")
    parser.add_argument("--total-steps", type=int, default=2)
    parser.add_argument("--save-milestone-freq", type=int, default=999)
    parser.add_argument("--data-split", default="train")
    parser.add_argument("--loader-workers", type=int, default=32)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--accum-steps", type=int, default=None)
    parser.add_argument("--n-state-parallel", type=int, default=None)
    parser.add_argument("--n-data-parallel", type=int, default=None)
    parser.add_argument("--dummy-dataset", action="store_true")
    parser.add_argument("--skip-execute", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--worker", action="store_true")
    return parser.parse_args()


def _parent_main(args: argparse.Namespace) -> int:
    artifact_dir = args.artifact_dir
    if artifact_dir is None:
        artifact_dir = args.artifact_root / f"local_125m_32k_fa_{_utc_slug()}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir = artifact_dir

    invocation = {
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "python": sys.executable,
    }
    _write_json(artifact_dir / "invocation.json", invocation)

    child_cmd = [sys.executable, __file__, "--worker"]
    for key, value in vars(args).items():
        if key == "worker":
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                child_cmd.append(flag)
        elif isinstance(value, list):
            for item in value:
                child_cmd.extend([flag, str(item)])
        elif value is not None:
            child_cmd.extend([flag, str(value)])

    compile_log = artifact_dir / "compile.log"
    with compile_log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            child_cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    _write_json(
        artifact_dir / "driver_result.json",
        {
            "returncode": int(proc.returncode),
            "compile_log": str(compile_log),
        },
    )
    return int(proc.returncode)


def _worker_main(args: argparse.Namespace) -> int:
    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

    cfg = _compose_cfg(args)
    (artifact_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    (artifact_dir / "unresolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=False))

    initialize_distibuted(cfg.backend)
    device_info = local_device_summary()
    key = set_random_seed(int(cfg.training.model_seed))
    optimizer, _optimizer_info = make_optimizer(cfg.training.optimizer_outer)
    model_sharding = ModelSharding(cfg)
    data_sharding = model_sharding.data_sharding()
    replicated_sharding = model_sharding.replicated_sharding()
    n_data_parallel = int(cfg.training.n_data_parallel)

    train_iter = iter(
        _make_train_dataset(cfg).to_iter_dataset(
            grain.ReadOptions(
                num_threads=max(1, int(cfg.training.loader_workers)),
                prefetch_buffer_size=32,
            )
        )
    )

    @eqx.filter_jit
    def create_sharded_model_and_state():
        model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=key)
        state = put_replicated(state, replicated_sharding)
        model = model_sharding.shard_params(model)
        return model, state

    @eqx.filter_jit
    def create_stepped_opt_state(model: MetaModel):
        trainable_params = model.trainable_parameters()
        opt_state = optimizer.init(trainable_params)
        _, opt_state = optimizer.update(
            trainable_params,
            opt_state,
            model.trainable_parameters(),
        )
        return opt_state

    @eqx.filter_jit
    def restore_model_weights(model: MetaModel, restored_weights):
        model = _restore_model(model, restored_weights)
        return model_sharding.shard_params(model)

    @eqx.filter_jit
    def restore_opt_state(model: MetaModel, restored_opt_state):
        opt_state = create_stepped_opt_state(model)
        opt_state, _ = unify_dict_with_eqx_module(restored_opt_state, opt_state)
        return opt_state

    with model_sharding.mesh:
        model, state = create_sharded_model_and_state()
        opt_state = optimizer.init(model.trainable_parameters())

        restore_payload = resolve_restore_payload(
            cfg=cfg,
            current_checkpoint_dir=Path(cfg.checkpoint.checkpoint_dir),
            targets={
                "model_weights": model.weights(),
                "opt_state": create_stepped_opt_state(model),
            },
        )
        restore_summary: dict[str, Any] = {"restored": restore_payload is not None}
        if restore_payload is not None:
            restore_summary["step"] = int(restore_payload.step)
            model = restore_model_weights(model, restore_payload.model_weights)
            if str(cfg.training.load_part) == "all" and restore_payload.opt_state is not None:
                opt_state = restore_opt_state(model, restore_payload.opt_state)

        raw_batch = next(train_iter)
        batch = to_data_parallel_batch(
            raw_batch,
            data_sharding=data_sharding,
            global_batch_size=int(cfg.training.global_batch_size),
            n_data_parallel=n_data_parallel,
        )
        batch = _block_tree(batch)
        train_step = make_train_step(cfg, optimizer)

        summary = {
            "device_info": device_info,
            "restore": restore_summary,
            "model_local_nbytes": _tree_summary(model.weights())["total_local_nbytes"],
            "opt_state_local_nbytes": _tree_summary(opt_state)["total_local_nbytes"],
            "state_local_nbytes": _tree_summary(state)["total_local_nbytes"],
            "batch_local_nbytes": _tree_summary(batch)["total_local_nbytes"],
        }
        _write_json(artifact_dir / "summary.json", summary)
        _write_json(artifact_dir / "model_tree.json", _tree_summary(model.weights()))
        _write_json(artifact_dir / "opt_state_tree.json", _tree_summary(opt_state))
        _write_json(artifact_dir / "state_tree.json", _tree_summary(state))
        _write_json(artifact_dir / "batch_tree.json", _tree_summary(batch))

        lowered = train_step.lower(state, model, opt_state, batch)
        (artifact_dir / "lowered_as_text.txt").write_text(lowered.as_text())
        inner_lowered = lowered.lowered
        try:
            compiler_ir = inner_lowered.compiler_ir(dialect="hlo")
        except TypeError:
            compiler_ir = inner_lowered.compiler_ir()
        except Exception as exc:  # pragma: no cover - environment specific
            compiler_ir = exc
        if not isinstance(compiler_ir, Exception):
            (artifact_dir / "compiler_ir.txt").write_text(_stringify_compiler_ir(compiler_ir))

        try:
            cost_analysis = inner_lowered.cost_analysis()
        except Exception as exc:  # pragma: no cover - backend specific
            cost_analysis = {"error": repr(exc)}
        _write_json(artifact_dir / "cost_analysis.json", {"cost_analysis": cost_analysis})

        compile_started = time.perf_counter()
        try:
            compiled = lowered.compile()
            compile_seconds = float(time.perf_counter() - compile_started)
            compile_result = {"status": "ok", "compile_seconds": compile_seconds}
            if hasattr(compiled, "memory_analysis"):
                try:
                    memory_analysis = compiled.memory_analysis()
                    if hasattr(memory_analysis, "_asdict"):
                        compile_result["memory_analysis"] = memory_analysis._asdict()
                    else:
                        compile_result["memory_analysis"] = str(memory_analysis)
                except Exception as exc:  # pragma: no cover - backend specific
                    compile_result["memory_analysis_error"] = repr(exc)
            _write_json(artifact_dir / "compile_result.json", compile_result)
        except Exception as exc:
            _write_json(
                artifact_dir / "compile_result.json",
                {
                    "status": "error",
                    "phase": "compile",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "compile_seconds": float(time.perf_counter() - compile_started),
                },
            )
            return 1

        if args.skip_execute:
            return 0

        execute_started = time.perf_counter()
        try:
            model, opt_state, loss_value, metrics = compiled(state, model, opt_state, batch)
            loss_value = float(jax.device_get(loss_value))
            metrics_host = {str(k): _json_safe(jax.device_get(v)) for k, v in metrics.items()}
            _write_json(
                artifact_dir / "execute_result.json",
                {
                    "status": "ok",
                    "execute_seconds": float(time.perf_counter() - execute_started),
                    "loss": loss_value,
                    "metric_keys": sorted(str(k) for k in metrics_host.keys()),
                },
            )
            _write_json(artifact_dir / "metrics_snapshot.json", {"metrics": metrics_host})
            return 0
        except Exception as exc:
            _write_json(
                artifact_dir / "execute_result.json",
                {
                    "status": "error",
                    "phase": "execute",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "execute_seconds": float(time.perf_counter() - execute_started),
                },
            )
            return 2


def main() -> int:
    args = parse_args()
    if args.worker:
        return _worker_main(args)
    return _parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
