#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ttt.dataloader import build_batch_iterator
from ttt.dataloader.lm_dataset import _load_token_stream
from ttt.infra import Phase1Checkpointer
from ttt.model import TokenStatsModel, TokenStatsState


def _parse_int_csv(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def _parse_float_csv(raw: str) -> list[float]:
    out: list[float] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(float(p))
    return out


def _parse_str_csv(raw: str) -> list[str]:
    out: list[str] = []
    for part in raw.split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out


def _flatten_targets(batch) -> list[int]:
    tokens: list[int] = []
    for sample in batch.samples:
        tokens.extend(sample.target_tokens)
    return tokens


def _flops_per_token_proxy(model_cfg: dict[str, Any], seq_len: int) -> float:
    hidden = int(model_cfg.get("hidden_size", 0))
    layers = int(model_cfg.get("num_hidden_layers", 0))
    intermediate = int(model_cfg.get("intermediate_size", 0))
    pattern = str(model_cfg.get("attention_pattern", "full"))
    sliding_window = int(model_cfg.get("sliding_window_size", seq_len))
    suffix_len = max(int(model_cfg.get("suffix_len", 0)), 0)

    if hidden <= 0 or layers <= 0:
        return 0.0

    span = seq_len
    if pattern in {"swa", "ttt_swa"}:
        span = max(1, min(seq_len, sliding_window))

    proj_cost = (8.0 * hidden * hidden) + (6.0 * hidden * max(intermediate, 1))
    attn_cost = 4.0 * hidden * span
    ttt_cost = 0.0
    if pattern == "ttt_swa" and suffix_len > 0:
        ttt_cost = 2.0 * hidden * max(intermediate, 1) * (suffix_len / max(1.0, float(seq_len)))

    return float(layers) * (proj_cost + attn_cost + ttt_cost)


def _dataset_root_for_key(dataset_key: str, args: argparse.Namespace) -> Path:
    norm = dataset_key.strip().lower()
    if norm in {"books3", "books"}:
        return args.books_root
    if norm in {"dclm_filter_8k", "dclm", "dclm8k"}:
        return args.dclm_root
    raise ValueError(f"Unsupported dataset key for eval: {dataset_key}")


def _load_run_cfg(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "phase1_resolved_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing resolved config: {cfg_path}")
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config payload in {cfg_path}")
    return cfg


def _load_latest_state(
    checkpoint_root: Path,
    exp_folder: str,
    exp_name: str,
) -> tuple[int, TokenStatsState]:
    checkpointer = Phase1Checkpointer(checkpoint_root / exp_folder / exp_name)
    restored = checkpointer.load(step=None)
    if restored is None:
        raise FileNotFoundError(
            f"Missing checkpoint for {exp_folder}/{exp_name} under {checkpoint_root}"
        )
    raw_state = restored.payload.get("model_state")
    if not isinstance(raw_state, dict):
        raise ValueError(
            f"Checkpoint for {exp_folder}/{exp_name} is missing model_state payload."
        )
    return restored.step, TokenStatsState.from_jsonable(raw_state)


def _eval_loss_efficiency(
    *,
    model: TokenStatsModel,
    model_state: TokenStatsState,
    dataset_path: Path,
    split: str,
    seq_len: int,
    batch_size: int,
    max_batches: int,
    seed: int,
    flops_per_token_proxy: float,
) -> dict[str, Any]:
    if max_batches <= 0:
        raise ValueError("max_batches must be > 0")

    iterator = build_batch_iterator(
        dataset_path=str(dataset_path),
        split=split,
        seq_len=seq_len,
        global_batch_size=batch_size,
        repeat=False,
        shuffle=False,
        seed=seed,
        dummy_dataset=False,
        vocab_size=model.vocab_size,
    )

    batches = 0
    tokens_seen = 0
    weighted_loss = 0.0
    start = time.perf_counter()
    for batch in iterator:
        targets = _flatten_targets(batch)
        if not targets:
            continue
        loss = model.nll(model_state, targets)
        weighted_loss += loss * len(targets)
        tokens_seen += len(targets)
        batches += 1
        if batches >= max_batches:
            break
    elapsed = max(time.perf_counter() - start, 1e-9)

    if tokens_seen <= 0:
        raise ValueError(
            f"No evaluation tokens produced for dataset={dataset_path} split={split} seq_len={seq_len}."
        )

    mean_loss = weighted_loss / float(tokens_seen)
    ppl = math.exp(min(mean_loss, 20.0))
    tps = tokens_seen / elapsed
    return {
        "loss": mean_loss,
        "perplexity": ppl,
        "tokens_evaluated": tokens_seen,
        "eval_batches": batches,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tps,
        "latency_ms_per_token": (1000.0 / tps) if tps > 0 else None,
        "flops_per_token_proxy": flops_per_token_proxy,
        "flops_per_second_proxy": flops_per_token_proxy * tps,
    }


def _sample_starts(total_tokens: int, window: int, n: int, seed: int) -> list[int]:
    if n <= 0:
        return []
    max_start = total_tokens - window
    if max_start < 0:
        return []
    if n >= max_start + 1:
        return list(range(max_start + 1))
    rng = random.Random(seed)
    starts: set[int] = set()
    while len(starts) < n:
        starts.add(rng.randint(0, max_start))
    return sorted(starts)


def _eval_decode_trend_proxy(
    *,
    model: TokenStatsModel,
    model_state: TokenStatsState,
    dataset_path: Path,
    split: str,
    context_len: int,
    decode_steps: int,
    prompts: int,
    seed: int,
) -> dict[str, Any]:
    if decode_steps <= 0:
        raise ValueError("decode_steps must be > 0")
    if prompts <= 0:
        raise ValueError("prompts must be > 0")

    tokens = _load_token_stream(str(dataset_path), split=split)
    window = context_len + decode_steps
    starts = _sample_starts(total_tokens=len(tokens), window=window, n=prompts, seed=seed)
    if not starts:
        raise ValueError(
            f"Dataset too short for decode proxy at context={context_len}, decode_steps={decode_steps}."
        )

    per_step_sum = [0.0 for _ in range(decode_steps)]
    start = time.perf_counter()
    used = 0
    for s in starts:
        prompt = tokens[s : s + context_len]
        continuation = tokens[s + context_len : s + window]
        if len(prompt) < context_len or len(continuation) < decode_steps:
            continue

        state = model.clone_state(model_state)
        model.update(state, prompt, weight=1.0)
        for idx, token in enumerate(continuation):
            prob = model._prob(state, token)
            per_step_sum[idx] += -math.log(max(prob, 1e-12))
            model.update(state, [token], weight=1.0)
        used += 1

    elapsed = max(time.perf_counter() - start, 1e-9)
    if used <= 0:
        raise ValueError("No decode prompts could be evaluated.")

    per_step_mean = [x / float(used) for x in per_step_sum]
    q = max(decode_steps // 4, 1)
    first_q = sum(per_step_mean[:q]) / float(q)
    last_q = sum(per_step_mean[-q:]) / float(q)
    overall = sum(per_step_mean) / float(len(per_step_mean))
    tokens_processed = used * (context_len + decode_steps)
    tps = tokens_processed / elapsed

    return {
        "prompts_used": used,
        "decode_steps": decode_steps,
        "mean_decode_nll": overall,
        "decode_nll_first_quarter": first_q,
        "decode_nll_last_quarter": last_q,
        "decode_nll_slope": last_q - first_q,
        "tokens_processed": tokens_processed,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tps,
    }


def _random_token_excluding(rng: random.Random, vocab_size: int, excluded: set[int]) -> int:
    while True:
        token = rng.randrange(vocab_size)
        if token not in excluded:
            return token


def _eval_niah_proxy(
    *,
    model: TokenStatsModel,
    model_state: TokenStatsState,
    vocab_size: int,
    context_len: int,
    examples: int,
    candidates: int,
    positions: list[float],
    seed: int,
) -> list[dict[str, Any]]:
    if examples <= 0:
        raise ValueError("examples must be > 0")
    if candidates < 2:
        raise ValueError("candidates must be >= 2")
    if context_len <= 0:
        raise ValueError("context_len must be > 0")

    out: list[dict[str, Any]] = []
    for pos in positions:
        pos_idx = int(round((context_len - 1) * pos))
        pos_idx = min(max(pos_idx, 0), context_len - 1)
        rng = random.Random(seed + int(pos * 1000) + context_len)
        correct = 0

        for _ in range(examples):
            needle = rng.randrange(vocab_size)
            context = [
                _random_token_excluding(rng, vocab_size, {needle}) for _ in range(context_len)
            ]
            context[pos_idx] = needle

            state = model.clone_state(model_state)
            model.update(state, context, weight=1.0)

            choice_set = [needle]
            seen = {needle}
            while len(choice_set) < candidates:
                cand = _random_token_excluding(rng, vocab_size, seen)
                choice_set.append(cand)
                seen.add(cand)

            pred = max(choice_set, key=lambda token: model._prob(state, token))
            if pred == needle:
                correct += 1

        out.append(
            {
                "position_fraction": pos,
                "position_index": pos_idx,
                "examples": examples,
                "candidates": candidates,
                "accuracy": correct / float(examples),
            }
        )

    return out


def _discover_run_dirs(exp_dir: Path, exp_folder: str, selected_runs: set[str]) -> list[Path]:
    folder = exp_dir / exp_folder
    if not folder.exists():
        return []
    run_dirs: list[Path] = []
    for cfg_path in folder.rglob("phase1_resolved_config.yaml"):
        run_dir = cfg_path.parent
        if selected_runs and run_dir.name not in selected_runs:
            continue
        run_dirs.append(run_dir)
    return sorted(run_dirs, key=lambda p: str(p))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    fieldnames = sorted(keys)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate external phase-1 runs with paper-style proxy metrics: "
            "loss-vs-context, efficiency, NIAH proxy, and decode-trend proxy."
        )
    )
    parser.add_argument("--exp-dir", type=Path, default=Path("./experiments"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--exp-folder", required=True, help="Experiment folder to evaluate.")
    parser.add_argument(
        "--runs",
        default="",
        help="Comma-separated exp_name list. Empty means all runs in exp-folder.",
    )

    parser.add_argument(
        "--contexts",
        default="8192,32768,65536,131072",
        help="Comma-separated context lengths for evaluation.",
    )
    parser.add_argument("--datasets", default="books3", help="Comma-separated dataset keys.")
    parser.add_argument("--dclm-root", type=Path, default=Path("/tmp/phase1_token_data_dclm"))
    parser.add_argument("--books-root", type=Path, default=Path("/tmp/phase1_token_data_books"))
    parser.add_argument("--eval-split", default="val")

    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=0)

    parser.add_argument("--niah-examples", type=int, default=64)
    parser.add_argument("--niah-candidates", type=int, default=16)
    parser.add_argument("--niah-positions", default="0.1,0.5,0.9")

    parser.add_argument("--decode-steps", type=int, default=256)
    parser.add_argument("--decode-prompts", type=int, default=8)

    parser.add_argument("--strict", action="store_true", help="Fail on first metric error.")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.eval_batches <= 0:
        raise ValueError("--eval-batches must be > 0")
    if args.eval_batch_size < 0:
        raise ValueError("--eval-batch-size must be >= 0")
    if args.decode_steps <= 0:
        raise ValueError("--decode-steps must be > 0")
    if args.decode_prompts <= 0:
        raise ValueError("--decode-prompts must be > 0")
    if args.niah_examples <= 0:
        raise ValueError("--niah-examples must be > 0")
    if args.niah_candidates < 2:
        raise ValueError("--niah-candidates must be >= 2")

    exp_dir = args.exp_dir.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    args.dclm_root = args.dclm_root.expanduser().resolve()
    args.books_root = args.books_root.expanduser().resolve()

    contexts = _parse_int_csv(args.contexts)
    datasets = _parse_str_csv(args.datasets)
    niah_positions = _parse_float_csv(args.niah_positions)
    selected_runs = set(_parse_str_csv(args.runs))

    if not contexts:
        raise ValueError("No context lengths provided.")
    if not datasets:
        raise ValueError("No datasets provided.")
    if not niah_positions:
        raise ValueError("No NIAH positions provided.")

    if args.out_json is None:
        args.out_json = exp_dir / f"{args.exp_folder}_external_eval.json"
    else:
        args.out_json = args.out_json.expanduser().resolve()

    if args.out_csv is None:
        args.out_csv = exp_dir / f"{args.exp_folder}_external_eval.csv"
    else:
        args.out_csv = args.out_csv.expanduser().resolve()

    run_dirs = _discover_run_dirs(exp_dir=exp_dir, exp_folder=args.exp_folder, selected_runs=selected_runs)
    if not run_dirs:
        raise FileNotFoundError(
            f"No run directories found under {exp_dir / args.exp_folder}."
        )

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run_name = run_dir.name
        try:
            cfg = _load_run_cfg(run_dir)
            training_cfg = cfg.get("training", {})
            model_cfg = cfg.get("model", {})
            exp_name = str(training_cfg.get("exp_name", run_name))
            vocab_size = int(model_cfg.get("vocab_size", 0))
            if vocab_size <= 0:
                raise ValueError(f"Invalid vocab_size in config for run {run_name}: {vocab_size}")

            ckpt_step, base_state = _load_latest_state(
                checkpoint_root=checkpoint_root,
                exp_folder=args.exp_folder,
                exp_name=exp_name,
            )
            model = TokenStatsModel(vocab_size=vocab_size)
        except Exception as exc:
            rows.append(
                {
                    "record_type": "run_status",
                    "run": run_name,
                    "exp_name": run_name,
                    "status": "failed_setup",
                    "error": str(exc),
                }
            )
            if args.strict:
                raise
            continue

        common = {
            "run": run_name,
            "exp_name": exp_name,
            "checkpoint_step": ckpt_step,
            "runtime_mode": str(training_cfg.get("runtime_mode", "")),
            "train_mode": str(training_cfg.get("train_mode", "")),
            "init_source": str(training_cfg.get("init_source", "")),
            "adapter_recipe": str(training_cfg.get("adapter_recipe", "")),
            "attention_pattern": str(model_cfg.get("attention_pattern", "")),
        }

        for dataset_key in datasets:
            for context_len in contexts:
                row = {
                    "record_type": "loss_efficiency",
                    "dataset_key": dataset_key,
                    "context_length": context_len,
                    **common,
                }
                try:
                    root = _dataset_root_for_key(dataset_key, args)
                    batch_size = args.eval_batch_size or int(training_cfg.get("global_batch_size", 1))
                    flop_proxy = _flops_per_token_proxy(model_cfg=model_cfg, seq_len=context_len)
                    metrics = _eval_loss_efficiency(
                        model=model,
                        model_state=base_state,
                        dataset_path=root,
                        split=args.eval_split,
                        seq_len=context_len,
                        batch_size=batch_size,
                        max_batches=args.eval_batches,
                        seed=args.eval_seed,
                        flops_per_token_proxy=flop_proxy,
                    )
                    row.update(metrics)
                    row["status"] = "ok"
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = str(exc)
                    if args.strict:
                        raise
                rows.append(row)

                decode_row = {
                    "record_type": "decode_trend_proxy",
                    "dataset_key": dataset_key,
                    "context_length": context_len,
                    **common,
                }
                try:
                    root = _dataset_root_for_key(dataset_key, args)
                    decode_metrics = _eval_decode_trend_proxy(
                        model=model,
                        model_state=base_state,
                        dataset_path=root,
                        split=args.eval_split,
                        context_len=context_len,
                        decode_steps=args.decode_steps,
                        prompts=args.decode_prompts,
                        seed=args.eval_seed,
                    )
                    decode_row.update(decode_metrics)
                    decode_row["status"] = "ok"
                except Exception as exc:
                    decode_row["status"] = "failed"
                    decode_row["error"] = str(exc)
                    if args.strict:
                        raise
                rows.append(decode_row)

        for context_len in contexts:
            try:
                niah_rows = _eval_niah_proxy(
                    model=model,
                    model_state=base_state,
                    vocab_size=vocab_size,
                    context_len=context_len,
                    examples=args.niah_examples,
                    candidates=args.niah_candidates,
                    positions=niah_positions,
                    seed=args.eval_seed,
                )
                for nr in niah_rows:
                    rows.append(
                        {
                            "record_type": "niah_proxy",
                            "context_length": context_len,
                            "status": "ok",
                            **common,
                            **nr,
                        }
                    )
            except Exception as exc:
                rows.append(
                    {
                        "record_type": "niah_proxy",
                        "context_length": context_len,
                        "status": "failed",
                        "error": str(exc),
                        **common,
                    }
                )
                if args.strict:
                    raise

    summary = {
        "exp_folder": args.exp_folder,
        "contexts": contexts,
        "datasets": datasets,
        "rows": rows,
    }
    _write_json(args.out_json, summary)
    _write_csv(args.out_csv, rows)

    print(f"Wrote eval JSON: {args.out_json}")
    print(f"Wrote eval CSV: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
