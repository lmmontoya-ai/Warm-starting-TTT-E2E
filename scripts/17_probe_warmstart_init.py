#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttt.dataloader import build_batch_iterator
from ttt.infra import Phase1Checkpointer
from ttt.model import TokenStatsModel, TokenStatsState
from ttt.research.types import utc_now_iso


def _flatten_targets(batch) -> list[int]:
    tokens: list[int] = []
    for sample in batch.samples:
        tokens.extend(sample.target_tokens)
    return tokens


def _infer_vocab_size(state: TokenStatsState, fallback: int) -> int:
    if state.token_counts:
        mx = max(int(k) for k in state.token_counts.keys())
        return max(fallback, mx + 1)
    return fallback


def _linear_slope(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / float(n)
    num = 0.0
    den = 0.0
    for idx, value in enumerate(values):
        dx = float(idx) - x_mean
        dy = value - y_mean
        num += dx * dy
        den += dx * dx
    if den <= 0.0:
        return 0.0
    return num / den


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe warm-start initialization quality by measuring early token_stats "
            "loss trajectory and slope over a fixed number of update steps."
        )
    )
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--exp-folder", required=True)
    parser.add_argument("--exp-name", required=True)

    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--probe-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=0)
    parser.add_argument("--update-weight", type=float, default=1.0)

    parser.add_argument("--report-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seq_length <= 0:
        raise ValueError("--seq-length must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.probe_steps <= 0:
        raise ValueError("--probe-steps must be > 0")
    if args.update_weight <= 0.0:
        raise ValueError("--update-weight must be > 0")

    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()

    checkpointer = Phase1Checkpointer(checkpoint_root / args.exp_folder / args.exp_name)
    restored = checkpointer.load(step=None)
    if restored is None:
        raise FileNotFoundError(
            f"Missing checkpoint for {args.exp_folder}/{args.exp_name} under {checkpoint_root}"
        )

    raw_state = restored.payload.get("model_state")
    if not isinstance(raw_state, dict):
        raise ValueError(
            f"Checkpoint {args.exp_name} is missing model_state. "
            "This probe requires token_stats-compatible payloads."
        )

    state = TokenStatsState.from_jsonable(raw_state)
    fallback_vocab = args.vocab_size if args.vocab_size > 0 else 32000
    vocab_size = _infer_vocab_size(state=state, fallback=fallback_vocab)
    model = TokenStatsModel(vocab_size=vocab_size)

    iterator = build_batch_iterator(
        dataset_path=str(dataset_root),
        split=args.split,
        seq_len=args.seq_length,
        global_batch_size=args.batch_size,
        repeat=False,
        shuffle=False,
        seed=args.seed,
        dummy_dataset=False,
        vocab_size=vocab_size,
    )

    steps: list[dict[str, float]] = []
    for idx in range(args.probe_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        targets = _flatten_targets(batch)
        if not targets:
            continue

        loss_before = model.nll(state, targets)
        model.update(state, targets, weight=args.update_weight)
        loss_after = model.nll(state, targets)
        steps.append(
            {
                "step": float(idx),
                "loss_before": float(loss_before),
                "loss_after": float(loss_after),
                "tokens": float(len(targets)),
            }
        )

    if not steps:
        raise ValueError("Probe consumed no tokens; dataset may be too short.")

    loss_before_curve = [float(row["loss_before"]) for row in steps]
    loss_after_curve = [float(row["loss_after"]) for row in steps]

    report = {
        "schema_version": "1.0",
        "created_at_utc": utc_now_iso(),
        "exp_folder": args.exp_folder,
        "exp_name": args.exp_name,
        "checkpoint_step": restored.step,
        "checkpoint_path": str(checkpoint_root / args.exp_folder / args.exp_name),
        "dataset_root": str(dataset_root),
        "split": args.split,
        "seq_length": args.seq_length,
        "batch_size": args.batch_size,
        "probe_steps_requested": args.probe_steps,
        "probe_steps_completed": len(steps),
        "vocab_size": vocab_size,
        "initial_loss": loss_before_curve[0],
        "final_loss_before": loss_before_curve[-1],
        "final_loss_after": loss_after_curve[-1],
        "loss_before_slope": _linear_slope(loss_before_curve),
        "loss_after_slope": _linear_slope(loss_after_curve),
        "steps": steps,
    }

    if args.report_out is None:
        report_out = (
            Path("./artifacts/warmstart_probe")
            / f"{args.exp_folder}__{args.exp_name}__probe.json"
        ).resolve()
    else:
        report_out = args.report_out.expanduser().resolve()
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"Wrote warm-start probe report: {report_out}")
    print(
        "Probe summary: "
        f"initial_loss={report['initial_loss']:.6f} "
        f"final_before={report['final_loss_before']:.6f} "
        f"slope={report['loss_before_slope']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
