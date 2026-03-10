#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic paper figures from table CSV files."
    )
    parser.add_argument("--paper-run-id", required=True)
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "matplotlib is required for figure generation. Install it in the env and rerun."
        ) from exc

    if args.tables_dir is None:
        tables_dir = (Path("./reports/paper") / args.paper_run_id / "tables").resolve()
    else:
        tables_dir = args.tables_dir.expanduser().resolve()

    if args.figures_dir is None:
        figures_dir = (Path("./reports/paper") / args.paper_run_id / "figures").resolve()
    else:
        figures_dir = args.figures_dir.expanduser().resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    frontier_rows = _load_csv(tables_dir / "quality_vs_gpu_hour_frontier.csv")
    core_rows = _load_csv(tables_dir / "warmstart_core_deltas.csv")

    # Figure 1: quality vs compute frontier (loss vs gpu-hours).
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in frontier_rows:
        x = _to_float(row.get("gpu_hours"))
        y = _to_float(row.get("loss_mean"))
        stage = str(row.get("stage_id", ""))
        if x is None or y is None or not stage:
            continue
        grouped[stage].append((x, y))

    fig1 = figures_dir / "quality_vs_gpu_hours.png"
    plt.figure(figsize=(8, 5))
    for stage, points in sorted(grouped.items()):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.scatter(xs, ys, label=stage)
    plt.xlabel("GPU-hours")
    plt.ylabel("Loss (mean)")
    plt.title("Quality vs Compute Frontier")
    if grouped:
        plt.legend()
    plt.tight_layout()
    plt.savefig(fig1, dpi=160)
    plt.close()

    # Figure 2: core stage deltas for loss metric.
    loss_row = None
    for row in core_rows:
        if str(row.get("metric", "")) == "loss_mean":
            loss_row = row
            break

    fig2 = figures_dir / "loss_stage_deltas.png"
    plt.figure(figsize=(7, 4))
    labels = ["S1-S0", "S2-S1", "S2-S3"]
    values = [
        _to_float(loss_row.get("s1_s0_delta")) if loss_row else None,
        _to_float(loss_row.get("s2_s1_delta")) if loss_row else None,
        _to_float(loss_row.get("s2_s3_delta")) if loss_row else None,
    ]
    values_plot = [0.0 if v is None else v for v in values]
    plt.bar(labels, values_plot)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("Delta (loss)")
    plt.title("Core Stage Deltas (Loss)")
    plt.tight_layout()
    plt.savefig(fig2, dpi=160)
    plt.close()

    # Figure 3: throughput vs quality tradeoff.
    tps_points = []
    for row in frontier_rows:
        x = _to_float(row.get("tokens_per_second_mean"))
        y = _to_float(row.get("loss_mean"))
        if x is None or y is None:
            continue
        tps_points.append((x, y))

    fig3 = figures_dir / "throughput_vs_loss.png"
    plt.figure(figsize=(8, 5))
    if tps_points:
        plt.scatter([p[0] for p in tps_points], [p[1] for p in tps_points], alpha=0.8)
    plt.xlabel("Tokens / second (mean)")
    plt.ylabel("Loss (mean)")
    plt.title("Throughput vs Quality")
    plt.tight_layout()
    plt.savefig(fig3, dpi=160)
    plt.close()

    print(f"Wrote figure: {fig1}")
    print(f"Wrote figure: {fig2}")
    print(f"Wrote figure: {fig3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
