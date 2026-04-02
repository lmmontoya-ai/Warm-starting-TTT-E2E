#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _find_row(rows: list[dict[str, Any]], stage_id: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("stage_id", "")).strip() == stage_id:
            return row
    raise KeyError(f"Missing stage row for {stage_id}")


def _band_rows(*, values: np.ndarray, stage_id: str, run_id: str, n_bands: int) -> list[dict[str, Any]]:
    n_positions = int(values.shape[0])
    edges = np.linspace(0, n_positions, n_bands + 1, dtype=int)
    rows: list[dict[str, Any]] = []
    for idx in range(n_bands):
        start = int(edges[idx])
        end = int(edges[idx + 1])
        band = values[start:end]
        if band.size == 0:
            continue
        lo_pct = int(round(100.0 * start / n_positions))
        hi_pct = int(round(100.0 * end / n_positions))
        rows.append(
            {
                "stage_id": stage_id,
                "run_id": run_id,
                "band_index": idx,
                "band_label": f"{lo_pct:02d}-{hi_pct:02d}%",
                "start_pos": start + 1,
                "end_pos": end,
                "n_positions": int(band.size),
                "mean_nll": float(np.mean(band)),
                "min_nll": float(np.min(band)),
                "max_nll": float(np.max(band)),
            }
        )
    return rows


def _curve_bins(values: np.ndarray, n_bins: int) -> list[dict[str, Any]]:
    n_positions = int(values.shape[0])
    edges = np.linspace(0, n_positions, n_bins + 1, dtype=int)
    rows: list[dict[str, Any]] = []
    for idx in range(n_bins):
        start = int(edges[idx])
        end = int(edges[idx + 1])
        band = values[start:end]
        if band.size == 0:
            continue
        center = start + (band.size / 2.0)
        rows.append(
            {
                "bin_index": idx,
                "start_pos": start + 1,
                "end_pos": end,
                "center_pos": center,
                "mean_nll": float(np.mean(band)),
            }
        )
    return rows


def _quarter_summary(values: np.ndarray) -> dict[str, float]:
    n = int(values.shape[0])
    q = max(1, n // 4)
    return {
        "mean_nll": float(np.mean(values)),
        "head_25_mean": float(np.mean(values[:q])),
        "middle_50_mean": float(np.mean(values[q : n - q])),
        "tail_25_mean": float(np.mean(values[n - q :])),
        "best_pos_nll": float(np.min(values)),
        "worst_pos_nll": float(np.max(values)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and plot per-position NLL curves from parity eval outputs."
    )
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--from-stage", default="S3")
    parser.add_argument("--to-stage", default="S2")
    parser.add_argument("--report-root", type=Path, default=None)
    parser.add_argument("--n-bands", type=int, default=10)
    parser.add_argument("--plot-bins", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "matplotlib is required for per-position NLL figure generation."
        ) from exc

    summary_json = args.summary_json.expanduser().resolve()
    payload = _load_json(summary_json)
    paper_run_id = str(payload.get("paper_run_id", "")).strip()
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"Invalid summary rows in {summary_json}")

    if args.report_root is None:
        report_root = summary_json.parent.parent.resolve()
    else:
        report_root = args.report_root.expanduser().resolve()
    eval_dir = report_root / "eval"
    tables_dir = report_root / "tables"
    figures_dir = report_root / "figures"
    eval_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    from_row = _find_row(rows, args.from_stage)
    to_row = _find_row(rows, args.to_stage)

    from_path = Path(str(from_row.get("per_position_nll_path", ""))).expanduser()
    to_path = Path(str(to_row.get("per_position_nll_path", ""))).expanduser()
    if not from_path.exists():
        raise FileNotFoundError(f"Missing per-position NLL curve: {from_path}")
    if not to_path.exists():
        raise FileNotFoundError(f"Missing per-position NLL curve: {to_path}")

    from_curve = np.asarray(np.load(from_path), dtype=np.float64)
    to_curve = np.asarray(np.load(to_path), dtype=np.float64)
    if from_curve.ndim != 1 or to_curve.ndim != 1:
        raise ValueError("Expected 1D per-position NLL curves")
    if from_curve.shape != to_curve.shape:
        raise ValueError(
            f"Mismatched curve lengths: {from_curve.shape} vs {to_curve.shape}"
        )

    delta_curve = to_curve - from_curve
    n_positions = int(to_curve.shape[0])

    stage_summary_rows = [
        {
            "paper_run_id": paper_run_id,
            "stage_id": str(from_row.get("stage_id", "")),
            "run_id": str(from_row.get("run_id", "")),
            "curve_path": str(from_path),
            "n_positions": n_positions,
            **_quarter_summary(from_curve),
        },
        {
            "paper_run_id": paper_run_id,
            "stage_id": str(to_row.get("stage_id", "")),
            "run_id": str(to_row.get("run_id", "")),
            "curve_path": str(to_path),
            "n_positions": n_positions,
            **_quarter_summary(to_curve),
        },
    ]

    delta_summary = {
        "paper_run_id": paper_run_id,
        "from_stage": args.from_stage,
        "to_stage": args.to_stage,
        "delta_definition": "to_stage minus from_stage",
        "n_positions": n_positions,
        **_quarter_summary(delta_curve),
    }

    band_rows = _band_rows(
        values=from_curve,
        stage_id=str(from_row.get("stage_id", "")),
        run_id=str(from_row.get("run_id", "")),
        n_bands=int(args.n_bands),
    ) + _band_rows(
        values=to_curve,
        stage_id=str(to_row.get("stage_id", "")),
        run_id=str(to_row.get("run_id", "")),
        n_bands=int(args.n_bands),
    )

    delta_band_rows = []
    for from_band, to_band in zip(
        _band_rows(
            values=from_curve,
            stage_id=str(from_row.get("stage_id", "")),
            run_id=str(from_row.get("run_id", "")),
            n_bands=int(args.n_bands),
        ),
        _band_rows(
            values=to_curve,
            stage_id=str(to_row.get("stage_id", "")),
            run_id=str(to_row.get("run_id", "")),
            n_bands=int(args.n_bands),
        ),
        strict=True,
    ):
        delta_band_rows.append(
            {
                "paper_run_id": paper_run_id,
                "from_stage": args.from_stage,
                "to_stage": args.to_stage,
                "band_index": int(to_band["band_index"]),
                "band_label": str(to_band["band_label"]),
                "start_pos": int(to_band["start_pos"]),
                "end_pos": int(to_band["end_pos"]),
                "delta_mean_nll": float(to_band["mean_nll"]) - float(from_band["mean_nll"]),
                "from_mean_nll": float(from_band["mean_nll"]),
                "to_mean_nll": float(to_band["mean_nll"]),
            }
        )

    from_bins = _curve_bins(from_curve, int(args.plot_bins))
    to_bins = _curve_bins(to_curve, int(args.plot_bins))
    delta_bins = _curve_bins(delta_curve, int(args.plot_bins))

    binned_curve_rows = []
    for row in from_bins:
        binned_curve_rows.append(
            {
                "paper_run_id": paper_run_id,
                "stage_id": args.from_stage,
                **row,
            }
        )
    for row in to_bins:
        binned_curve_rows.append(
            {
                "paper_run_id": paper_run_id,
                "stage_id": args.to_stage,
                **row,
            }
        )

    summary_path = eval_dir / "per_position_nll_summary.json"
    stage_summary_csv = tables_dir / "per_position_nll_stage_summary.csv"
    band_csv = tables_dir / "per_position_nll_band_summary.csv"
    delta_band_csv = tables_dir / "per_position_nll_delta_bands.csv"
    binned_csv = eval_dir / "per_position_nll_binned_curves.csv"
    overlay_fig = figures_dir / "per_position_nll_overlay.png"
    delta_fig = figures_dir / "per_position_nll_delta.png"

    _write_json(
        summary_path,
        {
            "schema_version": "1.0",
            "paper_run_id": paper_run_id,
            "summary_json": str(summary_json),
            "from_stage": args.from_stage,
            "to_stage": args.to_stage,
            "delta_definition": "to_stage minus from_stage",
            "n_positions": n_positions,
            "n_bands": int(args.n_bands),
            "plot_bins": int(args.plot_bins),
            "stage_summaries": stage_summary_rows,
            "delta_summary": delta_summary,
            "band_summary_csv": str(band_csv),
            "delta_band_csv": str(delta_band_csv),
            "binned_curve_csv": str(binned_csv),
            "overlay_figure": str(overlay_fig),
            "delta_figure": str(delta_fig),
        },
    )
    _write_csv(stage_summary_csv, stage_summary_rows)
    _write_csv(band_csv, band_rows)
    _write_csv(delta_band_csv, delta_band_rows)
    _write_csv(binned_csv, binned_curve_rows)

    plt.figure(figsize=(9, 5))
    plt.plot(
        [row["center_pos"] for row in from_bins],
        [row["mean_nll"] for row in from_bins],
        label=args.from_stage,
        linewidth=2.0,
    )
    plt.plot(
        [row["center_pos"] for row in to_bins],
        [row["mean_nll"] for row in to_bins],
        label=args.to_stage,
        linewidth=2.0,
    )
    plt.xlabel("Token Position")
    plt.ylabel("Per-position NLL")
    plt.title("Books32K Per-position NLL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(overlay_fig, dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(
        [row["center_pos"] for row in delta_bins],
        [row["mean_nll"] for row in delta_bins],
        linewidth=2.0,
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Token Position")
    plt.ylabel(f"Per-position NLL Delta ({args.to_stage} - {args.from_stage})")
    plt.title("Books32K Per-position NLL Delta")
    plt.tight_layout()
    plt.savefig(delta_fig, dpi=160)
    plt.close()

    print(f"Wrote summary: {summary_path}")
    print(f"Wrote table: {stage_summary_csv}")
    print(f"Wrote table: {band_csv}")
    print(f"Wrote table: {delta_band_csv}")
    print(f"Wrote table: {binned_csv}")
    print(f"Wrote figure: {overlay_fig}")
    print(f"Wrote figure: {delta_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
