#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURES_DIR = REPO_ROOT / "paper" / "plots" / "figures"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "paper" / "plots" / "plot_manifest.json"

COLORS = {
    "S0": "#c9b8a8",
    "S1": "#8e6d5a",
    "S2": "#2a9d8f",
    "S3": "#264653",
}


@dataclass(frozen=True)
class Point:
    label: str
    gpu_hours: float
    loss: float
    color: str
    marker: str
    filled: bool = True
    label_offset: tuple[float, float] = (8.0, 6.0)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(raw: str | float | int | None) -> float:
    if raw is None:
        raise ValueError("Expected numeric value, got None")
    return float(raw)


def _style_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _stage_summary_map(path: Path) -> dict[str, float]:
    rows = _load_csv(path)
    return {row["stage_id"]: _to_float(row["mean"]) for row in rows}


def _run_inventory_map(path: Path) -> dict[str, dict[str, float]]:
    rows = _load_csv(path)
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        stage_id = row["stage_id"]
        result[stage_id] = {
            "tokens_seen": _to_float(row["tokens_seen"]),
            "gpu_hours": _to_float(row["gpu_hours"]),
            "wall_seconds": _to_float(row["wall_seconds"]),
            "loss_mean": _to_float(row["loss_mean"]),
            "tokens_per_second_mean": _to_float(row["tokens_per_second_mean"]),
        }
    return result


def _extract_summary_loss(path: Path) -> dict[str, float]:
    data = _load_json(path)
    rows = data.get("rows", [])
    return {row["stage_id"]: _to_float(row["loss_ce_mean"]) for row in rows}


def _extract_binned_curves(path: Path) -> dict[str, list[tuple[float, float]]]:
    curves: dict[str, list[tuple[float, float]]] = {}
    for row in _load_csv(path):
        stage_id = row["stage_id"]
        curves.setdefault(stage_id, []).append(
            (_to_float(row["center_pos"]), _to_float(row["mean_nll"]))
        )
    return curves


def _extract_continuation_curves(path: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]], float, float]:
    rows = [row for row in _load_csv(path) if row["status"] == "succeeded"]
    iso_quality = sorted(
        (
            (_to_float(row["extra_steps"]), _to_float(row["loss_ce_mean"]))
            for row in rows
            if row["mode"] == "iso_quality"
        ),
        key=lambda item: item[0],
    )
    iso_total_tokens = sorted(
        (
            (_to_float(row["extra_steps"]), _to_float(row["loss_ce_mean"]))
            for row in rows
            if row["mode"] == "iso_total_tokens"
        ),
        key=lambda item: item[0],
    )
    if not iso_quality or not iso_total_tokens:
        raise RuntimeError("Missing continuation curve data")
    base_s2 = _to_float(rows[0]["reference_s2_loss"])
    base_s3 = _to_float(rows[0]["reference_s3_loss"])
    return iso_quality, iso_total_tokens, base_s2, base_s3


def _extract_extension_curves(path: Path) -> dict[str, list[tuple[float, float]]]:
    curves: dict[str, list[tuple[float, float]]] = {}
    for row in _load_csv(path):
        condition = row["condition"]
        curves.setdefault(condition, []).append(
            (_to_float(row["step"]), _to_float(row["loss_ce_smooth"]))
        )
    return curves


def _plot_main_comparison(figures_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    summary_125m = _stage_summary_map(
        REPO_ROOT / "reports/paper/protocol_r_125m_main_v1/tables/stage_summary_loss_mean.csv"
    )
    summary_760m = _stage_summary_map(
        REPO_ROOT
        / "reports/paper/protocol_r_760m_author_seed_v1/tables/stage_summary_loss_mean.csv"
    )

    stage_specs = [
        ("125M", "S0", "S0_125M", 0.0),
        ("125M", "S1", "S1_125M", 1.0),
        ("125M", "S2", "S2_125M", 2.0),
        ("125M", "S3", "S3_125M", 3.0),
        ("760M", "S2", "S2", 5.2),
        ("760M", "S3", "S3", 6.2),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    for scale, short_stage, stage_id, xpos in stage_specs:
        summary = summary_125m if scale == "125M" else summary_760m
        value = summary[stage_id]
        ax.bar(
            xpos,
            value,
            width=0.75,
            color=COLORS[short_stage],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.text(
            xpos,
            value + 0.08,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks([item[3] for item in stage_specs], [item[1] for item in stage_specs])
    ax.set_ylabel("Books32K validation loss")
    ax.text(1.5, 7.05, "125M", ha="center", va="bottom", fontsize=11, weight="bold")
    ax.text(5.7, 7.05, "760M", ha="center", va="bottom", fontsize=11, weight="bold")
    ax.set_ylim(0, 7.3)
    ax.axvline(4.25, color="#cccccc", linestyle="--", linewidth=1.0)

    legend_handles = [
        Patch(facecolor=COLORS["S0"], label="Simple baselines (S0, S1)"),
        Patch(facecolor=COLORS["S2"], label="Warm-start TTT-E2E (S2)"),
        Patch(facecolor=COLORS["S3"], label="Scratch TTT-E2E (S3)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(figures_dir / "main_comparison_bar.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_cost_quality(figures_dir: Path) -> dict[str, float]:
    import matplotlib.pyplot as plt

    cost_rows = _load_csv(
        REPO_ROOT / "reports/paper/warmstart_paper_v1/plot_data/figure1_cost_quality_points.csv"
    )
    run_inventory_760m = _run_inventory_map(
        REPO_ROOT
        / "reports/paper/protocol_r_760m_author_seed_v1/tables/run_inventory.csv"
    )
    adapt_eta = _load_json(
        REPO_ROOT
        / "reports/paper/protocol_r_760m_eta_live_v1_adapt/eta/eta_summary_s2_pair.json"
    )

    exact_points: list[Point] = []
    for row in cost_rows:
        if row["scale"] != "125M" or row["is_available"] != "True":
            continue
        kind = row["point_kind"]
        color = COLORS["S2"] if "warmstart" in kind else COLORS["S3"]
        exact_points.append(
            Point(
                label=f"125M {kind.replace('_', ' ')}",
                gpu_hours=_to_float(row["gpu_hours"]),
                loss=_to_float(row["loss_mean"]),
                color=color,
                marker="o",
                filled=True,
                label_offset={
                    "warmstart_marginal": (8.0, 10.0),
                    "warmstart_full": (8.0, -6.0),
                    "scratch": (8.0, 6.0),
                }[kind],
            )
        )

    adapt_row = next(
        row for row in adapt_eta["rows"] if row["paper_stage_label"] == "S2_ADAPT_760M"
    )
    hybrid_760m_warmstart_gpu_hours = _to_float(
        adapt_row["estimated_training_gpu_hours"]
    ) + run_inventory_760m["S2"]["gpu_hours"]

    partial_points = [
        Point(
            label="760M warm-start marginal",
            gpu_hours=hybrid_760m_warmstart_gpu_hours,
            loss=run_inventory_760m["S2"]["loss_mean"],
            color=COLORS["S2"],
            marker="^",
            filled=False,
        ),
        Point(
            label="760M scratch marginal",
            gpu_hours=run_inventory_760m["S3"]["gpu_hours"],
            loss=run_inventory_760m["S3"]["loss_mean"],
            color=COLORS["S3"],
            marker="D",
            filled=False,
        ),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for point in exact_points + partial_points:
        facecolor = point.color if point.filled else "white"
        ax.scatter(
            point.gpu_hours,
            point.loss,
            s=110,
            marker=point.marker,
            c=facecolor,
            edgecolors=point.color,
            linewidths=2.0,
            zorder=3,
        )
        ax.annotate(
            point.label,
            (point.gpu_hours, point.loss),
            xytext=point.label_offset,
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("GPU-hours")
    ax.set_ylabel("Books32K validation loss")
    ax.set_xlim(-2, 116)
    ax.set_ylim(2.62, 3.96)
    fig.tight_layout()
    fig.savefig(figures_dir / "cost_quality_pareto.pdf", bbox_inches="tight")
    plt.close(fig)

    return {
        "hybrid_760m_warmstart_gpu_hours": hybrid_760m_warmstart_gpu_hours,
        "scratch_760m_gpu_hours": run_inventory_760m["S3"]["gpu_hours"],
    }


def _plot_continuation(figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    iso_quality, iso_total_tokens, base_s2, base_s3 = _extract_continuation_curves(
        REPO_ROOT / "reports/paper/warmstart_paper_v1/plot_data/figure3_continuation_frontier.csv"
    )

    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.plot(
        [x for x, _ in iso_quality],
        [y for _, y in iso_quality],
        color=COLORS["S2"],
        linewidth=2.4,
        marker="o",
        markersize=4,
        label="S2 continuation",
    )
    ax.plot(
        [x for x, _ in iso_total_tokens],
        [y for _, y in iso_total_tokens],
        color=COLORS["S3"],
        linewidth=2.2,
        linestyle="--",
        marker="D",
        markersize=4,
        label="S3 token-equalized continuation",
    )
    ax.axhline(base_s2, color=COLORS["S2"], linestyle=":", linewidth=1.6, label="Base S2")
    ax.axhline(base_s3, color=COLORS["S3"], linestyle=":", linewidth=1.6, label="Base S3")
    ax.set_xlabel("Continuation steps")
    ax.set_ylabel("Books32K validation loss")
    ax.set_xlim(0, 1440)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(figures_dir / "continuation_trajectories.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_per_position_nll(figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    curves_125m = _extract_binned_curves(
        REPO_ROOT / "reports/paper/protocol_r_125m_main_v1/eval/per_position_nll_binned_curves.csv"
    )
    curves_760m = _extract_binned_curves(
        REPO_ROOT
        / "reports/paper/protocol_r_760m_author_seed_v1/eval/per_position_nll_binned_curves.csv"
    )
    summary_125m = _load_json(
        REPO_ROOT / "reports/paper/protocol_r_125m_main_v1/eval/per_position_nll_summary.json"
    )
    summary_760m = _load_json(
        REPO_ROOT
        / "reports/paper/protocol_r_760m_author_seed_v1/eval/per_position_nll_summary.json"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharex=True)
    legend_handles = None
    panel_specs = [
        (
            axes[0],
            "125M",
            curves_125m["S2_125M"],
            curves_125m["S3_125M"],
            summary_125m["delta_summary"]["mean_nll"],
        ),
        (
            axes[1],
            "760M",
            curves_760m["S2"],
            curves_760m["S3"],
            summary_760m["delta_summary"]["mean_nll"],
        ),
    ]
    for ax, title, s2_curve, s3_curve, mean_gap in panel_specs:
        ax.plot(
            [x / 1024.0 for x, _ in s2_curve],
            [y for _, y in s2_curve],
            color=COLORS["S2"],
            linewidth=2.2,
            label="S2 warm-start",
        )
        s3_line, = ax.plot(
            [x / 1024.0 for x, _ in s3_curve],
            [y for _, y in s3_curve],
            color=COLORS["S3"],
            linewidth=2.2,
            label="S3 scratch",
        )
        if legend_handles is None:
            legend_handles = [ax.lines[-2], s3_line]
        ax.set_xlabel("Token position (K)")
        ax.text(
            0.02,
            0.98,
            title,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
        )
        ax.text(
            0.98,
            0.96,
            f"mean gap = {mean_gap:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cccccc"},
        )
    axes[0].set_ylabel("Per-position NLL")
    for ax in axes:
        ax.set_xlim(0, 32)
    fig.legend(
        legend_handles,
        ["S2 warm-start", "S3 scratch"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figures_dir / "per_position_nll.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_extension_training_curves(figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    curves = _extract_extension_curves(
        REPO_ROOT / "reports/paper/warmstart_paper_v1/plot_data/figure4_extension_training_curves.csv"
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    line_styles = {
        "S0": "--",
        "S1": "-",
        "S2": "-",
        "S3": "-",
    }
    for condition in ["S0", "S1", "S2", "S3"]:
        curve = curves[condition]
        ax.plot(
            [x for x, _ in curve],
            [y for _, y in curve],
            linewidth=2.2,
            linestyle=line_styles[condition],
            color=COLORS[condition],
            label=condition,
        )
    ax.set_xlabel("32K extension step")
    ax.set_ylabel("Training loss (smoothed)")
    ax.set_xlim(0, 479)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "extension_training_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_dclm8k_comparison(figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    summary_125m = _extract_summary_loss(
        REPO_ROOT
        / "reports/paper/protocol_r_125m_main_v1/eval/dclm_8k_s2_s3_eval64_summary.json"
    )
    summary_760m = _extract_summary_loss(
        REPO_ROOT
        / "reports/paper/protocol_r_760m_author_seed_v1/eval/dclm_8k_s2_s3_eval64_summary.json"
    )

    specs = [
        ("125M", "S2", summary_125m["S2_125M"], 0.0),
        ("125M", "S3", summary_125m["S3_125M"], 1.0),
        ("760M", "S2", summary_760m["S2"], 3.2),
        ("760M", "S3", summary_760m["S3"], 4.2),
    ]

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for scale, stage, value, xpos in specs:
        ax.bar(
            xpos,
            value,
            width=0.75,
            color=COLORS[stage],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.text(xpos, value + 0.05, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks([item[3] for item in specs], [item[1] for item in specs])
    ax.set_ylabel("DCLM 8K validation loss")
    ax.text(0.22, 0.95, "125M", transform=ax.transAxes, ha="center", va="top", fontsize=11, weight="bold")
    ax.text(0.80, 0.95, "760M", transform=ax.transAxes, ha="center", va="top", fontsize=11, weight="bold")
    ax.axvline(2.1, color="#cccccc", linestyle="--", linewidth=1.0)
    ax.set_ylim(0, 4.5)
    fig.tight_layout()
    fig.savefig(figures_dir / "dclm8k_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready PDF plots.")
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Output directory for figure PDFs.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Where to write the plot manifest JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    figures_dir = args.figures_dir.expanduser().resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "matplotlib is required for plot generation. Install it in the active env."
        ) from exc

    _style_matplotlib()
    _plot_main_comparison(figures_dir)
    pareto_meta = _plot_cost_quality(figures_dir)
    _plot_continuation(figures_dir)
    _plot_per_position_nll(figures_dir)
    _plot_extension_training_curves(figures_dir)
    _plot_dclm8k_comparison(figures_dir)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "figures_dir": str(figures_dir),
        "files": sorted(str(path) for path in figures_dir.glob("*.pdf")),
        "caveats": {
            "cost_quality_pareto": [
                "125M points are exact branch and marginal costs from canonical run_inventory rows.",
                "760M warm-start marginal cost combines ETA-derived S2_ADAPT cost with observed resumed S2 cost.",
                "760M full-branch cost is intentionally omitted because author-provided seed GPU-hours are external.",
            ],
            "continuation_trajectories": [
                "Only 125M continuation data exists; 760M continuation was not run."
            ],
            "dclm8k_comparison": [
                "Plots final S2/S3 checkpoints on the DCLM 8K surface; 760M S0/S1 controls are not available."
            ],
        },
        "computed_values": pareto_meta,
    }
    args.manifest_path.expanduser().resolve().write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    for path in figures_dir.glob("*.pdf"):
        print(f"Wrote figure: {path}")
    print(f"Wrote manifest: {args.manifest_path.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
