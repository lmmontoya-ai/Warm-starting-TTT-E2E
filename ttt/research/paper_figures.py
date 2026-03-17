"""Dedicated publication-quality figure pipeline for the warm-start paper."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

from .types import utc_now_iso


FIGURE_IDS = ("figure1", "figure2", "figure3", "figure4")
MAIN_CONDITION_ORDER = ("S0", "S1", "S2", "S3")
FIGURE1_POINT_ORDER = ("warmstart_marginal", "warmstart_full", "scratch")
FIGURE3_MODE_ORDER = ("iso_quality", "iso_total_tokens")
FIGURE4_SMOOTH_WINDOW = 25

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class Figure1PointSpec:
    point_kind: str
    quality_stage_id: str
    cost_stage_ids: tuple[str, ...]
    cost_scope: str

    @staticmethod
    def from_dict(point_kind: str, payload: dict[str, Any]) -> "Figure1PointSpec":
        return Figure1PointSpec(
            point_kind=str(point_kind),
            quality_stage_id=str(payload.get("quality_stage_id", "")),
            cost_stage_ids=tuple(str(x) for x in payload.get("cost_stage_ids", [])),
            cost_scope=str(payload.get("cost_scope", "")),
        )


@dataclass(frozen=True)
class ScaleFigureSpec:
    scale: str
    main_paper_run_id: str
    main_report_root: str
    ablation_paper_run_id: str = ""
    ablation_report_root: str = ""
    stage_ids: dict[str, str] = field(default_factory=dict)
    enabled_figures: tuple[str, ...] = field(default_factory=tuple)
    required_in_strict: tuple[str, ...] = field(default_factory=tuple)
    figure1_cost_policy: str = ""
    figure1_points: dict[str, Figure1PointSpec] = field(default_factory=dict)
    figure1_omitted_points: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_dict(scale: str, payload: dict[str, Any]) -> "ScaleFigureSpec":
        figure1 = payload.get("figure1", {})
        if not isinstance(figure1, dict):
            figure1 = {}
        point_payload = figure1.get("points", {})
        if not isinstance(point_payload, dict):
            point_payload = {}
        omitted_payload = figure1.get("omitted_points", {})
        if not isinstance(omitted_payload, dict):
            omitted_payload = {}
        return ScaleFigureSpec(
            scale=str(scale),
            main_paper_run_id=str(payload.get("main_paper_run_id", "")),
            main_report_root=str(payload.get("main_report_root", "")),
            ablation_paper_run_id=str(payload.get("ablation_paper_run_id", "")),
            ablation_report_root=str(payload.get("ablation_report_root", "")),
            stage_ids={str(k): str(v) for k, v in dict(payload.get("stage_ids", {})).items()},
            enabled_figures=tuple(str(x) for x in payload.get("enabled_figures", [])),
            required_in_strict=tuple(str(x) for x in payload.get("required_in_strict", [])),
            figure1_cost_policy=str(figure1.get("cost_policy", "")),
            figure1_points={
                str(point_kind): Figure1PointSpec.from_dict(str(point_kind), dict(spec))
                for point_kind, spec in point_payload.items()
            },
            figure1_omitted_points={str(k): str(v) for k, v in omitted_payload.items()},
        )

    def is_enabled(self, figure_id: str) -> bool:
        return figure_id in self.enabled_figures

    def requires_strict(self, figure_id: str) -> bool:
        return figure_id in self.required_in_strict

    def resolve_main_report_root(self, repo_root: Path) -> Path:
        return _resolve_path(repo_root, self.main_report_root)

    def resolve_ablation_report_root(self, repo_root: Path) -> Path | None:
        if not self.ablation_report_root:
            return None
        return _resolve_path(repo_root, self.ablation_report_root)

    def figure1_point_kinds(self) -> list[str]:
        out = list(self.figure1_points.keys())
        for point_kind in self.figure1_omitted_points.keys():
            if point_kind not in out:
                out.append(point_kind)
        return out


@dataclass(frozen=True)
class FigureSetSpec:
    figure_set_id: str
    schema_version: str
    output_report_id: str
    scales: dict[str, ScaleFigureSpec]

    @staticmethod
    def from_dict(figure_set_id: str, payload: dict[str, Any]) -> "FigureSetSpec":
        scales_payload = payload.get("scales", {})
        if not isinstance(scales_payload, dict):
            raise ValueError(f"Expected mapping for figure_sets.{figure_set_id}.scales")
        scales = {
            str(scale): ScaleFigureSpec.from_dict(str(scale), dict(spec))
            for scale, spec in scales_payload.items()
        }
        spec = FigureSetSpec(
            figure_set_id=str(figure_set_id),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            output_report_id=str(payload.get("output_report_id", figure_set_id)),
            scales=scales,
        )
        _validate_figure_set(spec)
        return spec


def load_figure_set_spec(
    path: str | Path,
    *,
    figure_set_id: str,
) -> FigureSetSpec:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping root in {path}")
    payload = raw.get("figure_sets", {})
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping for figure_sets in {path}")
    if figure_set_id not in payload:
        raise KeyError(f"Unknown figure_set_id={figure_set_id} in {path}")
    node = payload[figure_set_id]
    if not isinstance(node, dict):
        raise ValueError(f"Expected mapping for figure_sets.{figure_set_id}")
    return FigureSetSpec.from_dict(figure_set_id, node)


def prepare_warmstart_figure_data(
    *,
    repo_root: Path,
    exp_dir: Path,
    spec: FigureSetSpec,
    plot_data_dir: Path,
    strict: bool,
    spec_path: Path,
) -> dict[str, Any]:
    plot_data_dir.mkdir(parents=True, exist_ok=True)

    figure1_rows, figure1_meta = _prepare_figure1_cost_quality(
        repo_root=repo_root,
        spec=spec,
        strict=strict,
    )
    figure2_rows, figure2_meta = _prepare_figure2_main_comparison(
        repo_root=repo_root,
        spec=spec,
        strict=strict,
    )
    figure3_rows, figure3_meta = _prepare_figure3_continuation(
        repo_root=repo_root,
        spec=spec,
        strict=strict,
    )
    figure4_rows, figure4_meta = _prepare_figure4_training_curves(
        repo_root=repo_root,
        exp_dir=exp_dir,
        spec=spec,
        strict=strict,
    )

    figure1_path = plot_data_dir / "figure1_cost_quality_points.csv"
    figure2_path = plot_data_dir / "figure2_main_comparison.csv"
    figure3_path = plot_data_dir / "figure3_continuation_frontier.csv"
    figure4_path = plot_data_dir / "figure4_extension_training_curves.csv"

    _write_csv(
        figure1_path,
        figure1_rows,
        fields=[
            "scale",
            "point_kind",
            "cost_scope",
            "gpu_hours",
            "loss_mean",
            "paper_run_id",
            "quality_stage_id",
            "cost_stage_ids",
            "cost_policy",
            "is_available",
            "omission_reason",
        ],
    )
    _write_csv(
        figure2_path,
        figure2_rows,
        fields=[
            "scale",
            "condition",
            "stage_id",
            "loss_mean",
            "gpu_hours",
            "status",
            "paper_run_id",
            "omission_reason",
        ],
    )
    _write_csv(
        figure3_path,
        figure3_rows,
        fields=[
            "scale",
            "mode",
            "extra_steps",
            "checkpoint_step",
            "loss_ce_mean",
            "matched_target",
            "reference_s2_loss",
            "reference_s3_loss",
            "stage_id",
            "run_id",
            "status",
            "snapshot_dir",
        ],
    )
    _write_csv(
        figure4_path,
        figure4_rows,
        fields=[
            "scale",
            "condition",
            "stage_id",
            "run_id",
            "step",
            "loss_ce_raw",
            "loss_ce_smooth",
        ],
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "figure_set_id": spec.figure_set_id,
        "output_report_id": spec.output_report_id,
        "created_at_utc": utc_now_iso(),
        "strict_mode": bool(strict),
        "spec_path": str(spec_path),
        "plot_data_dir": str(plot_data_dir),
        "exp_dir": str(exp_dir),
        "figures": {
            "figure1": {
                **figure1_meta,
                "csv_path": str(figure1_path),
                "row_count": len(figure1_rows),
            },
            "figure2": {
                **figure2_meta,
                "csv_path": str(figure2_path),
                "row_count": len(figure2_rows),
            },
            "figure3": {
                **figure3_meta,
                "csv_path": str(figure3_path),
                "row_count": len(figure3_rows),
            },
            "figure4": {
                **figure4_meta,
                "csv_path": str(figure4_path),
                "row_count": len(figure4_rows),
            },
        },
    }
    _write_json(plot_data_dir / "plot_data_manifest.json", manifest)
    return manifest


def render_warmstart_figures(
    *,
    figure_set_id: str,
    plot_data_dir: Path,
    figures_dir: Path,
    formats: Sequence[str],
    strict: bool,
) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "matplotlib is required for warm-start paper figures. Install it in the env and rerun."
        ) from exc

    plot_data_dir = plot_data_dir.resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_manifest = _load_json(plot_data_dir / "plot_data_manifest.json")
    _validate_plot_manifest_for_render(plot_manifest, strict=strict)

    _apply_publication_style(plt)

    outputs: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "figure_set_id": figure_set_id,
        "created_at_utc": utc_now_iso(),
        "figures_dir": str(figures_dir),
        "formats": list(formats),
        "figures": {},
    }

    figure1_rows = _load_csv(plot_data_dir / "figure1_cost_quality_points.csv")
    figure1_manifest = _plot_figure1_cost_quality(
        plt=plt,
        rows=figure1_rows,
        output_stem=figures_dir / "figure1_cost_quality",
        formats=formats,
        plot_manifest=plot_manifest["figures"]["figure1"],
    )
    if figure1_manifest is not None:
        outputs["figures"]["figure1"] = figure1_manifest
        _write_json(figures_dir / "figure1_cost_quality.manifest.json", figure1_manifest)

    figure2_rows = _load_csv(plot_data_dir / "figure2_main_comparison.csv")
    figure2_manifest = _plot_figure2_main_comparison(
        plt=plt,
        rows=figure2_rows,
        output_stem=figures_dir / "figure2_main_comparison",
        formats=formats,
        plot_manifest=plot_manifest["figures"]["figure2"],
    )
    if figure2_manifest is not None:
        outputs["figures"]["figure2"] = figure2_manifest
        _write_json(figures_dir / "figure2_main_comparison.manifest.json", figure2_manifest)

    figure3_rows = _load_csv(plot_data_dir / "figure3_continuation_frontier.csv")
    figure3_manifest = _plot_figure3_continuation(
        plt=plt,
        rows=figure3_rows,
        output_stem=figures_dir / "figure3_continuation",
        formats=formats,
        plot_manifest=plot_manifest["figures"]["figure3"],
    )
    if figure3_manifest is not None:
        outputs["figures"]["figure3"] = figure3_manifest
        _write_json(figures_dir / "figure3_continuation.manifest.json", figure3_manifest)

    figure4_rows = _load_csv(plot_data_dir / "figure4_extension_training_curves.csv")
    figure4_manifest = _plot_figure4_training_curves(
        plt=plt,
        rows=figure4_rows,
        output_stem=figures_dir / "figure4_extension_training",
        formats=formats,
        plot_manifest=plot_manifest["figures"]["figure4"],
    )
    if figure4_manifest is not None:
        outputs["figures"]["figure4"] = figure4_manifest
        _write_json(figures_dir / "figure4_extension_training.manifest.json", figure4_manifest)

    _write_json(figures_dir / "figure_render_manifest.json", outputs)
    return outputs


def _validate_figure_set(spec: FigureSetSpec) -> None:
    if not spec.output_report_id:
        raise ValueError(f"figure_set {spec.figure_set_id} missing output_report_id")
    if not spec.scales:
        raise ValueError(f"figure_set {spec.figure_set_id} has no scales")
    for scale, scale_spec in spec.scales.items():
        unknown_enabled = set(scale_spec.enabled_figures) - set(FIGURE_IDS)
        unknown_required = set(scale_spec.required_in_strict) - set(FIGURE_IDS)
        if unknown_enabled:
            raise ValueError(f"{scale}: unknown enabled figures {sorted(unknown_enabled)}")
        if unknown_required:
            raise ValueError(f"{scale}: unknown strict figures {sorted(unknown_required)}")
        if not scale_spec.main_paper_run_id:
            raise ValueError(f"{scale}: missing main_paper_run_id")
        if not scale_spec.main_report_root:
            raise ValueError(f"{scale}: missing main_report_root")
        for condition in ("S0", "S1", "S2", "S3"):
            if condition not in scale_spec.stage_ids:
                raise ValueError(f"{scale}: missing stage mapping for {condition}")
        if "figure1" in scale_spec.enabled_figures and not scale_spec.figure1_points:
            raise ValueError(f"{scale}: figure1 enabled but no point specs defined")


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = {field: row.get(field, "") for field in fields}
            writer.writerow(payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    parsed = _to_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _inventory_by_stage(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        stage_id = str(row.get("stage_id", "")).strip()
        if not stage_id:
            continue
        out[stage_id] = row
    return out


def _prepare_figure1_cost_quality(
    *,
    repo_root: Path,
    spec: FigureSetSpec,
    strict: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    available_scales: list[str] = []
    skipped_scales: list[str] = []
    omissions: list[dict[str, str]] = []
    sources: list[str] = []

    for scale_name, scale_spec in spec.scales.items():
        if not scale_spec.is_enabled("figure1"):
            continue
        inventory_path = scale_spec.resolve_main_report_root(repo_root) / "tables" / "run_inventory.csv"
        sources.append(str(inventory_path))
        point_kinds = scale_spec.figure1_point_kinds()
        if not inventory_path.exists():
            skipped_scales.append(scale_name)
            reason = "Missing canonical run_inventory.csv."
            for point_kind in point_kinds:
                omission_reason = scale_spec.figure1_omitted_points.get(point_kind, reason)
                rows.append(
                    {
                        "scale": scale_name,
                        "point_kind": point_kind,
                        "cost_scope": "",
                        "gpu_hours": None,
                        "loss_mean": None,
                        "paper_run_id": scale_spec.main_paper_run_id,
                        "quality_stage_id": "",
                        "cost_stage_ids": "",
                        "cost_policy": scale_spec.figure1_cost_policy,
                        "is_available": False,
                        "omission_reason": omission_reason,
                    }
                )
                omissions.append(
                    {
                        "figure": "figure1",
                        "scale": scale_name,
                        "point_kind": point_kind,
                        "reason": omission_reason,
                    }
                )
            if strict and scale_spec.requires_strict("figure1"):
                raise FileNotFoundError(
                    f"Figure 1 requires {inventory_path} for scale {scale_name} in strict mode."
                )
            continue

        inventory_rows = _load_csv(inventory_path)
        inventory = _inventory_by_stage(inventory_rows)
        scale_has_available = False
        for point_kind in point_kinds:
            if point_kind in scale_spec.figure1_omitted_points:
                omission_reason = scale_spec.figure1_omitted_points[point_kind]
                rows.append(
                    {
                        "scale": scale_name,
                        "point_kind": point_kind,
                        "cost_scope": "",
                        "gpu_hours": None,
                        "loss_mean": None,
                        "paper_run_id": scale_spec.main_paper_run_id,
                        "quality_stage_id": "",
                        "cost_stage_ids": "",
                        "cost_policy": scale_spec.figure1_cost_policy,
                        "is_available": False,
                        "omission_reason": omission_reason,
                    }
                )
                omissions.append(
                    {
                        "figure": "figure1",
                        "scale": scale_name,
                        "point_kind": point_kind,
                        "reason": omission_reason,
                    }
                )
                continue

            point_spec = scale_spec.figure1_points[point_kind]
            quality_row = inventory.get(point_spec.quality_stage_id)
            cost_rows = [inventory.get(stage_id) for stage_id in point_spec.cost_stage_ids]
            reason_parts: list[str] = []
            if quality_row is None:
                reason_parts.append(f"Missing quality stage {point_spec.quality_stage_id}")
            for stage_id, cost_row in zip(point_spec.cost_stage_ids, cost_rows):
                if cost_row is None:
                    reason_parts.append(f"Missing cost stage {stage_id}")
            loss_mean = _to_float((quality_row or {}).get("loss_mean"))
            if quality_row is not None and loss_mean is None:
                reason_parts.append(f"Missing loss_mean for {point_spec.quality_stage_id}")
            gpu_hours_values: list[float] = []
            for stage_id, cost_row in zip(point_spec.cost_stage_ids, cost_rows):
                if cost_row is None:
                    continue
                gpu_hours = _to_float(cost_row.get("gpu_hours"))
                if gpu_hours is None:
                    reason_parts.append(f"Missing gpu_hours for {stage_id}")
                else:
                    gpu_hours_values.append(gpu_hours)

            if reason_parts:
                omission_reason = "; ".join(reason_parts)
                rows.append(
                    {
                        "scale": scale_name,
                        "point_kind": point_kind,
                        "cost_scope": point_spec.cost_scope,
                        "gpu_hours": None,
                        "loss_mean": None,
                        "paper_run_id": scale_spec.main_paper_run_id,
                        "quality_stage_id": point_spec.quality_stage_id,
                        "cost_stage_ids": "|".join(point_spec.cost_stage_ids),
                        "cost_policy": scale_spec.figure1_cost_policy,
                        "is_available": False,
                        "omission_reason": omission_reason,
                    }
                )
                omissions.append(
                    {
                        "figure": "figure1",
                        "scale": scale_name,
                        "point_kind": point_kind,
                        "reason": omission_reason,
                    }
                )
                continue

            scale_has_available = True
            rows.append(
                {
                    "scale": scale_name,
                    "point_kind": point_kind,
                    "cost_scope": point_spec.cost_scope,
                    "gpu_hours": float(sum(gpu_hours_values)),
                    "loss_mean": float(loss_mean),
                    "paper_run_id": scale_spec.main_paper_run_id,
                    "quality_stage_id": point_spec.quality_stage_id,
                    "cost_stage_ids": "|".join(point_spec.cost_stage_ids),
                    "cost_policy": scale_spec.figure1_cost_policy,
                    "is_available": True,
                    "omission_reason": "",
                }
            )

        if scale_has_available:
            available_scales.append(scale_name)
        elif strict and scale_spec.requires_strict("figure1"):
            raise ValueError(f"Figure 1 has no available rows for required scale {scale_name}.")
        else:
            skipped_scales.append(scale_name)

    rows.sort(
        key=lambda row: (
            row.get("scale", ""),
            FIGURE1_POINT_ORDER.index(row.get("point_kind", "scratch"))
            if row.get("point_kind", "scratch") in FIGURE1_POINT_ORDER
            else len(FIGURE1_POINT_ORDER),
        )
    )
    return rows, {
        "enabled_scales": [scale for scale, cfg in spec.scales.items() if cfg.is_enabled("figure1")],
        "required_scales_in_strict": [
            scale for scale, cfg in spec.scales.items() if cfg.requires_strict("figure1")
        ],
        "available_scales": _stable_unique(available_scales),
        "skipped_scales": _stable_unique(skipped_scales),
        "omissions": omissions,
        "sources": _stable_unique(sources),
    }


def _prepare_figure2_main_comparison(
    *,
    repo_root: Path,
    spec: FigureSetSpec,
    strict: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    available_scales: list[str] = []
    skipped_scales: list[str] = []
    omissions: list[dict[str, str]] = []
    sources: list[str] = []

    for scale_name, scale_spec in spec.scales.items():
        if not scale_spec.is_enabled("figure2"):
            continue
        inventory_path = scale_spec.resolve_main_report_root(repo_root) / "tables" / "run_inventory.csv"
        sources.append(str(inventory_path))
        if not inventory_path.exists():
            reason = "Missing canonical run_inventory.csv."
            for condition in MAIN_CONDITION_ORDER:
                rows.append(
                    {
                        "scale": scale_name,
                        "condition": condition,
                        "stage_id": scale_spec.stage_ids.get(condition, ""),
                        "loss_mean": None,
                        "gpu_hours": None,
                        "status": "missing",
                        "paper_run_id": scale_spec.main_paper_run_id,
                        "omission_reason": reason,
                    }
                )
            skipped_scales.append(scale_name)
            omissions.append({"figure": "figure2", "scale": scale_name, "reason": reason})
            if strict and scale_spec.requires_strict("figure2"):
                raise FileNotFoundError(
                    f"Figure 2 requires {inventory_path} for scale {scale_name} in strict mode."
                )
            continue

        inventory = _inventory_by_stage(_load_csv(inventory_path))
        scale_has_available = False
        for condition in MAIN_CONDITION_ORDER:
            stage_id = scale_spec.stage_ids[condition]
            row = inventory.get(stage_id)
            loss_mean = _to_float((row or {}).get("loss_mean"))
            gpu_hours = _to_float((row or {}).get("gpu_hours"))
            status = str((row or {}).get("status", "missing") or "missing")
            omission_reason = ""
            if row is None:
                omission_reason = f"Missing stage {stage_id}"
                status = "missing"
            elif loss_mean is None:
                omission_reason = f"Missing loss_mean for {stage_id}"
                status = "missing"
            else:
                scale_has_available = True
            rows.append(
                {
                    "scale": scale_name,
                    "condition": condition,
                    "stage_id": stage_id,
                    "loss_mean": loss_mean,
                    "gpu_hours": gpu_hours,
                    "status": status,
                    "paper_run_id": scale_spec.main_paper_run_id,
                    "omission_reason": omission_reason,
                }
            )
            if omission_reason:
                omissions.append(
                    {
                        "figure": "figure2",
                        "scale": scale_name,
                        "condition": condition,
                        "reason": omission_reason,
                    }
                )
        if scale_has_available:
            available_scales.append(scale_name)
        elif strict and scale_spec.requires_strict("figure2"):
            raise ValueError(f"Figure 2 has no available rows for required scale {scale_name}.")
        else:
            skipped_scales.append(scale_name)

    rows.sort(
        key=lambda row: (
            row.get("scale", ""),
            MAIN_CONDITION_ORDER.index(row.get("condition", "S3"))
            if row.get("condition", "S3") in MAIN_CONDITION_ORDER
            else len(MAIN_CONDITION_ORDER),
        )
    )
    return rows, {
        "enabled_scales": [scale for scale, cfg in spec.scales.items() if cfg.is_enabled("figure2")],
        "required_scales_in_strict": [
            scale for scale, cfg in spec.scales.items() if cfg.requires_strict("figure2")
        ],
        "available_scales": _stable_unique(available_scales),
        "skipped_scales": _stable_unique(skipped_scales),
        "omissions": omissions,
        "sources": _stable_unique(sources),
    }


def _prepare_figure3_continuation(
    *,
    repo_root: Path,
    spec: FigureSetSpec,
    strict: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    available_scales: list[str] = []
    skipped_scales: list[str] = []
    omissions: list[dict[str, str]] = []
    sources: list[str] = []

    for scale_name, scale_spec in spec.scales.items():
        if not scale_spec.is_enabled("figure3"):
            continue
        report_root = scale_spec.resolve_ablation_report_root(repo_root)
        inventory_path = scale_spec.resolve_main_report_root(repo_root) / "tables" / "run_inventory.csv"
        if report_root is None:
            reason = "No ablation report root configured."
            skipped_scales.append(scale_name)
            omissions.append({"figure": "figure3", "scale": scale_name, "reason": reason})
            if strict and scale_spec.requires_strict("figure3"):
                raise FileNotFoundError(f"Figure 3 requires ablation_report_root for {scale_name}.")
            continue
        frontier_path = report_root / "frontier.csv"
        sources.extend([str(frontier_path), str(inventory_path)])
        if not frontier_path.exists() or not inventory_path.exists():
            reason = "Missing frontier.csv or canonical run_inventory.csv."
            skipped_scales.append(scale_name)
            omissions.append({"figure": "figure3", "scale": scale_name, "reason": reason})
            if strict and scale_spec.requires_strict("figure3"):
                raise FileNotFoundError(
                    f"Figure 3 requires {frontier_path} and {inventory_path} for scale {scale_name}."
                )
            continue

        inventory = _inventory_by_stage(_load_csv(inventory_path))
        s2_loss = _to_float(inventory[scale_spec.stage_ids["S2"]]["loss_mean"]) if scale_spec.stage_ids["S2"] in inventory else None
        s3_loss = _to_float(inventory[scale_spec.stage_ids["S3"]]["loss_mean"]) if scale_spec.stage_ids["S3"] in inventory else None
        if s2_loss is None or s3_loss is None:
            reason = "Missing canonical S2/S3 Books32K losses."
            skipped_scales.append(scale_name)
            omissions.append({"figure": "figure3", "scale": scale_name, "reason": reason})
            if strict and scale_spec.requires_strict("figure3"):
                raise ValueError(f"Figure 3 missing canonical reference losses for {scale_name}.")
            continue

        frontier_rows = _load_csv(frontier_path)
        scale_rows: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, int]] = set()
        for row in frontier_rows:
            mode = str(row.get("mode", "")).strip()
            if mode not in FIGURE3_MODE_ORDER:
                continue
            loss_ce = _to_float(row.get("loss_ce_mean"))
            snapshot_dir = str(row.get("snapshot_dir", "")).strip()
            if loss_ce is None and snapshot_dir:
                resolved_snapshot_dir = _resolve_snapshot_dir(
                    repo_root=repo_root,
                    report_root=report_root,
                    raw_snapshot_dir=snapshot_dir,
                )
                if resolved_snapshot_dir is not None:
                    snapshot_manifest = resolved_snapshot_dir / "eval_manifest_snapshot.json"
                    if snapshot_manifest.exists():
                        payload = _load_json(snapshot_manifest)
                        metrics = payload.get("metrics", {})
                        if isinstance(metrics, dict):
                            loss_ce = _to_float(metrics.get("loss_ce_mean"))
                            if loss_ce is not None:
                                sources.append(str(snapshot_manifest))
            if loss_ce is None:
                continue

            checkpoint_step = _to_int(row.get("checkpoint_step")) or 0
            scale_rows.append(
                {
                    "scale": scale_name,
                    "mode": mode,
                    "extra_steps": _to_int(row.get("extra_steps")) or 0,
                    "checkpoint_step": checkpoint_step,
                    "loss_ce_mean": float(loss_ce),
                    "matched_target": _to_bool(row.get("matched_target")),
                    "reference_s2_loss": float(s2_loss),
                    "reference_s3_loss": float(s3_loss),
                    "stage_id": str(row.get("stage_id", "")),
                    "run_id": str(row.get("run_id", "")),
                    "status": str(row.get("status", "")),
                    "snapshot_dir": snapshot_dir,
                }
            )
            seen_keys.add((mode, checkpoint_step))

        for mode in FIGURE3_MODE_ORDER:
            condition = "S2" if mode == "iso_quality" else "S3"
            source_stage_id = scale_spec.stage_ids[condition]
            source_row = inventory.get(source_stage_id)
            source_run_id = str((source_row or {}).get("run_id", "")).strip()
            source_loss = s2_loss if condition == "S2" else s3_loss
            sources.append(str(report_root / "eval_snapshots" / mode))

            snapshot_rows = _scan_snapshot_rows_for_mode(
                report_root=report_root,
                mode=mode,
                scale=scale_name,
                source_stage_id=source_stage_id,
                source_run_id=source_run_id,
                source_loss=source_loss,
                reference_s2_loss=s2_loss,
                reference_s3_loss=s3_loss,
            )
            for row in snapshot_rows:
                key = (str(row["mode"]), int(row["checkpoint_step"]))
                if key in seen_keys:
                    continue
                scale_rows.append(row)
                seen_keys.add(key)

        scale_rows.sort(
            key=lambda row: (
                FIGURE3_MODE_ORDER.index(row["mode"]),
                int(row["extra_steps"]),
                int(row["checkpoint_step"]),
            )
        )
        if scale_rows:
            rows.extend(scale_rows)
            available_scales.append(scale_name)
        elif strict and scale_spec.requires_strict("figure3"):
            raise ValueError(
                f"Figure 3 had no numeric continuation rows for required scale {scale_name}."
            )
        else:
            skipped_scales.append(scale_name)
            omissions.append(
                {
                    "figure": "figure3",
                    "scale": scale_name,
                    "reason": "No numeric continuation rows were available after snapshot backfill.",
                }
            )

    return rows, {
        "enabled_scales": [scale for scale, cfg in spec.scales.items() if cfg.is_enabled("figure3")],
        "required_scales_in_strict": [
            scale for scale, cfg in spec.scales.items() if cfg.requires_strict("figure3")
        ],
        "available_scales": _stable_unique(available_scales),
        "skipped_scales": _stable_unique(skipped_scales),
        "omissions": omissions,
        "sources": _stable_unique(sources),
    }


def _prepare_figure4_training_curves(
    *,
    repo_root: Path,
    exp_dir: Path,
    spec: FigureSetSpec,
    strict: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    available_scales: list[str] = []
    skipped_scales: list[str] = []
    omissions: list[dict[str, str]] = []
    sources: list[str] = []

    for scale_name, scale_spec in spec.scales.items():
        if not scale_spec.is_enabled("figure4"):
            continue
        inventory_path = scale_spec.resolve_main_report_root(repo_root) / "tables" / "run_inventory.csv"
        sources.append(str(inventory_path))
        if not inventory_path.exists():
            reason = "Missing canonical run_inventory.csv."
            skipped_scales.append(scale_name)
            omissions.append({"figure": "figure4", "scale": scale_name, "reason": reason})
            if strict and scale_spec.requires_strict("figure4"):
                raise FileNotFoundError(
                    f"Figure 4 requires {inventory_path} for scale {scale_name} in strict mode."
                )
            continue

        inventory = _inventory_by_stage(_load_csv(inventory_path))
        scale_has_rows = False
        for condition in MAIN_CONDITION_ORDER:
            stage_id = scale_spec.stage_ids[condition]
            run_row = inventory.get(stage_id)
            if run_row is None:
                omissions.append(
                    {
                        "figure": "figure4",
                        "scale": scale_name,
                        "condition": condition,
                        "reason": f"Missing stage {stage_id}",
                    }
                )
                continue
            run_id = str(run_row.get("run_id", "")).strip()
            if not run_id:
                omissions.append(
                    {
                        "figure": "figure4",
                        "scale": scale_name,
                        "condition": condition,
                        "reason": f"Missing run_id for stage {stage_id}",
                    }
                )
                continue
            metrics_path = exp_dir / scale_spec.main_paper_run_id / stage_id / run_id / "metrics.jsonl"
            sources.append(str(metrics_path))
            series = _load_training_curve(metrics_path)
            if not series:
                omissions.append(
                    {
                        "figure": "figure4",
                        "scale": scale_name,
                        "condition": condition,
                        "reason": f"No train metrics found in {metrics_path}",
                    }
                )
                continue
            smooth = _centered_rolling_mean(
                [point["loss_ce_raw"] for point in series],
                window=FIGURE4_SMOOTH_WINDOW,
            )
            for point, smooth_value in zip(series, smooth):
                rows.append(
                    {
                        "scale": scale_name,
                        "condition": condition,
                        "stage_id": stage_id,
                        "run_id": run_id,
                        "step": int(point["step"]),
                        "loss_ce_raw": float(point["loss_ce_raw"]),
                        "loss_ce_smooth": float(smooth_value),
                    }
                )
            scale_has_rows = True
        if scale_has_rows:
            available_scales.append(scale_name)
        elif strict and scale_spec.requires_strict("figure4"):
            raise ValueError(f"Figure 4 has no training-curve rows for required scale {scale_name}.")
        else:
            skipped_scales.append(scale_name)

    rows.sort(
        key=lambda row: (
            row.get("scale", ""),
            MAIN_CONDITION_ORDER.index(row.get("condition", "S3"))
            if row.get("condition", "S3") in MAIN_CONDITION_ORDER
            else len(MAIN_CONDITION_ORDER),
            int(row.get("step", 0)),
        )
    )
    return rows, {
        "enabled_scales": [scale for scale, cfg in spec.scales.items() if cfg.is_enabled("figure4")],
        "required_scales_in_strict": [
            scale for scale, cfg in spec.scales.items() if cfg.requires_strict("figure4")
        ],
        "available_scales": _stable_unique(available_scales),
        "skipped_scales": _stable_unique(skipped_scales),
        "omissions": omissions,
        "sources": _stable_unique(sources),
    }


def _resolve_snapshot_dir(
    *,
    repo_root: Path,
    report_root: Path,
    raw_snapshot_dir: str,
) -> Path | None:
    candidates: list[Path] = []
    raw = raw_snapshot_dir.strip()
    if not raw:
        return None
    direct = Path(raw)
    candidates.append(direct)
    if not direct.is_absolute():
        candidates.append((report_root / direct).resolve())
    marker = "/reports/paper/"
    idx = raw.find(marker)
    if idx >= 0:
        rel = raw[idx + 1 :]
        candidates.append((repo_root / rel).resolve())
    marker = "eval_snapshots/"
    idx = raw.find(marker)
    if idx >= 0:
        rel = raw[idx:]
        candidates.append((report_root / rel).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _scan_snapshot_rows_for_mode(
    *,
    report_root: Path,
    mode: str,
    scale: str,
    source_stage_id: str,
    source_run_id: str,
    source_loss: float,
    reference_s2_loss: float,
    reference_s3_loss: float,
) -> list[dict[str, Any]]:
    mode_root = report_root / "eval_snapshots" / mode
    if not mode_root.exists():
        return []

    snapshot_payloads: list[dict[str, Any]] = []
    checkpoint_steps: list[int] = []
    for snapshot_manifest in sorted(mode_root.rglob("eval_manifest_snapshot.json")):
        checkpoint_dir = snapshot_manifest.parent
        checkpoint_step = _parse_checkpoint_step_from_dir(checkpoint_dir)
        if checkpoint_step is None:
            continue
        payload = _load_json(snapshot_manifest)
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        loss_ce = _to_float(metrics.get("loss_ce_mean"))
        if loss_ce is None:
            continue
        checkpoint_steps.append(checkpoint_step)
        snapshot_payloads.append(
            {
                "checkpoint_step": checkpoint_step,
                "loss_ce_mean": float(loss_ce),
                "snapshot_dir": str(checkpoint_dir),
                "run_id": checkpoint_dir.parent.name,
                "status": str(payload.get("status", "")),
            }
        )

    if not snapshot_payloads:
        return []

    source_checkpoint_step = min(checkpoint_steps) - 1
    rows: list[dict[str, Any]] = [
        {
            "scale": scale,
            "mode": mode,
            "extra_steps": 0,
            "checkpoint_step": source_checkpoint_step,
            "loss_ce_mean": float(source_loss),
            "matched_target": bool(source_loss <= reference_s3_loss),
            "reference_s2_loss": float(reference_s2_loss),
            "reference_s3_loss": float(reference_s3_loss),
            "stage_id": source_stage_id,
            "run_id": source_run_id,
            "status": "succeeded",
            "snapshot_dir": "",
        }
    ]

    for payload in sorted(snapshot_payloads, key=lambda row: int(row["checkpoint_step"])):
        checkpoint_step = int(payload["checkpoint_step"])
        loss_ce = float(payload["loss_ce_mean"])
        rows.append(
            {
                "scale": scale,
                "mode": mode,
                "extra_steps": max(0, checkpoint_step - source_checkpoint_step),
                "checkpoint_step": checkpoint_step,
                "loss_ce_mean": loss_ce,
                "matched_target": bool(loss_ce <= reference_s3_loss),
                "reference_s2_loss": float(reference_s2_loss),
                "reference_s3_loss": float(reference_s3_loss),
                "stage_id": source_stage_id,
                "run_id": str(payload["run_id"]),
                "status": str(payload["status"]),
                "snapshot_dir": str(payload["snapshot_dir"]),
            }
        )
    return rows


def _parse_checkpoint_step_from_dir(path: Path) -> int | None:
    name = path.name
    if name.startswith("step_"):
        return _to_int(name.split("_", 1)[1])
    if name.isdigit():
        return int(name)
    return None


def _load_training_curve(metrics_path: Path) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for row in _load_jsonl(metrics_path):
        step = _to_int(row.get("step"))
        loss_ce = _to_float(row.get("loss_ce"))
        if step is None or loss_ce is None:
            continue
        out.append({"step": float(step), "loss_ce_raw": float(loss_ce)})
    out.sort(key=lambda point: int(point["step"]))
    return out


def _centered_rolling_mean(values: Sequence[float], *, window: int) -> list[float]:
    if window <= 1 or not values:
        return [float(x) for x in values]
    half = window // 2
    out: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        segment = values[start:end]
        out.append(float(sum(segment) / len(segment)))
    return out


def _validate_plot_manifest_for_render(plot_manifest: dict[str, Any], *, strict: bool) -> None:
    figures = plot_manifest.get("figures", {})
    if not isinstance(figures, dict):
        raise ValueError("plot_data_manifest.json missing figures payload")
    if not strict:
        return
    for figure_id, meta in figures.items():
        if not isinstance(meta, dict):
            continue
        required_scales = [str(x) for x in meta.get("required_scales_in_strict", [])]
        available_scales = {str(x) for x in meta.get("available_scales", [])}
        missing = [scale for scale in required_scales if scale not in available_scales]
        if missing:
            raise ValueError(
                f"{figure_id} is missing required strict scales: {', '.join(sorted(missing))}"
            )


def _apply_publication_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#d6d9de",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _condition_color(condition: str) -> str:
    palette = {
        "S0": "#5B6770",
        "S1": "#C06B2C",
        "S2": "#11806A",
        "S3": "#B5383B",
    }
    return palette.get(condition, "#333333")


def _marker_for_point_kind(point_kind: str) -> str:
    markers = {
        "warmstart_marginal": "o",
        "warmstart_full": "^",
        "scratch": "s",
    }
    return markers.get(point_kind, "o")


def _point_display_label(point_kind: str) -> str:
    labels = {
        "warmstart_marginal": "warm-start (marg.)",
        "warmstart_full": "warm-start (full)",
        "scratch": "scratch",
    }
    return labels.get(point_kind, point_kind.replace("_", " "))


def _save_figure(fig: Any, output_stem: Path, formats: Sequence[str]) -> list[str]:
    output_files: list[str] = []
    for fmt in formats:
        path = output_stem.with_suffix(f".{fmt}")
        fig.savefig(path, bbox_inches="tight")
        output_files.append(str(path))
    return output_files


def _plot_figure1_cost_quality(
    *,
    plt: Any,
    rows: list[dict[str, str]],
    output_stem: Path,
    formats: Sequence[str],
    plot_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    plotted_rows = [row for row in rows if _to_bool(row.get("is_available"))]
    if not plotted_rows:
        return None

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    plotted_scales: list[str] = []
    annotations: list[str] = []
    for row in plotted_rows:
        point_kind = str(row["point_kind"])
        scale = str(row["scale"])
        gpu_hours = float(row["gpu_hours"])
        loss_mean = float(row["loss_mean"])
        color = _condition_color("S2" if point_kind.startswith("warmstart") else "S3")
        marker = _marker_for_point_kind(point_kind)
        ax.scatter(
            gpu_hours,
            loss_mean,
            s=120,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        offset_y = {
            "warmstart_marginal": -10,
            "warmstart_full": 8,
            "scratch": 8,
        }.get(point_kind, 6)
        label = f"{scale} {_point_display_label(point_kind)}"
        ax.annotate(
            label,
            (gpu_hours, loss_mean),
            xytext=(7, offset_y),
            textcoords="offset points",
            fontsize=9,
            color=color,
        )
        plotted_scales.append(scale)
        annotations.append(label)

    ax.set_xlabel("GPU-hours")
    ax.set_ylabel("Books32K validation loss (lower is better)")
    ax.set_axisbelow(True)

    handles = []
    labels = []
    for point_kind in FIGURE1_POINT_ORDER:
        if not any(str(row["point_kind"]) == point_kind for row in plotted_rows):
            continue
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=_marker_for_point_kind(point_kind),
                color="none",
                markerfacecolor="#444444",
                markeredgecolor="white",
                markeredgewidth=0.9,
                markersize=10,
                linewidth=0,
            )
        )
        labels.append(_point_display_label(point_kind))
    if handles:
        ax.legend(handles, labels, loc="upper right")

    fig.tight_layout()
    output_files = _save_figure(fig, output_stem, formats)
    plt.close(fig)
    omitted_rows = [
        row
        for row in rows
        if not _to_bool(row.get("is_available")) and str(row.get("point_kind", "")).strip()
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "figure_id": "figure1",
        "output_files": output_files,
        "plotted_rows": len(plotted_rows),
        "plotted_scales": _stable_unique(plotted_scales),
        "omitted_rows": len(omitted_rows),
        "omissions": plot_manifest.get("omissions", []),
        "plot_data_meta": plot_manifest,
    }


def _plot_figure2_main_comparison(
    *,
    plt: Any,
    rows: list[dict[str, str]],
    output_stem: Path,
    formats: Sequence[str],
    plot_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    plotted_rows = [row for row in rows if str(row.get("status", "")) == "succeeded" and _to_float(row.get("loss_mean")) is not None]
    if not plotted_rows:
        return None

    scales = _stable_unique(row["scale"] for row in plotted_rows)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    x = np.arange(len(scales))
    width = 0.18

    for idx, condition in enumerate(MAIN_CONDITION_ORDER):
        condition_rows = {row["scale"]: row for row in plotted_rows if row["condition"] == condition}
        heights = [_to_float((condition_rows.get(scale) or {}).get("loss_mean")) or np.nan for scale in scales]
        positions = x + (idx - 1.5) * width
        bars = ax.bar(
            positions,
            heights,
            width=width,
            color=_condition_color(condition),
            label=condition,
            zorder=3,
        )
        for bar, value in zip(bars, heights):
            if np.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.05,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=_condition_color(condition),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.set_xlabel("Scale")
    ax.set_ylabel("Books32K validation loss (lower is better)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.05))
    ax.set_axisbelow(True)
    fig.tight_layout()
    output_files = _save_figure(fig, output_stem, formats)
    plt.close(fig)
    omitted_rows = [row for row in rows if str(row.get("status", "")) != "succeeded"]
    return {
        "schema_version": SCHEMA_VERSION,
        "figure_id": "figure2",
        "output_files": output_files,
        "plotted_rows": len(plotted_rows),
        "plotted_scales": scales,
        "omitted_rows": len(omitted_rows),
        "omissions": plot_manifest.get("omissions", []),
        "plot_data_meta": plot_manifest,
    }


def _plot_figure3_continuation(
    *,
    plt: Any,
    rows: list[dict[str, str]],
    output_stem: Path,
    formats: Sequence[str],
    plot_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    plotted_rows = [row for row in rows if _to_float(row.get("loss_ce_mean")) is not None]
    if not plotted_rows:
        return None

    scales = _stable_unique(row["scale"] for row in plotted_rows)
    fig, axes = plt.subplots(
        1,
        len(scales),
        figsize=(8.4 if len(scales) == 1 else 8.4 * len(scales), 5.0),
        sharey=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    plotted_modes: list[str] = []
    for ax, scale in zip(axes, scales):
        scale_rows = [row for row in plotted_rows if row["scale"] == scale]
        reference_s2 = _to_float(scale_rows[0].get("reference_s2_loss")) or 0.0
        reference_s3 = _to_float(scale_rows[0].get("reference_s3_loss")) or 0.0
        xmax = max(int(_to_int(row.get("extra_steps")) or 0) for row in scale_rows)
        ax.axhline(reference_s2, color=_condition_color("S2"), linestyle="--", linewidth=1.4)
        ax.axhline(reference_s3, color=_condition_color("S3"), linestyle="--", linewidth=1.4)
        label_x_s2 = max(0.0, xmax * 0.86)
        label_x_s3 = max(0.0, xmax * 0.66)
        ax.text(
            label_x_s2,
            reference_s2 + 0.004,
            "canonical S2",
            color=_condition_color("S2"),
            ha="left",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            label_x_s3,
            reference_s3 - 0.006,
            "canonical S3",
            color=_condition_color("S3"),
            ha="left",
            va="top",
            fontsize=8,
        )
        for mode in FIGURE3_MODE_ORDER:
            mode_rows = [row for row in scale_rows if row["mode"] == mode]
            if not mode_rows:
                continue
            mode_rows.sort(key=lambda row: (int(_to_int(row.get("extra_steps")) or 0), int(_to_int(row.get("checkpoint_step")) or 0)))
            xs = [int(_to_int(row.get("extra_steps")) or 0) for row in mode_rows]
            ys = [float(row["loss_ce_mean"]) for row in mode_rows]
            color = _condition_color("S2" if mode == "iso_quality" else "S3")
            label = "S2 continuation" if mode == "iso_quality" else "S3 equal-token continuation"
            ax.plot(xs, ys, color=color, linewidth=2.4, marker="o", markersize=4, label=label)
            plotted_modes.append(mode)
        ax.set_xlabel("Continuation steps")
        ax.set_title(scale)
        ax.set_axisbelow(True)
        ax.legend(loc="center", bbox_to_anchor=(0.47, 0.54))
    axes[0].set_ylabel("Books32K validation loss (lower is better)")
    fig.tight_layout()
    output_files = _save_figure(fig, output_stem, formats)
    plt.close(fig)
    return {
        "schema_version": SCHEMA_VERSION,
        "figure_id": "figure3",
        "output_files": output_files,
        "plotted_rows": len(plotted_rows),
        "plotted_scales": scales,
        "plotted_modes": _stable_unique(plotted_modes),
        "omitted_rows": 0,
        "omissions": plot_manifest.get("omissions", []),
        "plot_data_meta": plot_manifest,
    }


def _plot_figure4_training_curves(
    *,
    plt: Any,
    rows: list[dict[str, str]],
    output_stem: Path,
    formats: Sequence[str],
    plot_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    plotted_rows = [row for row in rows if _to_float(row.get("loss_ce_smooth")) is not None]
    if not plotted_rows:
        return None

    scales = _stable_unique(row["scale"] for row in plotted_rows)
    fig, axes = plt.subplots(
        1,
        len(scales),
        figsize=(8.4 if len(scales) == 1 else 8.4 * len(scales), 5.2),
        sharey=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, scale in zip(axes, scales):
        scale_rows = [row for row in plotted_rows if row["scale"] == scale]
        for condition in MAIN_CONDITION_ORDER:
            condition_rows = [row for row in scale_rows if row["condition"] == condition]
            if not condition_rows:
                continue
            condition_rows.sort(key=lambda row: int(_to_int(row.get("step")) or 0))
            xs = [int(_to_int(row.get("step")) or 0) for row in condition_rows]
            raw = [float(row["loss_ce_raw"]) for row in condition_rows]
            smooth = [float(row["loss_ce_smooth"]) for row in condition_rows]
            color = _condition_color(condition)
            ax.plot(xs, raw, color=color, linewidth=0.9, alpha=0.18)
            ax.plot(xs, smooth, color=color, linewidth=2.4, label=condition)
        ax.set_xlabel("32K extension step")
        ax.set_title(scale)
        ax.set_axisbelow(True)
        ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.05))
    axes[0].set_ylabel("Training loss_ce")
    fig.tight_layout()
    output_files = _save_figure(fig, output_stem, formats)
    plt.close(fig)
    return {
        "schema_version": SCHEMA_VERSION,
        "figure_id": "figure4",
        "output_files": output_files,
        "plotted_rows": len(plotted_rows),
        "plotted_scales": scales,
        "omitted_rows": 0,
        "omissions": plot_manifest.get("omissions", []),
        "plot_data_meta": plot_manifest,
    }
