"""RULER-style metric extraction helpers.

Current implementation derives RULER-aligned retrieval accuracy proxies from the
existing eval CSV outputs (NIAH rows). This keeps manifests stable while we add
full benchmark harnesses.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _safe_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def compute_ruler_metrics_from_eval_csv(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    values: list[float] = []
    by_len: dict[int, list[float]] = {}

    for row in rows:
        if str(row.get("record_type", "")) != "niah_proxy":
            continue
        if str(row.get("status", "")) != "ok":
            continue

        val = _safe_float(row.get("niah_accuracy") or row.get("accuracy"))
        if val is None:
            continue
        values.append(val)

        try:
            ctx = int(str(row.get("context_length", "")).strip())
        except ValueError:
            continue
        by_len.setdefault(ctx, []).append(val)

    if not values:
        return {}

    out: dict[str, float] = {
        "ruler_accuracy_mean": float(sum(values) / len(values)),
    }
    for ctx, ctx_vals in sorted(by_len.items()):
        out[f"ruler_by_length_{ctx}"] = float(sum(ctx_vals) / len(ctx_vals))
    return out


def merge_metrics(existing: dict[str, Any], extra: dict[str, float]) -> dict[str, Any]:
    merged = dict(existing)
    merged.update(extra)
    return merged
