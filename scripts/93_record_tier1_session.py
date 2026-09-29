#!/usr/bin/env python3
"""Record a closed Tier-1 GPU session and enforce the reviewer reserve."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from ttt.research.tier1 import load_tier1_config, sha256_payload, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append a terminated Tier-1 session to the cost ledger."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/research/tier1_eval.yaml")
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument(
        "--provider", required=True, choices=["Prime Intellect", "Vast.ai", "other"]
    )
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--gpu-description", required=True)
    parser.add_argument("--hourly-rate-usd", type=float, required=True)
    parser.add_argument("--duration-hours", type=float, required=True)
    parser.add_argument("--status", required=True, choices=["terminated", "destroyed"])
    parser.add_argument("--stages", required=True)
    parser.add_argument("--spent-before-usd", type=float, required=True)
    parser.add_argument("--authorized-envelope-usd", type=float, default=1000.0)
    parser.add_argument(
        "--ledger", type=Path, default=Path("docs/REVISION_V2_COST_LEDGER.md")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("reports/revision_v2/tier1/cost_sessions")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_tier1_config(args.config)
    budget = config["budget"]
    if args.hourly_rate_usd < 0 or args.duration_hours < 0 or args.spent_before_usd < 0:
        raise ValueError("Cost inputs must be non-negative")
    if not (
        float(budget["authorized_min_usd"])
        <= args.authorized_envelope_usd
        <= float(budget["authorized_max_usd"])
    ):
        raise ValueError("Authorized envelope is outside the project-approved range")
    session_cost = args.hourly_rate_usd * args.duration_hours
    spent_after = args.spent_before_usd + session_cost
    remaining = args.authorized_envelope_usd - spent_after
    reserve = float(budget["reviewer_response_reserve_usd"])
    if remaining < reserve:
        raise RuntimeError(
            f"Session would breach reviewer-response reserve: remaining ${remaining:.2f} < ${reserve:.2f}"
        )
    record = {
        "schema_version": "1.0",
        "paper_run_id": config["paper_run_id"],
        "session_id": args.session_id,
        "date": dt.date.today().isoformat(),
        "provider": args.provider,
        "instance_id": args.instance_id,
        "gpu_description": args.gpu_description,
        "hourly_rate_usd": args.hourly_rate_usd,
        "duration_hours": args.duration_hours,
        "session_cost_usd": session_cost,
        "spent_before_usd": args.spent_before_usd,
        "spent_after_usd": spent_after,
        "authorized_envelope_usd": args.authorized_envelope_usd,
        "reviewer_response_reserve_usd": reserve,
        "remaining_after_usd": remaining,
        "instance_status": args.status,
        "stages": [part.strip() for part in args.stages.split(",") if part.strip()],
    }
    record["record_sha256"] = sha256_payload(record)
    record_path = args.out_dir / f"{args.session_id}.json"
    if record_path.exists():
        raise FileExistsError(f"Session record already exists: {record_path}")
    write_json(record_path, record)

    ledger = args.ledger.expanduser().resolve()
    text = ledger.read_text(encoding="utf-8")
    marker = f"### Tier-1 session `{args.session_id}`"
    if marker in text:
        raise FileExistsError(f"Session already appears in ledger: {args.session_id}")
    block = (
        f"\n\n{marker}\n\n"
        f"- Date: {record['date']}\n"
        f"- Provider / instance: {args.provider} / `{args.instance_id}`\n"
        f"- GPU profile: {args.gpu_description}\n"
        f"- Rate and duration: ${args.hourly_rate_usd:.4f}/hr × {args.duration_hours:.4f} hr\n"
        f"- Session cost: ${session_cost:.2f}\n"
        f"- Stages: {', '.join(record['stages'])}\n"
        f"- Instance status after close-out: {args.status}\n"
        f"- Authorized-envelope remainder: ${remaining:.2f}; reviewer reserve retained: ${reserve:.2f}\n"
        f"- Machine record: `{record_path.as_posix()}` (`{record['record_sha256']}`)\n"
    )
    ledger.write_text(text.rstrip() + block + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
