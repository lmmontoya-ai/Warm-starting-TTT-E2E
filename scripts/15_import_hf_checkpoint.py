#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from ttt.dataloader.lm_dataset import _load_token_stream
from ttt.infra import Phase1Checkpointer
from ttt.model import TokenStatsState
from ttt.research.lineage import file_sha256
from ttt.research.types import utc_now_iso


@dataclass(frozen=True)
class ImportModelDefaults:
    hf_model_id: str
    import_exp_name: str


DEFAULTS: dict[str, ImportModelDefaults] = {
    "qwen2_5_0_5b": ImportModelDefaults(
        hf_model_id="Qwen/Qwen2.5-0.5B",
        import_exp_name="import-qwen05-fa-base",
    ),
    "smollm2_360m": ImportModelDefaults(
        hf_model_id="HuggingFaceTB/SmolLM2-360M",
        import_exp_name="import-smol360-fa-base",
    ),
}


def _slug(raw: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", raw.strip().lower()).strip("-")
    return slug or "external"


def _load_profile(profile_root: Path, model_key: str) -> tuple[Path, dict]:
    profile_path = profile_root / model_key / "model_profile.json"
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Missing profile for {model_key}: {profile_path}. "
            "Run scripts/07_prepare_external_models.py first or provide your profile."
        )
    payload = json.loads(profile_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Profile must be a JSON object: {profile_path}")
    return profile_path, payload


def _zipf_prior_state(vocab_size: int, prior_tokens: int) -> tuple[TokenStatsState, int]:
    n_tokens = max(1, min(prior_tokens, vocab_size))
    counts: dict[int, float] = {}
    total = 0.0
    for rank in range(1, n_tokens + 1):
        token_id = rank - 1
        count = 1.0 / float(rank)
        counts[token_id] = count
        total += count
    return TokenStatsState(token_counts=counts, total_count=total), n_tokens


def _dataset_calibration_state(
    *,
    dataset_root: Path,
    split: str,
    vocab_size: int,
    max_tokens: int,
) -> tuple[TokenStatsState, int]:
    tokens = _load_token_stream(str(dataset_root), split=split)
    counts: dict[int, float] = {}
    used = 0
    for token in tokens:
        if used >= max_tokens:
            break
        if token < 0 or token >= vocab_size:
            continue
        counts[token] = counts.get(token, 0.0) + 1.0
        used += 1

    if used <= 0:
        raise ValueError(
            f"No usable calibration tokens in range [0,{vocab_size}) at "
            f"{dataset_root}/{split}"
        )

    return TokenStatsState(token_counts=counts, total_count=float(used)), used


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an external import checkpoint seed for adapter warm-start runs. "
            "This is the required root checkpoint for imported FA stages."
        )
    )
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--hf-model-id", default="")
    parser.add_argument("--import-exp-name", default="")
    parser.add_argument("--profile-root", type=Path, default=Path("./artifacts/external_models"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--exp-folder", default="external_phase1")
    parser.add_argument("--step", type=int, default=0)

    parser.add_argument(
        "--seed-mode",
        choices=["zipf", "dataset"],
        default="zipf",
        help="Initialization mode for token statistics payload.",
    )
    parser.add_argument("--prior-tokens", type=int, default=4096)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-tokens", type=int, default=200000)

    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.step < 0:
        raise ValueError("--step must be >= 0")
    if args.prior_tokens <= 0:
        raise ValueError("--prior-tokens must be > 0")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")

    defaults = DEFAULTS.get(args.model_key)
    hf_model_id = args.hf_model_id or (defaults.hf_model_id if defaults else "")
    import_exp_name = args.import_exp_name or (
        defaults.import_exp_name if defaults else f"import-{_slug(args.model_key)}-fa-base"
    )

    if not hf_model_id:
        raise ValueError(
            "Missing --hf-model-id for unknown model key. "
            "Provide a Hugging Face model id."
        )

    profile_root = args.profile_root.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    checkpoint_dir = checkpoint_root / args.exp_folder / import_exp_name
    latest_path = checkpoint_dir / "latest.json"

    if latest_path.exists() and not args.force:
        raise FileExistsError(
            f"Checkpoint already exists: {latest_path}. Use --force to overwrite."
        )

    profile_path, profile = _load_profile(profile_root=profile_root, model_key=args.model_key)
    vocab_size = int(profile.get("vocab_size", 0))
    if vocab_size <= 0:
        raise ValueError(f"Invalid vocab_size in profile: {profile_path}")

    seed_source = ""
    if args.seed_mode == "zipf":
        state, tokens_used = _zipf_prior_state(vocab_size=vocab_size, prior_tokens=args.prior_tokens)
    else:
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required when --seed-mode=dataset")
        dataset_root = args.dataset_root.expanduser().resolve()
        state, tokens_used = _dataset_calibration_state(
            dataset_root=dataset_root,
            split=args.split,
            vocab_size=vocab_size,
            max_tokens=args.max_tokens,
        )
        seed_source = str(dataset_root)

    checkpointer = Phase1Checkpointer(checkpoint_dir=checkpoint_dir)
    payload = {
        "exp_name": import_exp_name,
        "exp_folder": args.exp_folder,
        "elapsed_seconds": 0.0,
        "loss": 0.0,
        "gradient_norm": 0.0,
        "model_state": state.to_jsonable(),
        "init_source": "external_hf",
        "external_model_id": hf_model_id,
        "adapter_recipe": "import_seed",
        "seed_mode": args.seed_mode,
        "seed_source": seed_source,
        "seed_tokens_used": tokens_used,
        "created_at_unix": int(time.time()),
        "profile_revision": str(profile.get("source_revision", "")),
        "profile_sha256": str(profile.get("source_config_sha256", "")),
    }
    ckpt_path = checkpointer.save(step=args.step, payload=payload)
    ckpt_sha = file_sha256(ckpt_path)

    report = {
        "schema_version": "1.0",
        "created_at_utc": utc_now_iso(),
        "model_key": args.model_key,
        "hf_model_id": hf_model_id,
        "import_exp_name": import_exp_name,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "latest_pointer": str(latest_path),
        "profile_path": str(profile_path),
        "profile_revision": str(profile.get("source_revision", "")),
        "profile_config_sha256": str(profile.get("source_config_sha256", "")),
        "vocab_size": vocab_size,
        "seed_mode": args.seed_mode,
        "seed_tokens_used": tokens_used,
        "seed_source": seed_source,
    }

    report_out = args.report_out
    if report_out is None:
        report_out = checkpoint_dir / "import_report.json"
    else:
        report_out = report_out.expanduser().resolve()

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"Seeded import checkpoint: {ckpt_path}")
    print(f"Updated latest pointer: {latest_path}")
    print(f"Wrote import report: {report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
