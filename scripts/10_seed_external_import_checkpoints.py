#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from ttt.dataloader.lm_dataset import _load_token_stream
from ttt.infra import Phase1Checkpointer
from ttt.model import TokenStatsState


@dataclass(frozen=True)
class ImportTarget:
    model_key: str
    hf_model_id: str
    import_exp_name: str


TARGETS: dict[str, ImportTarget] = {
    "qwen2_5_0_5b": ImportTarget(
        model_key="qwen2_5_0_5b",
        hf_model_id="Qwen/Qwen2.5-0.5B",
        import_exp_name="import-qwen05-fa-base",
    ),
    "smollm2_360m": ImportTarget(
        model_key="smollm2_360m",
        hf_model_id="HuggingFaceTB/SmolLM2-360M",
        import_exp_name="import-smol360-fa-base",
    ),
}


def _load_profile(profile_root: Path, model_key: str) -> dict:
    profile_path = profile_root / model_key / "model_profile.json"
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Missing profile for {model_key}: {profile_path}. "
            "Run scripts/07_prepare_external_models.py first."
        )
    payload = json.loads(profile_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Profile must be a JSON object: {profile_path}")
    return payload


def _profile_path(profile_root: Path, model_key: str) -> Path:
    return profile_root / model_key / "model_profile.json"


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


def _calibration_state(
    dataset_root: Path,
    split: str,
    vocab_size: int,
    max_tokens: int,
) -> tuple[TokenStatsState, int]:
    tokens = _load_token_stream(str(dataset_root), split=split)
    counts: dict[int, float] = {}
    n_used = 0
    for token in tokens:
        if n_used >= max_tokens:
            break
        if token < 0 or token >= vocab_size:
            continue
        counts[token] = counts.get(token, 0.0) + 1.0
        n_used += 1

    if n_used <= 0:
        raise ValueError(
            f"Calibration split had no usable tokens in vocab range [0,{vocab_size}): "
            f"{dataset_root}/{split}"
        )

    return TokenStatsState(token_counts=counts, total_count=float(n_used)), n_used


def _checkpoint_exists(checkpoint_root: Path, exp_folder: str, exp_name: str) -> bool:
    return (checkpoint_root / exp_folder / exp_name / "latest.json").exists()


def _seed_one(
    *,
    target: ImportTarget,
    profile_root: Path,
    checkpoint_root: Path,
    exp_folder: str,
    step: int,
    prior_tokens: int,
    dataset_root: Path | None,
    split: str,
    max_tokens: int,
    force: bool,
) -> Path:
    checkpoint_dir = checkpoint_root / exp_folder / target.import_exp_name
    if _checkpoint_exists(checkpoint_root, exp_folder, target.import_exp_name) and not force:
        return checkpoint_dir / "latest.json"

    profile = _load_profile(profile_root=profile_root, model_key=target.model_key)
    vocab_size = int(profile.get("vocab_size", 0))
    if vocab_size <= 0:
        raise ValueError(f"Invalid vocab_size in profile for {target.model_key}: {vocab_size}")

    if dataset_root is None:
        state, used = _zipf_prior_state(vocab_size=vocab_size, prior_tokens=prior_tokens)
        seed_mode = "zipf_prior"
        seed_source = ""
    else:
        state, used = _calibration_state(
            dataset_root=dataset_root,
            split=split,
            vocab_size=vocab_size,
            max_tokens=max_tokens,
        )
        seed_mode = "dataset_calibration"
        seed_source = str(dataset_root)

    checkpointer = Phase1Checkpointer(checkpoint_dir=checkpoint_dir)
    payload = {
        "exp_name": target.import_exp_name,
        "exp_folder": exp_folder,
        "elapsed_seconds": 0.0,
        "loss": 0.0,
        "gradient_norm": 0.0,
        "model_state": state.to_jsonable(),
        "init_source": "external_hf",
        "external_model_id": target.hf_model_id,
        "adapter_recipe": "import_seed",
        "seed_mode": seed_mode,
        "seed_source": seed_source,
        "seed_tokens_used": used,
        "created_at_unix": int(time.time()),
        "profile_revision": str(profile.get("source_revision", "")),
        "profile_sha256": str(profile.get("source_config_sha256", "")),
    }
    checkpointer.save(step=step, payload=payload)
    return checkpoint_dir / "latest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create phase-1 import checkpoints required by adapter-path external runs "
            "(import-qwen05-fa-base and import-smol360-fa-base)."
        )
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "qwen2_5_0_5b", "smollm2_360m"],
        help="Which import checkpoint(s) to seed.",
    )
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=Path("./artifacts/external_models"),
        help="Root containing model_profile.json files from scripts/07_prepare_external_models.py.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("./checkpoints"),
        help="Checkpoint root (same as deploy_paths.checkpoint).",
    )
    parser.add_argument("--exp-folder", default="external_phase1", help="Checkpoint experiment folder.")
    parser.add_argument("--step", type=int, default=0, help="Checkpoint step index to write.")

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Optional token dataset root for calibration seeding. If omitted, uses a deterministic "
            "Zipf prior."
        ),
    )
    parser.add_argument("--split", default="train", help="Split name for calibration dataset root.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200000,
        help="Maximum calibration tokens to consume when --dataset-root is set.",
    )
    parser.add_argument(
        "--prior-tokens",
        type=int,
        default=4096,
        help="Number of prior token IDs to seed in Zipf mode.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing import checkpoints.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.step < 0:
        raise ValueError("--step must be >= 0")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be > 0")
    if args.prior_tokens <= 0:
        raise ValueError("--prior-tokens must be > 0")

    profile_root = args.profile_root.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve() if args.dataset_root else None

    if args.model == "all":
        selected = [TARGETS["qwen2_5_0_5b"], TARGETS["smollm2_360m"]]
    else:
        selected = [TARGETS[args.model]]

    checkpoint_root.mkdir(parents=True, exist_ok=True)

    missing_profiles = [
        _profile_path(profile_root, t.model_key)
        for t in selected
        if not _profile_path(profile_root, t.model_key).exists()
    ]
    if missing_profiles:
        missing = ", ".join(str(p) for p in missing_profiles)
        raise FileNotFoundError(
            f"Missing required external model profiles: {missing}. "
            "Run scripts/07_prepare_external_models.py first."
        )

    for target in selected:
        latest_path = _seed_one(
            target=target,
            profile_root=profile_root,
            checkpoint_root=checkpoint_root,
            exp_folder=args.exp_folder,
            step=args.step,
            prior_tokens=args.prior_tokens,
            dataset_root=dataset_root,
            split=args.split,
            max_tokens=args.max_tokens,
            force=args.force,
        )
        print(f"Seeded import checkpoint: {target.import_exp_name} -> {latest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
