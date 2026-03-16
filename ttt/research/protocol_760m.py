from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


PROTOCOL_R_760M_PAPER_RUN_ID = "protocol_r_760m_author_seed_v1"
PROTOCOL_R_760M_EXP_FOLDER = "protocol_r_760m_author_seed_v1"

FAITHFUL_EXT_GLOBAL_BATCH_SIZE = 32
FAITHFUL_ADAPT_GLOBAL_BATCH_SIZE = 64
REVISED_8X_H200_GLOBAL_BATCH_SIZE = 8

FAITHFUL_EXT_STEPS = 725
FAITHFUL_ADAPT_STEPS = 2900

ALL_760M_STAGE_IDS = ("S0", "S1", "S2_ADAPT", "S2", "S3")
CORE_760M_STAGE_IDS = ("S0", "S1", "S2", "S3")

AUTHOR_SEED_SOURCES_760M: dict[str, str] = {
    "S0": "760m_fa",
    "S1": "760m_fa",
    "S2_ADAPT": "760m_fa",
    "S3": "760m_e2e",
}

PAPER_STAGE_LABELS_760M: dict[str, str] = {
    "S0": "S0_760M",
    "S1": "S1_760M",
    "S2_ADAPT": "S2_ADAPT_760M",
    "S2": "S2_760M",
    "S3": "S3_760M",
}


@dataclass(frozen=True)
class StageProtocol760M:
    stage_id: str
    paper_stage_label: str
    kind: str
    faithful_global_batch_size: int
    faithful_total_steps: int
    revised_global_batch_size: int
    revised_total_steps: int
    author_seed_key: str = ""
    token_budget_preserved: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def scaled_steps_for_token_budget(
    *,
    original_steps: int,
    original_global_batch_size: int,
    revised_global_batch_size: int,
) -> int:
    if original_steps <= 0:
        raise ValueError("original_steps must be positive.")
    if original_global_batch_size <= 0:
        raise ValueError("original_global_batch_size must be positive.")
    if revised_global_batch_size <= 0:
        raise ValueError("revised_global_batch_size must be positive.")
    return int(math.ceil(original_steps * original_global_batch_size / revised_global_batch_size))


def build_protocol_r_760m_stage_map(
    *,
    revised_global_batch_size: int = REVISED_8X_H200_GLOBAL_BATCH_SIZE,
) -> dict[str, StageProtocol760M]:
    ext_steps = scaled_steps_for_token_budget(
        original_steps=FAITHFUL_EXT_STEPS,
        original_global_batch_size=FAITHFUL_EXT_GLOBAL_BATCH_SIZE,
        revised_global_batch_size=revised_global_batch_size,
    )
    adapt_steps = scaled_steps_for_token_budget(
        original_steps=FAITHFUL_ADAPT_STEPS,
        original_global_batch_size=FAITHFUL_ADAPT_GLOBAL_BATCH_SIZE,
        revised_global_batch_size=revised_global_batch_size,
    )
    return {
        "S0": StageProtocol760M(
            stage_id="S0",
            paper_stage_label=PAPER_STAGE_LABELS_760M["S0"],
            kind="ext",
            faithful_global_batch_size=FAITHFUL_EXT_GLOBAL_BATCH_SIZE,
            faithful_total_steps=FAITHFUL_EXT_STEPS,
            revised_global_batch_size=revised_global_batch_size,
            revised_total_steps=ext_steps,
            author_seed_key=AUTHOR_SEED_SOURCES_760M["S0"],
        ),
        "S1": StageProtocol760M(
            stage_id="S1",
            paper_stage_label=PAPER_STAGE_LABELS_760M["S1"],
            kind="ext",
            faithful_global_batch_size=FAITHFUL_EXT_GLOBAL_BATCH_SIZE,
            faithful_total_steps=FAITHFUL_EXT_STEPS,
            revised_global_batch_size=revised_global_batch_size,
            revised_total_steps=ext_steps,
            author_seed_key=AUTHOR_SEED_SOURCES_760M["S1"],
        ),
        "S2_ADAPT": StageProtocol760M(
            stage_id="S2_ADAPT",
            paper_stage_label=PAPER_STAGE_LABELS_760M["S2_ADAPT"],
            kind="adapt",
            faithful_global_batch_size=FAITHFUL_ADAPT_GLOBAL_BATCH_SIZE,
            faithful_total_steps=FAITHFUL_ADAPT_STEPS,
            revised_global_batch_size=revised_global_batch_size,
            revised_total_steps=adapt_steps,
            author_seed_key=AUTHOR_SEED_SOURCES_760M["S2_ADAPT"],
        ),
        "S2": StageProtocol760M(
            stage_id="S2",
            paper_stage_label=PAPER_STAGE_LABELS_760M["S2"],
            kind="ext",
            faithful_global_batch_size=FAITHFUL_EXT_GLOBAL_BATCH_SIZE,
            faithful_total_steps=FAITHFUL_EXT_STEPS,
            revised_global_batch_size=revised_global_batch_size,
            revised_total_steps=ext_steps,
        ),
        "S3": StageProtocol760M(
            stage_id="S3",
            paper_stage_label=PAPER_STAGE_LABELS_760M["S3"],
            kind="ext",
            faithful_global_batch_size=FAITHFUL_EXT_GLOBAL_BATCH_SIZE,
            faithful_total_steps=FAITHFUL_EXT_STEPS,
            revised_global_batch_size=revised_global_batch_size,
            revised_total_steps=ext_steps,
            author_seed_key=AUTHOR_SEED_SOURCES_760M["S3"],
        ),
    }


def build_protocol_r_760m_manifest(
    *,
    paper_run_id: str,
    exp_folder: str,
    revised_global_batch_size: int = REVISED_8X_H200_GLOBAL_BATCH_SIZE,
    save_milestone_freq: int = 120,
    seed: int = 0,
) -> dict[str, Any]:
    stage_map = build_protocol_r_760m_stage_map(
        revised_global_batch_size=revised_global_batch_size,
    )
    return {
        "schema_version": "1.0",
        "paper_run_id": paper_run_id,
        "exp_folder": exp_folder,
        "protocol": "revised",
        "description": (
            "Author-seeded 760M revised matched protocol for 8x H200. "
            "All train stages run at the smallest passing global batch size, "
            "with total steps scaled to preserve the original token budgets."
        ),
        "revised_global_batch_size": revised_global_batch_size,
        "faithful_ext_global_batch_size": FAITHFUL_EXT_GLOBAL_BATCH_SIZE,
        "faithful_adapt_global_batch_size": FAITHFUL_ADAPT_GLOBAL_BATCH_SIZE,
        "faithful_ext_steps": FAITHFUL_EXT_STEPS,
        "faithful_adapt_steps": FAITHFUL_ADAPT_STEPS,
        "effective_ext_steps": stage_map["S0"].revised_total_steps,
        "effective_adapt_steps": stage_map["S2_ADAPT"].revised_total_steps,
        "save_milestone_freq": save_milestone_freq,
        "seed": seed,
        "stages": {stage_id: spec.to_dict() for stage_id, spec in stage_map.items()},
    }
