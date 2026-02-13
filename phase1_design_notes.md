# Phase 1 Design Notes (Local Re-implementation)

This document records how Phase 1 re-implements pieces of the TTT-E2E workflow
outside the reference snapshot and why each design was chosen.

## Scope
Phase 1 focuses on experiment orchestration and warm-start flow validation, not
full JAX model parity.

## Source Mapping and Re-implementation Choices

1. Local config schema (`ttt/config.py`)
- Inspiration: `reference/e2e/ttt/config.py` and existing `configs/` tree.
- Re-implementation choice:
  - Keep the same high-level config group shape (`training`, `model`, `backend`,
    `checkpoint`, `deploy_paths`) so existing Hydra config files compose without
    edits.
  - Preserve key enums (`train_mode`, `load_part`) and warm-start fields
    (`resume_exp_name`, `load_part`) because they are required for pretrained
    initialization experiments.
  - Keep implementation lightweight and dependency-minimal for local Phase 1
    validation.

2. Local runtime + checkpoints (`ttt/runtime.py`, `ttt/infra/checkpoint.py`)
- Inspiration: orchestration semantics from `reference/e2e/ttt/train.py` and
  checkpoint intent from `reference/e2e/ttt/infra/checkpoint.py`.
- Re-implementation choice:
  - Implement deterministic simulated training loop (`runtime_mode=simulate`)
    to validate run sequencing, warm-start behavior, checkpoint directories, and
    metric logging quickly.
  - Add a lightweight token-statistics trainer (`runtime_mode=token_stats`) that
    consumes real token batches and performs real token-dependent updates/losses,
    while remaining dependency-light.
  - Implement JSON checkpoints with `latest` pointer and `load_part`
    behavior:
    - `none`: fresh start.
    - `params`: warm-start metadata only, step resets to 0.
    - `all`: full resume semantics, loop resumes from `restore_step + 1`.
  - This captures experiment control-flow logic now, while avoiding premature
    coupling to CUDA/JAX kernels.

3. Train entrypoint (`ttt/train.py`)
- Inspiration: Hydra entrypoint shape from `reference/e2e/ttt/train.py`.
- Re-implementation choice:
  - Keep the same `train` CLI pattern and config composition.
  - Always materialize resolved/unresolved configs to run dirs for provenance.
  - Run simulator only when `training.dummy_dataset=true` to keep dry-run and
    execution modes explicit.

4. Pretrained warm-start experiment configs (`configs/experiment/760m/pretrained/*.yaml`)
- Inspiration: existing local pretrain/ext E2E/FA configs in
  `configs/experiment/760m/`.
- Re-implementation choice:
  - Add three configs corresponding to the planned research matrix:
    - `adapt-760m-e2e-8K-from-fa` (bridge stage B)
    - `ext-760m-e2e-32K-from-fa-direct` (direct warm-start)
    - `ext-760m-e2e-32K-from-fa-bridge` (bridge stage C)
  - Reuse established model/training defaults and explicitly set
    `load_part=params` + `resume_exp_name` to encode checkpoint lineage.

5. Matrix launcher helper (`scripts/03_pretrained_matrix.py`)
- Inspiration: Phase-1 research matrix documented in
  `pretrained_efficiency_research_plan.md`.
- Re-implementation choice:
  - Generate copy-paste-ready launch commands for B1/B2/P1/P2 variants with
    deploy/data/checkpoint overrides and phase-1 runtime mode selection.

6. Local token batch pipeline (`ttt/dataloader/lm_dataset.py`,
   `ttt/model/token_stats.py`, `scripts/04_make_token_data.py`)
- Inspiration: role of token-sequence batching in `reference/e2e/ttt/dataloader`.
- Re-implementation choice:
  - Add a portable loader that supports:
    - deterministic dummy streams,
    - local `train.json` / `val.json`,
    - optional `train.txt` / `val.txt`,
    - optional zarr splits when zarr is installed.
  - Add a token-frequency model so phase-1 runs can produce real NLL trends and
    warm-start effects without full framework dependencies.

7. Experiment tracker (`scripts/05_phase1_report.py`)
- Inspiration: efficiency-reporting requirement in
  `pretrained_efficiency_research_plan.md` (tokens + wall-clock + lineage table).
- Re-implementation choice:
  - Read each run's local artifacts (`phase1_resolved_config.yaml`,
    `phase1_metrics.jsonl`) and checkpoint pointers (`latest.json`) to summarize:
    tokens seen, final loss, elapsed wall-clock, throughput, and restore metadata.
  - Keep output portable (`table`, `json`, `csv`) so results can be pasted into
    reports or consumed by later analysis scripts.
  - Avoid external tracking dependencies in Phase 1; use repo-local logs as the
    source of truth.

8. Pilot matrix orchestrator (`scripts/06_phase1_pilot.py`)
- Inspiration: immediate-next-action requirement to run short pilot schedules and
  validate warm-start behavior before full compute runs.
- Re-implementation choice:
  - Encode the planned B1/B2/P1/P2 stage graph in one script and execute each
    stage with explicit step budgets (`pretrain`, `adapt`, `ext`).
  - Add resumability controls (`--skip-existing`) and optional local token-data
    bootstrap to reduce setup friction for pilot checks.
  - Emit a run manifest plus consolidated CSV/JSON summaries by invoking
    `scripts/05_phase1_report.py` at the end of each pilot run.

9. External-pretrained track scaffolding (Qwen/Smol)
- Inspiration: follow-up research direction to initialize TTT-E2E from external
  pretrained dense models while preserving comparability.
- Re-implementation choice:
  - Add local model/training/experiment configs for:
    - `Qwen/Qwen2.5-0.5B`
    - `HuggingFaceTB/SmolLM2-360M`
  - Cover both scratch and adapter workflows:
    - FA scratch pretrain + 32K extension,
    - FA imported controls,
    - SWA bridge + TTT bridge,
    - direct TTT extension.
  - Add automation scripts for external runs:
    - `scripts/07_prepare_external_models.py` (profile bootstrap),
    - `scripts/08_external_matrix.py` (command generation),
    - `scripts/09_external_pilot.py` (execution + manifest/report).

## Why this order
- First make experiment graph and lineage executable.
- Then port full model internals module-by-module.
- This minimizes wasted effort and allows early validation that the pretrained
  direction can be run cleanly in this repo.
