# Agent Instructions

**IMPORTANT**
- `og_repo/` is a read-only reference snapshot of the original paper code and artifacts.
- In this workspace, `reference/` is also a read-only snapshot of original paper artifacts.
- All **reproduction implementation** must live in this repo (outside `og_repo/`).
- All **reproduction implementation** must live outside `reference/` as well.
- Do not modify or add new code inside `og_repo/`.
- Do not modify or add new code inside `reference/`.
- When you need to mirror a config or function from the original code, re‑implement it in the main repo and note the source.
- **Learning requirement:** do not copy code or configs wholesale from `og_repo/`. Re‑implement step‑by‑step and explain the logic/intuition and design choices at each stage.
- Apply the same non-copy rule for `reference/`.

## Current Reproduction Workflow (Phase 1)
- Use local runtime only (`ttt/` in main repo), not snapshot runtime code.
- Runtime modes:
  - `training.runtime_mode=simulate` for orchestration dry runs.
  - `training.runtime_mode=token_stats` for token-driven pilot runs.
- Pretrained matrix helper: `scripts/03_pretrained_matrix.py`.
- Local token data generator for pilots: `scripts/04_make_token_data.py`.
- Phase-1 run summarizer (tokens, wall-clock, checkpoint lineage): `scripts/05_phase1_report.py`.
- Pilot matrix orchestrator (B1/B2/P1/P2 short runs + reports): `scripts/06_phase1_pilot.py`.
- External model profile bootstrapper (Qwen/Smol HF configs): `scripts/07_prepare_external_models.py`.
- External profile default location: `./artifacts/external_models/<model_key>/model_profile.json`.
- External experiment command generator (scratch + adapter matrix): `scripts/08_external_matrix.py`.
- External pilot orchestrator (Qwen/Smol scratch + adapter runs): `scripts/09_external_pilot.py`.
