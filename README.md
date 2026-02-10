# Reproducing TTT-E2E

This workspace organizes our reproduction of the TTT-E2E paper.

- Target reproduction claims:
- Scaling with context length (8K–128K) for a 3B model.
- Constant inference latency and ~2.7× speedup vs full attention at 128K on H100.
- NIAH (RULER) recall gap between TTT-E2E and full attention.

- Original artifacts are in `og_repo/`.
- Paper: `og_repo/TTT-E2E.pdf`.
- Code snapshot (no git history): `og_repo/e2e/`.
- Snapshot metadata: `og_repo/README.md`.
- Research summary: `report.md`.
- Reproduction plan: `reproduction_roadmap.md`.

Original code snapshot commit: `f73017b516781a7afee51237489476372012c171`.

## Phase 1 Local Runtime

Phase 1 adds lightweight local runtimes to validate warm-start experiment
orchestration before full JAX parity.

- `training.runtime_mode=simulate`: deterministic orchestration checks.
- `training.runtime_mode=token_stats`: token-driven updates/loss on local token
  streams.

### Quick Commands

Generate small local token files for smoke tests:

```bash
uv run --exact python scripts/04_make_token_data.py --out /tmp/phase1_token_data
```

Print pretrained matrix launch commands:

```bash
uv run --exact python scripts/03_pretrained_matrix.py --runtime-mode token_stats
```

Summarize completed runs (tokens, wall-clock, checkpoints, restore status):

```bash
uv run --exact python scripts/05_phase1_report.py --exp-dir ./experiments
```
