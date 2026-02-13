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

Run the short-budget pilot matrix (B1/B2/P1/P2) and emit consolidated reports:

```bash
uv run --exact python scripts/06_phase1_pilot.py \
  --bootstrap-token-data \
  --skip-existing
```

Prepare external model profiles (Qwen2.5-0.5B + SmolLM2-360M):

```bash
uv run --exact python scripts/07_prepare_external_models.py --model all
```

Seed required adapter import checkpoints:

```bash
uv run --exact python scripts/10_seed_external_import_checkpoints.py \
  --model all \
  --exp-folder external_phase1
```

Important: adapter-path runs require import checkpoints to exist:
- `./checkpoints/<exp_folder>/import-qwen05-fa-base/latest.json`
- `./checkpoints/<exp_folder>/import-smol360-fa-base/latest.json`

Print external-model experiment commands (scratch + adapter paths):

```bash
uv run --exact python scripts/08_external_matrix.py --model all --path all
```

Run external-model short-budget pilot matrix and emit reports:

```bash
uv run --exact python scripts/09_external_pilot.py \
  --model all \
  --path all \
  --bootstrap-token-data \
  --dclm-root /tmp/phase1_token_data_dclm \
  --books-root /tmp/phase1_token_data_books \
  --skip-existing
```

Evaluate completed runs with paper-style proxy metrics:

```bash
uv run --exact python scripts/11_external_eval.py \
  --exp-folder external_phase1_pilot \
  --dclm-root /tmp/phase1_token_data_dclm \
  --books-root /tmp/phase1_token_data_books
```

Run the full external flow end-to-end in one command:

```bash
uv run --exact python scripts/12_external_e2e_research.py \
  --model all \
  --path all \
  --budget pilot \
  --exp-folder external_phase1_research \
  --bootstrap-token-data \
  --dclm-root /tmp/phase1_token_data_dclm \
  --books-root /tmp/phase1_token_data_books
```
