# Reproducibility

The stage ladder lives in
[`configs/research/warmstart_registry.yaml`](../configs/research/warmstart_registry.yaml).
Each stage names its parent checkpoint, its data, and its config overrides.
The orchestrator in `ttt/research/orchestrator.py` runs stages in order. It
checks that each parent checkpoint exists and matches what the registry
expects, and it writes a run manifest that records the command, the parents,
and the dataset fingerprints.

## Runtime modes

`--runtime-mode` (or `training.runtime_mode`) picks what a stage actually does.

| Mode | What it does | Needs |
| --- | --- | --- |
| `simulate` | writes made-up losses and checkpoints | nothing |
| `token_stats` | fits a token-frequency model to the data | tokenized data |
| `jax_train` | trains the JAX model | tokenized data, GPUs for real sizes |
| `jax_eval` | scores a checkpoint with the JAX model | tokenized data, a checkpoint |

`simulate` and `token_stats` exist to test the orchestration on a laptop. Only
`jax_train` and `jax_eval` produce numbers worth reporting.

## Levels of reproduction

1. **CPU smoke run.** The quick start in the [README](../README.md#quick-start-on-cpu)
   runs the whole ladder in `token_stats` mode on synthetic tokens. The test
   `tests/test_registry_ladder_smoke.py` runs a subset of it in CI.
2. **Restore audit on CPU.** With the 125M seed and the Books3 validation split
   on disk, `scripts/84_audit_warmstart_restore.py` restores the seed into each
   converted config, compares every inherited tensor with the checkpoint, and
   scores the converted model next to its parent. Section 0 of the
   [runbook](REVISION_V3_RUNBOOK.md) lists the expected values.
3. **The revision-v3 ladder on 8x H100.** The [runbook](REVISION_V3_RUNBOOK.md)
   covers the bootstrap, data, seeds, a smoke run, the pilot, evaluation, and
   the pilot gates. `scripts/96_run_revision_v3_ladder.py` launches it.

## Evaluation

`scripts/34_eval_matrix_jax.py` restores each run's checkpoint in float32 and
scores it on Books3 at 32K (and DCLM at 8K for the bridges). It refuses any
checkpoint that does not restore completely into its own config.
`scripts/80_analyze_e1_bridge_effect.py` turns a `34` summary into the
preregistered E1 statistic, its paired seed bootstrap interval, and the margin
decision.

`scripts/18_eval_matrix.py` runs the token-statistics evaluator. It cannot
score Orbax checkpoints, so use it only with `token_stats` runs.

## Stage map

| Stage | Description |
| --- | --- |
| S0 | full-attention seed extended directly to 32K |
| S1 | full-attention seed converted to sliding-window attention, then extended |
| S2 | full-attention seed converted to TTT-E2E, adapted through the 8K bridge, then extended |
| S2-minus | S2 without the 8K bridge |
| S3 | TTT-E2E trained from scratch at 8K, then extended |
| S3W | S3 at S2's parameter count |

Data and seeds are not in git. [DATASETS.md](DATASETS.md) and
[ARTIFACTS.md](ARTIFACTS.md) explain where to get them.
