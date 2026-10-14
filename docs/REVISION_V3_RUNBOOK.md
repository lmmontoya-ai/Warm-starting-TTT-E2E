# Revision-v3 runbook: the fixed 125M warm-start ladder on 8x H100

This runbook runs the revision-v3 125M program with the fixes in
`PREREGISTRATION_REVISION_V3.md` Amendment 1: a value-checked restore, LR
schedules that fit each run, the published extension recipe, RoPE base 2e6 for
S0, and one bridge per seed. It covers the Phase 1 pilot. Phase 2 reuses the
same commands with more seeds, once the pilot gates pass and the maintainer
confirms the amendment's pilot section.

```text
box:        8x H100 80GB SXM from any provider, Ubuntu 22.04, CUDA 12
topology:   data=8, state=1
extension:  120 steps x global batch 32 (accum 4, one 32K sequence per GPU per micro-step)
bridge:     480 steps x global batch 64 at 8K (no accumulation)
profile:    +deploy=revision_v3_vast_h100_8x
paper run:  revision_v3_widthpreserving_v1
```

Do not use Blackwell (B200/B300) parts; the pinned JAX stack failed there in June.

## Budget

Training GPU-hours from the March and June runs (a 126M-token extension costs
the same at 480 x 8 and at 120 x 32 with accumulation):

| Phase 1 pilot run | GPU-h each | Count | Total |
| --- | ---: | ---: | ---: |
| S0 full attention 32K | 0.5 | 1 | 0.5 |
| S1 sliding window 32K | 1.5 | 1 | 1.5 |
| S3 re-extension | 1.5 | 1 | 1.5 |
| S2-minus | 1.9 | 1 | 1.9 |
| Bridge (10%) | 2.1 | 3 LRs | 6.3 |
| S2 extension | 1.4 | 3 LRs | 4.2 |
| Total | | | about 16 |

That is about 2 hours of training on 8 GPUs. Plan 5 to 6 box hours for setup,
the smoke run, and evaluation. At about $25 an hour for 8x H100 (the September
2026 market price) the pilot costs $130 to $220. Phase 2 (more seeds, controls, frontier) is estimated at
about 38 training GPU-hours.

## 0. Before renting anything ($0)

```bash
uv run --exact python -m pytest -q \
  tests/test_warmstart_guard.py tests/test_warmstart_validity.py tests/test_schedule_guard.py \
  tests/test_jax_checkpoint_partial_restore.py tests/test_jax_runtime_warmstart_gate.py \
  tests/test_revision_v3_launcher.py tests/test_jax_rope_short_context.py
uv run --exact python scripts/95_check_registry_warmstart_shapes.py \
  --json-out reports/revision_v3/registry_warmstart_shape_check.json
```

The shape check must print `WARM-START SHAPE CHECK: PASS` with no `[ERROR]` or
`[WARNING]` rows. Expected `[INFO]` rows: the added prime modules and S0's
declared `model.rope_theta 500000.0 -> 2000000.0`.

If the FA seed and Books3 validation data are on the laptop, run the restore
audit with the loss probe locally (CPU, a few minutes per stage):

```bash
for stage in S0_125M S1_125M S2_ADAPT_125M S2_MINUS_125M; do
  uv run --exact python scripts/84_audit_warmstart_restore.py \
    --stage-id "$stage" --checkpoint-dir /path/to/pretrain-125m-fa \
    --probe-root /path/to/books3 --probe-parent \
    --json-out "reports/revision_v3/preflight_local/audit_$stage.json"
done
```

Expected (2026-09-28 run): all `PASS`, value check `15/15`, probe losses about
3.91 (S0), 3.89 (S1), 4.21 (bridge), 4.25 (direct) against the seed's 3.89.

## 1. Constants on the box

```bash
export REPO_ROOT=$HOME/Warm-starting-TTT-E2E
export DATA_ROOT=$HOME/ttt-e2e-data
export DCLM_ROOT=$DATA_ROOT/dclm_filter_8k
export BOOKS_ROOT=$DATA_ROOT/books3
export SEED_BUNDLE=$HOME/ttt-e2e-seeds
export GCS_BILLING_PROJECT=<your-gcp-project>
export PAPER_RUN_ID=revision_v3_widthpreserving_v1
export EXP_FOLDER=revision_v3_widthpreserving_v1
# S0 full attention at 32K OOMed during XLA autotuning without this (June 2026).
export XLA_FLAGS=--xla_gpu_autotune_level=3
```

Keep credentials in `.env` and keep it untracked on the box.

## 2. Clone and bootstrap

```bash
git clone https://github.com/lmmontoya-ai/Warm-starting-TTT-E2E.git "$REPO_ROOT"
cd "$REPO_ROOT"
scripts/bootstrap_gpu_host.sh --repo-root "$REPO_ROOT" --expected-gpus 8
```

The script installs uv and Python 3.12, syncs the locked environment, and
checks that JAX sees 8 GPUs. It ends with `backend=gpu device_count=8` and
`GPU host ready`.

## 3. Restore the two training parents

The ladder starts from two 125M checkpoints: the full-attention seed and the
scratch TTT-E2E pretrain. Both are valid. Everything downstream of them is
retrained. Request them as described in [ARTIFACTS.md](ARTIFACTS.md) and
extract the bundle into `$SEED_BUNDLE`. Then restore each one into this
ladder's layout:

```bash
cd "$REPO_ROOT"
for spec in \
  "S0_PRETRAIN_FA_125M pretrain-125m-fa" \
  "S3_PRETRAIN_E2E_125M pretrain-125m-e2e"
do
  set -- $spec
  uv run --exact python scripts/46_restore_stage.py \
    --source-dir "$SEED_BUNDLE" \
    --source-paper-run-id protocol_r_125m_main_v1 \
    --source-stage-id "$1" \
    --source-run-id "$2" \
    --target-paper-run-id "$PAPER_RUN_ID"
done
```

Expected checkpoint roots:

```text
checkpoints/revision_v3_widthpreserving_v1/pretrain-125m-fa
checkpoints/revision_v3_widthpreserving_v1/pretrain-125m-e2e
```

## 4. Stage the tokenized data and fingerprints

Download the 125M slice of the authors' Requester Pays buckets, about 40 GB
(DCLM train 10.4, DCLM val 20.4, Books3 train 0.8, Books3 val 8.4). The reads
are billed to `$GCS_BILLING_PROJECT`. See [DATASETS.md](DATASETS.md).

```bash
uv run --exact python scripts/00_data_fetch.py \
  --dest-root "$DATA_ROOT" --budget 125m \
  --billing-project "$GCS_BILLING_PROJECT"
for dataset in dclm_filter_8k books3; do
  for split in train val; do
    uv run --exact python scripts/13_dataset_fingerprint.py \
      --dataset-id "$dataset" --path "$DATA_ROOT/$dataset" --split "$split"
  done
done
```

If you keep a copy in your own S3-compatible bucket, replace the first command
with `scripts/28_fetch_s3_dataset.py --dest-root "$DATA_ROOT" --bucket <bucket> --prefix <prefix>`.

## 5. Restore audit on the box

```bash
for stage in S0_125M S1_125M S2_ADAPT_125M S2_MINUS_125M; do
  uv run --exact python scripts/84_audit_warmstart_restore.py \
    --stage-id "$stage" --checkpoint-root ./checkpoints --exp-folder "$EXP_FOLDER" \
    --probe-root "$BOOKS_ROOT" --probe-parent \
    --json-out "reports/revision_v3/$PAPER_RUN_ID/restore_audit_$stage.json"
done
```

Every stage must print `WARM-START RESTORE AUDIT: PASS`, a value check with
all same-shape tensors equal, and probe losses close to the section-0 values.

## 6. Smoke run in its own folder

A separate paper run id and exp folder, so no smoke checkpoint can be resumed
by the real ladder:

```bash
uv run --exact python scripts/96_run_revision_v3_ladder.py \
  --paper-run-id revision_v3_smoke_v1 --exp-folder revision_v3_smoke_v1 \
  --deploy revision_v3_vast_h100_8x \
  --dclm-root "$DCLM_ROOT" --books-root "$BOOKS_ROOT" \
  --stages S0_125M,S1_125M,S3_125M,S2_ADAPT_125M,S2_125M,S2_MINUS_125M \
  --seeds 1 --adapt-steps 10 --ext-steps 4 --save-milestone-freq 5
```

The smoke needs the two parents under `checkpoints/revision_v3_smoke_v1/`;
copy or symlink them from step 3. Then read each
`experiments/revision_v3_smoke_v1/<stage>/<run>/metrics.jsonl`:

- step-0 losses: S0 and S1 near the seed's Books loss, bridge and direct about
  0.3 higher, S3 near 3.4 (the trainer aborts above 7.0). S0's step-0 loss is
  measured at 32K, three quarters of which lies beyond the seed's 8K training
  length, so it can sit above the 2K probe value. If S0 alone trips the 7.0
  gate while its `restore_report.json` value check is complete, that is
  length extrapolation, not a failed restore: record it, raise
  `training.warmstart_max_initial_loss` for S0 only, and note it in the pilot
  memo;
- `restore_report.json` next to each warm-started run shows the value check
  with every same-shape tensor equal;
- extension steps with accumulation 4 take about four times a batch-8 step.

Note the steady-state seconds per step, extrapolate the pilot, then delete the
smoke folders:

```bash
rm -rf checkpoints/revision_v3_smoke_v1 experiments/revision_v3_smoke_v1
```

## 7. Pilot (Phase 1)

Confirm the pilot section of Amendment 1 first. Then three launches, one per
bridge peak LR. The baselines (S0, S1, S3, S2-minus) run in the first launch
and are skipped as complete in the next two.

```bash
for lr in 4e-4 1e-3 3e-3; do
  uv run --exact python scripts/96_run_revision_v3_ladder.py \
    --paper-run-id "$PAPER_RUN_ID" --exp-folder "$EXP_FOLDER" \
    --deploy revision_v3_vast_h100_8x \
    --dclm-root "$DCLM_ROOT" --books-root "$BOOKS_ROOT" \
    --stages S0_125M,S1_125M,S3_125M,S2_MINUS_125M,S2_ADAPT_125M,S2_125M \
    --seeds 1 --variant "lr$lr" --adapt-peak-lr "$lr"
done
```

To upload each finished stage to a Hugging Face repo of your own, add
`--export-to-hf --hf-repo-id <user>/<repo>`. The launcher refuses
`--export-to-hf` without a repo.

An LR-only variant tags the bridge and S2 (`adapt-125m-e2e-8K-from-fa-lr1e-3-seed001`,
`ext-125m-e2e-32K-from-fa-bridge-lr1e-3-seed001`) but not S2-minus, which does
not depend on the bridge LR; the three launches share one
`ext-125m-e2e-32K-from-fa-direct-seed001`. A `--prime-post-norm-init` variant
tags S2-minus as well, because it changes that run.

Every run writes `restore_report.json`, the resolved LR schedule under
`run_started.lr_schedule` in `events.jsonl`, and a ladder summary under
`reports/revision_v3/$PAPER_RUN_ID/`.

## 8. Checkpoint-based 64-batch evaluation

Same surface as revision v2: float32 restore, Books3 validation, context
`32768`, `64` batches of `8`. Use `scripts/34_eval_matrix_jax.py`
(`scripts/18_eval_matrix.py` runs the token-statistics evaluator and cannot
score Orbax checkpoints).

```bash
uv run --exact python scripts/34_eval_matrix_jax.py \
  --paper-run-id "$PAPER_RUN_ID" --exp-folder "$EXP_FOLDER" \
  --checkpoint-root ./checkpoints \
  --dclm-root "$DCLM_ROOT" --books-root "$BOOKS_ROOT" \
  --eval-id jax_books32k_eval64 --contexts 32768 --datasets books3 \
  --eval-batches 64 --eval-batch-size 8 \
  --stages S0_125M,S1_125M,S3_125M,S2_125M,S2_MINUS_125M \
  --summary-json "reports/revision_v3/$PAPER_RUN_ID/books32k_eval64_summary.json" \
  --summary-csv  "reports/revision_v3/$PAPER_RUN_ID/books32k_eval64_summary.csv" \
  --strict

uv run --exact python scripts/34_eval_matrix_jax.py \
  --paper-run-id "$PAPER_RUN_ID" --exp-folder "$EXP_FOLDER" \
  --checkpoint-root ./checkpoints \
  --dclm-root "$DCLM_ROOT" --books-root "$BOOKS_ROOT" \
  --eval-id jax_dclm8k_eval64 --contexts 8192 --datasets dclm_filter_8k \
  --eval-batches 64 --eval-batch-size 8 \
  --stages S2_ADAPT_125M \
  --summary-json "reports/revision_v3/$PAPER_RUN_ID/bridge_dclm8k_eval64_summary.json" \
  --summary-csv  "reports/revision_v3/$PAPER_RUN_ID/bridge_dclm8k_eval64_summary.csv" \
  --strict
```

Adding `--eval-after` to the step-7 launches runs the same two commands
automatically after training. The eval now refuses any checkpoint that does not
restore completely into its own config.

## 9. Pilot gates

Read against the pilot section of Amendment 1:

1. every run passed the trainer gates;
2. S0 Books32K within `0.2` of the re-extended S3; if not, stop and diagnose;
3. pick the bridge peak LR by the committed rule;
4. write the pilot memo under `reports/revision_v3/$PAPER_RUN_ID/` before any
   Phase 2 launch.

## 10. Tear down

1. If you used `--export-to-hf`, check that every ladder summary shows
   `hf_export.status == succeeded`.
2. Copy `reports/revision_v3/` and the `restore_report.json` files off the box.
3. Destroy the instance and confirm in the provider's console that it is gone.
4. Record the session's hours and cost in the pilot memo.
