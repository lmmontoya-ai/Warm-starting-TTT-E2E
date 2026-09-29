# Revision V3 Runbook: Fixed 125M Warm-Start Ladder On 8x H100

Purpose: run the revision-v3 125M program with the fixes in
`PREREGISTRATION_REVISION_V3.md` Amendment 1 (value-checked restore, LR
schedules that fit the run, published extension recipe, RoPE base 2e6 for S0,
one bridge per seed). This runbook covers the Phase 1 pilot; Phase 2 reuses the
same commands with more seeds once the pilot gates pass and the amendment's
pilot section is confirmed.

```text
box:        8x H100 80GB SXM (Vast.ai or Prime Intellect), Ubuntu 22.04, CUDA 12
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

About 2 hours of training on 8 GPUs; plan 5 to 6 box hours for setup, smoke,
evals and export. At the September 2026 Vast price (8x H100 at about $25/h)
that is $130 to $220. Phase 2 (more seeds, controls, frontier) is estimated at
about 38 training GPU-hours.

## 0. Before Renting Anything ($0)

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

## 1. Constants On The Box

```bash
export REPO_ROOT=$HOME/Warm-starting-TTT-E2E
export DATA_ROOT=$HOME/ttt-e2e-data
export DCLM_ROOT=$DATA_ROOT/dclm_filter_8k
export BOOKS_ROOT=$DATA_ROOT/books3
export PAPER_RUN_ID=revision_v3_widthpreserving_v1
export EXP_FOLDER=revision_v3_widthpreserving_v1
export HF_125M_RESULTS_REPO=Luxel/ttt-e2e-125m-results
export HF_HUB_DISABLE_XET=1
export REVISION_V2_EXPECTED_JAX_DEVICES=8
# S0 full attention at 32K OOMed during XLA autotuning without this (June 2026).
export XLA_FLAGS=--xla_gpu_autotune_level=3
```

Keep `.env.hf` and `.env.backblaze` untracked on the box.

## 2. Clone And Bootstrap

```bash
git clone https://github.com/lmmontoya-ai/Warm-starting-TTT-E2E.git "$REPO_ROOT"
cd "$REPO_ROOT" && git checkout release-hardening
scripts/78_bootstrap_revision_v2_prime_runtime.sh "$REPO_ROOT"
```

Expected: `backend=gpu device_count=8 local_device_count=8`. Confirm the
checkout contains Amendment 1 (`grep -c "Amendment 1" PREREGISTRATION_REVISION_V3.md`).

## 3. Restore The Two Training Parents

Both pretrain seeds are valid; everything downstream of them is retrained.

```bash
cd "$REPO_ROOT"
for spec in \
  "S0_PRETRAIN_FA_125M pretrain-125m-fa" \
  "S3_PRETRAIN_E2E_125M pretrain-125m-e2e"
do
  set -- $spec
  uv run --with huggingface_hub python scripts/46_restore_stage_from_hf.py \
    --repo-id "$HF_125M_RESULTS_REPO" \
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

## 4. Stage Tokenized Data And Fingerprints

```bash
# About 40 GB: dclm train 10.4, dclm val 20.4, books train 0.8, books val 8.4.
uv run --exact python scripts/28_fetch_b2_dataset.py \
  --env-file .env.backblaze --dest-root "$DATA_ROOT" \
  --b2-prefix ttt-e2e-datasets/paper_budget_125m_val-full \
  --datasets dclm_filter_8k,books3 --splits train,val
for dataset in dclm_filter_8k books3; do
  for split in train val; do
    uv run --exact python scripts/13_dataset_fingerprint.py \
      --dataset-id "$dataset" --path "$DATA_ROOT/$dataset" --split "$split"
  done
done
```

## 5. Restore Audit On The Box

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

## 6. Smoke In Its Own Folder

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
    --seeds 1 --variant "lr$lr" --adapt-peak-lr "$lr" \
    --export-to-hf --hf-repo-id "$HF_125M_RESULTS_REPO"
done
```

An LR-only variant tags the bridge and S2 (`adapt-125m-e2e-8K-from-fa-lr1e-3-seed001`,
`ext-125m-e2e-32K-from-fa-bridge-lr1e-3-seed001`) but not S2-minus, which does
not depend on the bridge LR; the three launches share one
`ext-125m-e2e-32K-from-fa-direct-seed001`. A `--prime-post-norm-init` variant
tags S2-minus as well, because it changes that run.

Every run writes `restore_report.json`, the resolved LR schedule under
`run_started.lr_schedule` in `events.jsonl`, and a ladder summary under
`reports/revision_v3/$PAPER_RUN_ID/`.

## 8. Checkpoint-Based 64-Batch Evaluation

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

## 9. Pilot Gates

Read against the pilot section of Amendment 1:

1. every run passed the trainer gates;
2. S0 Books32K within `0.2` of the re-extended S3; if not, stop and diagnose;
3. pick the bridge peak LR by the committed rule;
4. write the pilot memo under `reports/revision_v3/$PAPER_RUN_ID/` before any
   Phase 2 launch.

## 10. Tear Down

1. Verify every stage exported to HF (`hf_export.status == succeeded` in the
   ladder summaries).
2. Copy `reports/revision_v3/` and the `restore_report.json` files off the box.
3. Destroy the instance and confirm with `vastai show instances --raw` (or
   `prime pods list`).
4. Record the session cost in `docs/REVISION_V2_COST_LEDGER.md` under a
   revision-v3 heading.
