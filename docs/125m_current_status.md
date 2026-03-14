# 125M Current Status

This note freezes the current `125M` ladder status for paper work and future runs.

Current execution shape:
- completed canonical prefix:
  - `S0_PRETRAIN_FA_125M`
  - `S0_125M`
- pending `H200` subladder `h200_a`:
  - `S1_125M`
  - `S2_ADAPT_125M`
  - `S2_125M`
- pending `S3` path:
  - diagnostic gate `s3_diag`
  - canonical subladder `s3_ladder`:
    - `S3_PRETRAIN_E2E_125M`
    - `S3_125M`

## Validated Results

### `S0_PRETRAIN_FA_125M`

Recovered and scientifically valid.

- Canonical DCLM `8K` parity eval:
  - `loss_ce = 3.484375`
  - artifact: `artifacts/vast_eval_bundle_20260314/experiments/pretrain-125m-fa/eval_manifest.json`
- Books3 `8K` recheck after the restore fix:
  - `loss_ce = 3.359375`
  - artifact: `artifacts/restore_fix_validation_20260314/s0_pretrain_books3_8k_fix.json`

### `S0_125M`

Recovered under Protocol R and scientifically valid again after the restore fix.

- Protocol R stage run:
  - final step `479`
  - final observed train `loss_ce = 6.4375`
  - artifact: `artifacts/vast_h200_s0_20260313/reports/h200_s0.json`
  - training trace: `artifacts/vast_h200_s0_20260313/experiments/ext-125m-fa-32K/metrics.jsonl`
- Books3 `32K` recheck after the restore fix:
  - `loss_ce = 6.375`
  - artifact: `artifacts/restore_fix_validation_20260314/s0_125m_canonical_recheck_fix.json`

## Frozen Operational Boundaries

### `S2_ADAPT_125M`

Freeze this as an operational fact:

- `S2_ADAPT_125M` does not fit on `8x H100 80GB`
- decisive artifact:
  - `artifacts/h100_b_gate_h100_20260314/h100_b_gates_fix_v3.log`

Interpretation:

- this stage requires `>80GB/GPU` under the current faithful `125M` settings
- this is a hardware-envelope fact, not a reason to silently change methodology

### `S3_PRETRAIN_E2E_125M`

Do not freeze "does not fit on H100" as a fact yet.

What is frozen:

- the faithful config is still `global_batch_size = 64`
- local config matches the reference snapshot exactly:
  - `configs/experiment/125m/pretrain/pretrain-125m-e2e.yaml`
  - `ttte2e_reference/e2e/configs/experiment/125m/pretrain/pretrain-125m-e2e.yaml`
  - `configs/training/125m/pretrain-8K.yaml`
  - `ttte2e_reference/e2e/configs/training/125m/pretrain-8K.yaml`

What remains unresolved:

- the local runtime scales poorly beyond small multi-GPU topologies for this meta-training path
- probes were run and synced to:
  - `artifacts/s3_scaling_diag_20260314/`

Observed probe status:

- `1 GPU`: checkpointed successfully
- `2 GPU`: checkpointed successfully
- `4 GPU`: checkpointed successfully in the probe, but compile/save overhead is already much worse
- `8 GPU (state-parallel probe)`: unresolved as a clean production-ready path

Interpretation:

- the remaining issue is runtime scaling/compile behavior in the local `jax_train` meta path
- it is not a config mismatch against the paper snapshot
- the next official gate is now `s3_diag`, which must classify the faithful `8 GPU` path before any canonical `S3` rerun

## Alignment Notes

For `S3_PRETRAIN_E2E_125M`, the current reproduction is materially aligned with the reference on the parts that matter most:

- experiment config matches the reference snapshot
- training config matches the reference snapshot
- model sharding structure matches the reference layout
- train/eval loop structure remains close to the reference loop stack

Important remaining caveats:

- local training still uses the in-repo `to_data_parallel_batch(...)` path in `ttt/jax_runtime/sharding.py`, while the reference train path constructs the sharded batch inside `_make_train_iterator(...)` in `ttte2e_reference/e2e/ttt/train.py`
- the `S1` reference diagnostic helper is a constructed SWA-equivalent probe, because the reference snapshot does not ship a dedicated `125m/pretrained/ext-125m-swa-32K-from-fa` config

So the next engineering target remains:

- reduce compile/runtime scaling cost in:
  - `ttt/jax_runtime/train.py`
  - `ttt/jax_runtime/loop.py`
  - `ttt/jax_runtime/sharding.py`

before spending more on blind large-pod runs.

## Ladder Status

`IS_DONE` below means the stage has a completed canonical checkpoint/export lineage that is durably present in HF under `protocol_r_125m_main_v1/stages/...`.

Important:

- split-batch JSON summaries alone are not sufficient proof of canonical completion
- those summaries can reflect dry-runs, gate-only invocations, or non-durable local execution records
- the durable authority for completion is the canonical HF export path
- canonical also requires parity eval artifacts in the stage run tree before the export is treated as paper-ready

| NAME | IS_DONE | RESULT | INHERITS MODEL FROM |
|---|---|---|---|
| `S0_PRETRAIN_FA_125M` | `yes` | recovered-success; canonical DCLM `8K` eval `3.484375`, Books3 `8K` recheck `3.359375` | `scratch` |
| `S0_125M` | `yes` | recovered-success under Protocol R; final train `loss_ce 6.4375`, Books3 `32K` recheck `6.375` | `S0_PRETRAIN_FA_125M` |
| `S1_125M` | `no` | split-batch diagnostics/runs were recorded, but no canonical HF export exists yet under `protocol_r_125m_main_v1/stages/S1_125M/...` | `S0_PRETRAIN_FA_125M` |
| `S2_ADAPT_125M` | `no` | gate/runtime evidence exists, and `>80GB/GPU` on H100 is frozen, but no canonical HF export exists yet | `S0_PRETRAIN_FA_125M` |
| `S2_125M` | `no` | split-batch run/export report exists, but no canonical HF export exists yet | `S2_ADAPT_125M` |
| `S3_PRETRAIN_E2E_125M` | `no` | faithful config is aligned (`global_batch_size=64`), but canonical HF export does not exist yet and multi-GPU runtime scaling remains unresolved | `scratch` |
| `S3_125M` | `no` | split-batch run/export report exists, but no canonical HF export exists yet | `S3_PRETRAIN_E2E_125M` |

## Recommended Next Step

For a world-class paper, do not spend more GPU money until one of these happens:

1. canonically run the `H200` subladder `h200_a`:
   - `S1_125M`
   - `S2_ADAPT_125M`
   - `S2_125M`
2. run `s3_diag` and classify the faithful `8 GPU` S3 path exactly as:
   - `reference_pass_local_pass`
   - `reference_pass_local_fail`
   - `reference_fail_local_fail`
3. only if `s3_diag` passes, launch the canonical `S3` subladder:
   - `S3_PRETRAIN_E2E_125M`
   - `S3_125M`
