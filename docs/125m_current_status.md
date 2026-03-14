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
  - completed diagnostic gate `s3_diag`
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

What is now classified:

- faithful `8x H200`, pure data-parallel `8 GPU (8:1)` passed on the **reference** runtime
- faithful `8x H200`, pure data-parallel `8 GPU (8:1)` passed on the **local** runtime
- decisive artifacts:
  - `artifacts/s3_scaling_diag/reference_125m_s3_pretrain_smoke.result.json`
  - `artifacts/s3_scaling_diag/local_125m_s3_scaling_probe/summary.json`
  - `reports/paper/protocol_r_125m_main_v1/split_batches/s3_diag.json`

Observed faithful `8:1` result:

- reference smoke:
  - status `succeeded`
  - first-step completion `true`
  - checkpoint written `true`
  - latest step `1`
- local smoke:
  - status `passed`
  - first metric seen `true`
  - checkpoint written `true`
  - latest step `1`
  - peak GPU memory `109193 MiB`

Interpretation:

- the old blocker was **not** generic `8 GPU` support
- the faithful pure data-parallel production topology is viable on both codebases
- the bad behavior we saw earlier belongs to exploratory state-parallel topologies, not to the faithful paper path
- `s3_ladder` is now unblocked by `s3_diag`

## Alignment Notes

For `S3_PRETRAIN_E2E_125M`, the current reproduction is materially aligned with the reference on the parts that matter most:

- experiment config matches the reference snapshot
- training config matches the reference snapshot
- model sharding structure matches the reference layout
- train/eval loop structure remains close to the reference loop stack

Important remaining caveats:

- local train/eval now construct sharded batches at the iterator boundary, matching the reference structure more closely instead of feeding the compiled step through the old reshape-based batch path
- faithful `8 GPU (8:1)` now has a real reference-vs-local classification artifact on `8x H200`
- the `S1` reference diagnostic helper is a constructed SWA-equivalent probe, because the reference snapshot does not ship a dedicated `125m/pretrained/ext-125m-swa-32K-from-fa` config

So the next engineering target changes to:

- launch the canonical `s3_ladder`
- then verify:
  - `S3_PRETRAIN_E2E_125M`
  - `S3_125M`
  complete with parity eval plus canonical HF export

## Ladder Status

`IS_DONE` below means the stage has a completed canonical checkpoint/export lineage that is durably present in HF under `protocol_r_125m_main_v1/stages/...`.

Important:

- split-batch JSON summaries alone are not sufficient proof of canonical completion
- those summaries can reflect dry-runs, gate-only invocations, or non-durable local execution records
- the durable authority for completion is the canonical HF export path
- canonical also requires:
  - successful parity eval in the stage run tree
  - a matching `hf_export_manifest.json`
  before the export is treated as paper-ready

| NAME | IS_DONE | RESULT | INHERITS MODEL FROM |
|---|---|---|---|
| `S0_PRETRAIN_FA_125M` | `yes` | recovered-success; canonical DCLM `8K` eval `3.484375`, Books3 `8K` recheck `3.359375` | `scratch` |
| `S0_125M` | `yes` | recovered-success under Protocol R; final train `loss_ce 6.4375`, Books3 `32K` recheck `6.375` | `S0_PRETRAIN_FA_125M` |
| `S1_125M` | `no` | split-batch diagnostics/runs were recorded, but no canonical HF export exists yet under `protocol_r_125m_main_v1/stages/S1_125M/...` | `S0_PRETRAIN_FA_125M` |
| `S2_ADAPT_125M` | `no` | gate/runtime evidence exists, and `>80GB/GPU` on H100 is frozen, but no canonical HF export exists yet | `S0_PRETRAIN_FA_125M` |
| `S2_125M` | `no` | split-batch run/export report exists, but no canonical HF export exists yet | `S2_ADAPT_125M` |
| `S3_PRETRAIN_E2E_125M` | `no` | faithful config is aligned (`global_batch_size=64`) and `s3_diag` now passed on faithful `8x H200` pure data-parallel `8:1`, but canonical HF export does not exist yet | `scratch` |
| `S3_125M` | `no` | split-batch run/export report exists, but no canonical HF export exists yet | `S3_PRETRAIN_E2E_125M` |

## Recommended Next Step

For a world-class paper, the next spend should go to one of these:

1. canonically run the `H200` subladder `h200_a`:
   - `S1_125M`
   - `S2_ADAPT_125M`
   - `S2_125M`
2. launch the canonical `S3` subladder, because `s3_diag` has already classified the faithful `8 GPU` S3 path as:
   - `reference_pass_local_pass`
   - artifact: `reports/paper/protocol_r_125m_main_v1/split_batches/s3_diag.json`
3. for the canonical `S3` subladder:
   - `S3_PRETRAIN_E2E_125M`
   - `S3_125M`
