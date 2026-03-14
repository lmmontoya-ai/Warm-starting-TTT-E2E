# 125M Dual-Protocol Note

This note defines the paper-facing protocol split for the `125M` in-family ladder.

Current run status, recovered results, and frozen operational boundaries are tracked in:
- `docs/125m_current_status.md`

## Protocol F: Faithful Feasibility

Protocol F keeps the published `125M` `32K` extension recipe unchanged and treats hardware feasibility as an empirical result.

Canonical evidence:
- local parity runtime `32K` FA extension OOM on `8x H100 80GB`
- local parity runtime `32K` FA extension OOM on `8x A100 80GB`
- reference `32K` FA extension OOM on `8x H200 141GB`

Decisive artifact:
- `artifacts/oom_diagnosis/reference_125m_32k_fa_h200/reference_125m_32k_fa_smoke.log`

Interpretation:
- the `125M` `32K` FA extension is a real hardware-feasibility problem under the faithful recipe
- this is not just a local-runtime bug

Protocol F is therefore a paper contribution in its own right: a faithful reproducibility finding.

## Protocol R: Revised Matched Protocol

Protocol R introduces the smallest explicit training-method change needed to make the full `125M` ladder runnable.

Allowed deviation:
- reduce `32K` extension `training.global_batch_size`

Frozen constraints:
- `seq_length=32768`
- `suffix_len=0`
- `force_flash=true`
- same restored FA `8K` seed
- same datasets
- same optimizer settings
- same runtime architecture

Reference-first search order:
- valid `8`-way data-parallel candidates: `32`, `24`, `16`, `8`
- invalid for the current `8`-way sharding topology: `12`, `4`

Selection rule:
- choose `B* = highest batch size` for which the **reference** `2`-step smoke completes without OOM and reports sane CE

Frozen `125M` Protocol R result:
- `B* = 8`
- evidence:
  - `32`: OOM
  - `24`: OOM
  - `16`: OOM
  - `8`: reference train-fit passed on `8x H200 141GB`
- canonical artifacts:
  - `artifacts/protocol_r/reference_batch_search_20260312T231848Z/reference_protocol_r_batch_search.json`
  - `reports/paper/protocol_r_local_gate_h200_b8_clean/protocol_r_local_gate_summary.json`

Token-budget preservation:
- once `B*` is selected, preserve the extension token budget by scaling extension steps:
- `new_ext_steps = original_ext_steps * (32 / B*)`

Apply the revised extension schedule to:
- `S0_125M`
- `S1_125M`
- `S2_125M`
- `S3_125M`

Do not change:
- `S0_PRETRAIN_FA_125M`
- `S2_ADAPT_125M`
- `S3_PRETRAIN_E2E_125M`

## Execution Surfaces

Reference-first batch search:
- `scripts/44_search_reference_125m_32k_fa_batch.py`

Local runtime gate for the chosen `B*`:
- `scripts/45_gate_local_125m_protocol_r.py`

Full ladder launcher with extension-only batch override and token-budget preservation:
- `scripts/35_run_125m_ladder.py`

Relevant flags:
- `--protocol revised`
- `--ext-global-batch-size <B*>`
- `--preserve-ext-token-budget`
- `--base-ext-global-batch-size 32`

## GPU-Use Policy

To keep the study efficient and publication-grade:

- use **spot** GPUs only for:
  - faithful feasibility probes
  - reference-first batch search
  - local Protocol R gates
- use **on-demand** GPUs for:
  - the final full `125M` ladder under the frozen protocol
- stop the reference-first search at the **first passing batch size** in descending order
- avoid unnecessary checkpoint churn during probes:
  - faithful/reference smokes run for `2` steps
  - local benchmark gates run for `6–8` steps
  - no periodic checkpoint saves during those probes
- export each completed ladder stage to HF immediately during the real run
- keep only the latest complete checkpoint plus one fallback locally

This maximizes GPU time spent on scientific signal rather than bootstrap, repeated compilation, or redundant checkpoint storage.

## Blackwell Branch

If `GB200/B200` hardware becomes available, run exactly one faithful reference gate before Protocol R:
- reference `S0_125M` `32K` FA smoke
- original config, no method changes

Only if the faithful reference gate passes should a faithful local gate and faithful full ladder be attempted.

We do **not** assume the authors used `GB200/B200` without verification.

## Frozen Execution Decision

The current `125M` full ladder should now run under:
- `--protocol revised`
- `--ext-global-batch-size 8`
- `--preserve-ext-token-budget`
- `--base-ext-global-batch-size 32`

This keeps the extension token budget matched while changing only one training variable:
- `32K` extension `training.global_batch_size`

## Split Execution Decision

The `125M` Protocol R ladder runs under one canonical lineage:

- canonical `paper_run_id = protocol_r_125m_main_v1`
- canonical `exp_folder = protocol_r_125m_main_v1`
- HF is the durable source of truth between hardware batches

Current canonical execution shape:

- completed prefix:
  - `S0_PRETRAIN_FA_125M`
  - `S0_125M`
- `H200` subladder `h200_a`:
  - `S1_125M`
  - `S2_ADAPT_125M`
  - `S2_125M`
- `S3` diagnosis gate `s3_diag`
- `S3` subladder `s3_ladder`:
  - `S3_PRETRAIN_E2E_125M`
  - `S3_125M`

Frozen lineage:

- `S0_125M <- S0_PRETRAIN_FA_125M`
- `S1_125M <- S0_PRETRAIN_FA_125M`
- `S2_ADAPT_125M <- S0_PRETRAIN_FA_125M`
- `S2_125M <- S2_ADAPT_125M`
- `S3_PRETRAIN_E2E_125M <- scratch`
- `S3_125M <- S3_PRETRAIN_E2E_125M`

Important:
- `S1_125M` must **not** resume from `S0_125M`
- use the split-batch runner `scripts/47_run_125m_split_batch.py`
- use the HF restore utility `scripts/46_restore_stage_from_hf.py`
- use `scripts/50_run_reference_125m_32k_swa_smoke.py` and `scripts/51_probe_local_125m_32k_swa.py` before any full `S1_125M` rerun
- use `scripts/56_run_reference_125m_s3_pretrain_smoke.py` and `scripts/57_probe_local_125m_s3_scaling.py` before any canonical `S3` rerun

Frozen `S3` interpretation:
- `S3_PRETRAIN_E2E_125M` remains on the faithful config (`global_batch_size = 64`)
- the reference snapshot does not expose a special `125M` multi-GPU workaround in its README or configs
- the next decision point is whether `s3_diag` classifies the `8 GPU` path as:
  - `reference_pass_local_pass`
  - `reference_pass_local_fail`
  - `reference_fail_local_fail`
