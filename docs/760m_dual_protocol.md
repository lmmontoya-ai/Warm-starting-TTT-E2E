# 760M Author-Seed Dual-Protocol Note

This note defines the paper-facing protocol split for the `760M` author-seeded
ladder.

The `760M` study uses the author-shared `8K` checkpoints as fixed Stage-A
seeds:

- `760m_fa` seeds `S0_760M`, `S1_760M`, and `S2_ADAPT_760M`
- `760m_e2e` seeds `S3_760M`

Registry stage ids remain:

- `S0`
- `S1`
- `S2_ADAPT`
- `S2`
- `S3`

## Protocol F: Faithful Feasibility

Protocol F keeps the published `760M` recipe unchanged on `8x H200`.

Faithful batch sizes:

- `S0_760M`, `S1_760M`, `S2_760M`, `S3_760M`
  - `seq_length = 32768`
  - `global_batch_size = 32`
  - `total_steps = 725`
- `S2_ADAPT_760M`
  - `seq_length = 8192`
  - `global_batch_size = 64`
  - `total_steps = 2900`

Current feasibility evidence on `8x H200`:

- faithful `S0` `32K` `gb=32`: pre-step OOM
- faithful `S1` `32K` `gb=32`: pre-step OOM
- faithful `S3` `32K` `gb=32`: pre-step OOM
- faithful `S2_ADAPT` `8K` `gb=64`: pre-step OOM
- reduced `S2_ADAPT` `gb=32`: still failed

Decisive synced artifacts:

- `artifacts/vast_760m_author_smokes_20260316/reports/gb8_summary.json`
- `artifacts/vast_760m_author_smokes_20260316/reports/s2_adapt_summary.json`
- `artifacts/vast_760m_author_smokes_20260316/reports/s2_adapt_gb32_summary.json`

Interpretation:

- the author seeds are compatible with the local runtime
- the faithful `760M` long-context and bridge batch sizes are not feasible on
  `8x H200`
- this is a hardware-feasibility boundary for the faithful `760M` author-seeded
  ladder on the currently available surface

## Protocol R: Revised Matched Protocol

Protocol R introduces the smallest explicit training-method change needed to
make the `760M` author-seeded ladder runnable on `8x H200`.

Allowed deviation:

- reduce `training.global_batch_size`

Frozen constraints:

- same author-shared `8K` seeds
- same datasets
- same architecture
- same optimizer settings
- same context lengths
- same train modes
- same local runtime implementation

Passing reduced-batch gate:

- `global_batch_size = 8`

Current `2`-step smoke evidence at `gb=8`:

- `S0`: pass, checkpoint written
- `S1`: pass, checkpoint written
- `S2_ADAPT`: pass, checkpoint written
- `S3`: pass, checkpoint written

So the first runnable `8x H200` protocol is:

- `S0_760M`: `gb=8`
- `S1_760M`: `gb=8`
- `S2_ADAPT_760M`: `gb=8`
- `S2_760M`: `gb=8`
- `S3_760M`: `gb=8`

### Token-Budget Preservation

Protocol R preserves each stage token budget by scaling total steps with the
ratio:

- `new_steps = original_steps * (original_global_batch_size / revised_global_batch_size)`

Frozen `760M` Protocol R result:

- revised `global_batch_size = 8`
- extension stages:
  - original `725` steps at `gb=32`
  - revised `2900` steps at `gb=8`
- bridge stage:
  - original `2900` steps at `gb=64`
  - revised `23200` steps at `gb=8`

Apply the revised schedule to:

- `S0_760M`
- `S1_760M`
- `S2_ADAPT_760M`
- `S2_760M`
- `S3_760M`

Do not change:

- model architecture
- dataset mapping
- sequence lengths
- optimizer definitions
- seed provenance

## Execution Surfaces

Author checkpoint fetch / mirror:

- `scripts/30_fetch_author_orbax.py`
- `scripts/33_download_author_orbax_from_hf.py`

Checkpoint audit:

- `scripts/31_probe_author_orbax.py`
- `scripts/64_audit_author_orbax_local_runtime.py`

Reduced-batch feasibility gate:

- `scripts/65_run_760m_author_seed_smokes.py`

Full revised ladder launcher:

- `scripts/66_run_760m_author_seed_ladder.py`

## Frozen Execution Decision

The current `760M` author-seeded program should run under:

- `paper_run_id = protocol_r_760m_author_seed_v1`
- `exp_folder = protocol_r_760m_author_seed_v1`
- `training.global_batch_size = 8`
- `effective_ext_steps = 2900`
- `effective_adapt_steps = 23200`

This keeps the token budgets matched while changing only one training variable:

- `training.global_batch_size`

Execution will be staged.

### Stage C1: Core Comparison

Run first:

- `S2_ADAPT_760M`
- `S2_760M`
- `S3_760M`

Purpose:

- answer the main scale-up question directly
- compare warm-started TTT-E2E against scratch TTT-E2E at `760M`
- defer non-decisive controls until after the main comparison is secured

Measured training-only ETA on `8x H200`:

- about `10.12` wall-clock hours
- about `80.94` GPU-hours

### Stage C2: Deferred Controls

Run after Stage C1:

- `S1_760M`
- `S0_760M`

Purpose:

- complete the control surface
- test whether the `125M` FA/SWA ordering also holds at `760M`
- strengthen the paper, but not gate the primary `760M` claim

Measured training-only ETA on `8x H200`:

- about `3.83` wall-clock hours
- about `30.64` GPU-hours

## Canonical Lineage

Frozen lineage for the revised `760M` run:

- `S0_760M <- author 760m_fa`
- `S1_760M <- author 760m_fa`
- `S2_ADAPT_760M <- author 760m_fa`
- `S2_760M <- S2_ADAPT_760M`
- `S3_760M <- author 760m_e2e`

Important:

- `S1_760M` must not resume from `S0_760M`
- `S2_760M` must resume from `S2_ADAPT_760M`
- the faithful registry remains untouched; Protocol R is expressed in the
  launcher and protocol manifest, not by mutating the faithful stage specs

## Launcher Use

The revised ladder launcher now supports execution by tranche:

- core comparison:
  - `python scripts/66_run_760m_author_seed_ladder.py --phase core ...`
- deferred controls:
  - `python scripts/66_run_760m_author_seed_ladder.py --phase controls ...`
- full ladder in one pass:
  - `python scripts/66_run_760m_author_seed_ladder.py --phase all ...`

Tables and figures should only be generated from the full `S0/S1/S2/S3`
comparison set. For `core` and `controls`, the launcher skips those steps
automatically.
