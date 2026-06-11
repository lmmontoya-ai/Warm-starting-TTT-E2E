# Revision V2 E3 Hardware Feasibility Notes

Status as of 2026-06-11 18:15 UTC: E3/continuation scaffolding is ready, but
the remaining blocker is a suitable GPU topology for the bridge-budget frontier.
No Prime pods were running after the checks below.

## Requirement

E3 bridge arms run the 125M TTT-E2E bridge at 8K context with global batch 64.
The production run needs a topology that:

- fits the 8K meta-training bridge without gradient accumulation,
- runs materially faster than the rejected A100/accumulation attempts,
- preserves the committed in-repo JAX/runtime stack, and
- keeps S2 bridge-budget arms and S2-minus continuation on one documented
  execution profile.

## Tried Topologies

| Date | Provider / GPU | Result | Decision |
|---|---|---|---|
| 2026-06-11 | Prime 2x H100 80GB PCIe | Extension/resume smoke passed. The 40% bridge arm OOMed without accumulation. Accumulation values that fit were around 64-103 seconds/step in smoke runs. | Reject for full E3; too slow/costly. |
| 2026-06-11 | Prime 8x A100 80GB PCIe | Runtime bootstrap, parent restore, data fingerprinting, and dry-run passed. Three-step no-accum bridge smoke fit, but step time was roughly 100 seconds/step. | Reject for full E3; too slow/costly. |
| 2026-06-11 | Prime 8x B300 262GB SXM6 spot | Runtime bootstrap saw 8 JAX GPU devices with driver 580.126.09. Canonical parents restored and all four dataset fingerprints matched. Dry-run passed. Training failed under pinned `jax==0.5.3`: default path hit `ptxas` errors because compute capability 10.3 was treated as `sm_101`; with `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`, simple BF16 matmul succeeded but the bridge smoke failed in cuDNN frontend with `No execution plans support the graph`. | Reject under current reproducibility constraints; using B300 would require dependency/runtime surgery. |
| 2026-06-11 | Prime 8x A100 80GB SXM4 | Availability listed Vultr DE/US candidates at `$22.40/hr`, but create attempts failed before allocation (`HTTP 400` for DE; `No valid GPU configuration found` for US). | No usable pod allocated; keep as conditional candidate only if Prime creation succeeds later and a fresh timing smoke passes. |
| 2026-06-11 | Vast 8x H200 141GB | Runtime bootstrap passed with 8 JAX GPU devices. Canonical parents restored once, all four dataset fingerprints matched, and E3 dry-run validity checks passed. A 3-step 40% bridge smoke fit at global batch 64 with `n_data_parallel=8`, but took 228.7s wall total; after compile, step 2 landed about 42.2s after step 1. At the observed Vast price (`~$27.33/hr`), a full E3 frontier would materially exceed the Phase-2 `$120` ceiling. | Reject this specific Vast H200 offer for production E3; it is a successful fit/validity probe but not cost-feasible. Instance was destroyed after copying smoke summaries locally. |

## B300 Details

The B300 smoke pod was `86615544202641c38c2e80b32555ba32`
(`revision-v2-e3-b300-smoke`) and was terminated after the failed smoke.

Positive checks before rejection:

- OS saw 8x `NVIDIA B300 SXM6 AC`, 275040 MiB each.
- `scripts/78_bootstrap_revision_v2_prime_runtime.sh` completed with
  `jax=0.5.3`, `backend=gpu`, and `device_count=8`.
- Parent checkpoints restored:
  - `pretrain-125m-fa`, latest step 4799.
  - `ext-125m-e2e-32K-from-fa-direct-seed001`, latest step 479.
- Canonical dataset fingerprints matched:
  - `dclm_filter_8k/train`: 2520000000 tokens,
    `abf077fd05f8796ac45d463f3dad707649b26dee1772d98790978781f2131213`.
  - `dclm_filter_8k/val`: 5000017938 tokens,
    `6cca144b79f43edd57261e7bd7a79d9a7bd26e304e6c0d32754af90f0f9ed176`.
  - `books3/train`: 126000000 tokens,
    `3477192d72a699d8868bcab5e6eabe6ef04032e0e27799c7b0aefbb975ea7c9e`.
  - `books3/val`: 2000168321 tokens,
    `cb3a86e8899f38dcffa4571be7f6343e9aa2a557ba9780af3a823c02a833e7e6`.
- E3 dry-run validity checks passed with the 8-device deploy profile.

Failure mode:

```text
Unknown compute capability 10.3. Defaulting to telling LLVM that we're compiling for sm_101
ptxas ... error: Instruction 'tcgen05.alloc' not supported on .target 'sm_101'
```

With Triton GEMM disabled:

```text
jaxlib.xla_extension.XlaRuntimeError:
INTERNAL: [cudnn_frontend] Error: No execution plans support the graph.
```

Because the project is using a pinned checkpoint/eval/training stack for
revision-v2 reproducibility, the B300 path is not acceptable for production E3
without a separate dependency-change preregistration and validation pass.

## Current Recommendation

Wait for a topology that is both validated and cost-feasible under the Phase-2
budget. The next production candidate must pass the same 3-step bridge smoke
and extrapolate to the full frontier inside the session cap before launching
the 40% arm.

Before any next production launch, use the E3 runner's topology overrides to
smoke the bridge stage itself across candidate meshes (`8:1`, `4:2`, `2:4`,
and `1:8` via `--n-data-parallel` / `--n-state-parallel`). The previous Vast
H200 probe used `8:1`; full E3 should launch only if one of these committed
mesh smokes materially changes the step-time projection.

- 8x H200 141GB remains technically preferred only if the hourly price and
  smoke timing jointly fit the cap; the rejected Vast offer did not.
- 8x H100 80GB is acceptable if available and a no-accumulation bridge smoke is
  materially faster than the rejected attempts.
- 8x A100 80GB SXM4 only if a short bridge smoke demonstrates a step time that
  fits the budget and Prime can actually allocate it; the PCIe A100 result
  should not be repeated.

Do not run production E3 on:

- 2x H100 with bridge accumulation.
- 8x A100 PCIe.
- 8x B300 under the current pinned JAX/runtime stack.
