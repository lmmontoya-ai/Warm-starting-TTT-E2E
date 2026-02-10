# Pretrained-Initialization Research Plan for TTT-E2E Efficiency

## Objective
Evaluate whether initializing TTT-E2E from a pretrained non-TTT model can keep long-context quality while materially reducing total training wall-clock.

## What Was Verified
1. The paper explicitly flags training latency as a major limitation and proposes two future directions: custom kernels and initializing TTT-E2E from a pretrained Transformer without TTT (Section 3.7, pp. 16-17 of `reference/TTT-E2E.pdf`).
2. In code, meta mode (`train_mode=meta`) triggers inner-loop updates and gradients-of-gradients (`reference/e2e/ttt/model/transformer.py`, `reference/e2e/ttt/model/loop.py`).
3. Full-attention pretraining already exists as configs (`configs/experiment/*/pretrain/*-fa.yaml`) and can serve as a warm-start source.
4. Checkpoint loading supports parameter-only restore and partial structure matching (`training.load_part=params`, `unify_dict_with_eqx_module`), which is exactly what warm-starting needs (`reference/e2e/ttt/train.py`, `reference/e2e/ttt/infra/checkpoint.py`).
5. E2E configs update only prime MLP params in suffix blocks in the inner loop (`spec_inner` with `suffix_blocks.feed_forward_prime`) and use SWA + mini-batch TTT.

## Practical Constraints (Critical)
1. The root repo is not yet runnable for experiments because there is no local `ttt/` package in the main repo; runnable code currently sits in `reference/e2e/ttt/`.
2. The current machine environment is not suitable for the original training stack (CUDA-only JAX dependencies and platform wheel constraints), so execution requires a Linux GPU environment.
3. Per `AGENTS.md`, implementation must be re-implemented outside the reference snapshot (no direct copying into `og_repo`/reference snapshot).

## Core Hypotheses
1. Warm-starting from non-TTT pretraining will reduce total training time substantially because most short-context compute can avoid gradients-of-gradients.
2. A short TTT adaptation phase at 8K before long-context extension will recover most of the quality gap versus training TTT-E2E from scratch.
3. Directly jumping from non-TTT pretraining to long-context TTT extension may underperform versus adding an 8K TTT adaptation bridge.

## Experiment Matrix (Recommended First Pass: 760M)
Use 760M first because configs are already complete and cheap enough for iteration.

1. Baseline B1 (paper-style TTT-E2E):
- `pretrain-760m-e2e` -> `ext-760m-e2e-32K`

2. Baseline B2 (non-TTT Transformer):
- `pretrain-760m-fa` -> `ext-760m-fa-32K`

3. Proposed P1 (warm-start with adaptation):
- Stage A: `pretrain-760m-fa`
- Stage B: 8K TTT adaptation initialized from Stage A (`train_mode=meta`, SWA, prime=True, `load_part=params`)
- Stage C: `ext-760m-e2e-32K` initialized from Stage B

4. Proposed P2 (warm-start without adaptation):
- Stage A: `pretrain-760m-fa`
- Stage C directly: `ext-760m-e2e-32K` initialized from Stage A

5. Proposed P3 (SWA non-TTT warm-start ablation):
- Stage A: SWA + `train_mode=pretrain` (new config)
- Stage B/C same as P1

6. Adaptation budget sweep (Stage B total steps):
- 5%, 10%, 20% of original E2E pretraining steps

## Fairness Protocol
Report two comparisons for each variant.

1. Token-matched comparison:
- Keep total trained tokens equal to B1.

2. Wall-clock-matched comparison:
- Keep total GPU-hours equal to B1.

This separates pure efficiency gains from budget reallocation effects.

## Metrics and Success Criteria
1. Quality:
- Books validation loss at 32K (primary)
- Loss breakdown by token index (secondary)
- Optional NIAH check for regression sanity

2. Efficiency:
- Total GPU-hours to reach target 32K loss
- Time per 1K training tokens at 8K and 32K

3. Success gate for this direction:
- >=30% reduction in total training wall-clock at <=0.01 loss delta versus B1 at 32K.

## Why This Should Work
At 8K, paper-reported training latency shows TTT-E2E is much slower than regular Transformer training. If most 8K tokens are trained in non-TTT mode and only a smaller suffix is trained with TTT-E2E, total runtime should drop sharply while preserving the TTT-specific long-context adaptation phase.

## Implementation Plan in This Repo (Outside Reference Snapshot)
1. Re-implement runnable `ttt/` package in main repo (config, dataloader, model, loop, optimizer, checkpoint) with source notes and rationale per module.
2. Add warm-start experiment configs (new group, e.g., `configs/experiment/760m/pretrained/`).
3. Add explicit adaptation-stage configs (8K meta stage initialized from FA checkpoint).
4. Add a small experiment tracker script that records tokens, wall-clock, and resulting checkpoints in one table.
5. Add reproducibility docs with exact launch commands and resume/load flags.

## Architecture-Change Track (Phase 2)
Only after P1/P2 are stable.

1. External pretrained checkpoint import (if desired).
2. If architecture mismatch exists (e.g., head layout, vocab, norm choices), add explicit conversion adapters.
3. Re-run the same matrix with one external checkpoint to test transfer value beyond in-family warm-start.

## Immediate Next Actions
1. Stand up a runnable local training package outside `reference/`.
2. Implement P1/P2 configs and launch scripts for 760M.
3. Run a short pilot (e.g., 1-5% schedule) to validate checkpoint compatibility and adaptation behavior before full runs.
