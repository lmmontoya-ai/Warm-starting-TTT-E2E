# Revision V2 Preregistration

Status: active preregistration draft, 2026-06-08.

This document fixes the decision rules for the revision-v2 experiments before
new compute is interpreted. It complements `CANONICAL_RESULTS.md`: the ledger is
the historical source of truth, while this preregistration defines how new
revision-v2 gates are evaluated.

## Scope

The load-bearing scientific question is whether the 8K bridge contributes to
the 125M warm-start path beyond direct FA-to-TTT-E2E conversion plus 32K
extension.

The critical new condition is `S2_MINUS_125M`:

```text
FA seed -> TTT-E2E/SWA converted architecture -> Books32K 32K extension
```

It skips the 8K bridge. It must otherwise match `S2_125M` extension as closely
as possible.

## E1: Bridge-Isolation Loss Gate

### Primary Statistic

For paired seed `s`:

```text
delta_bridge_s = loss(S2_MINUS_s) - loss(S2_s)
```

All losses must come from checkpoint restore plus the same current float32
Books32K evaluation pipeline.

Interpretation:

- Positive `delta_bridge_s`: the bridge helped because S2-minus is worse.
- Negative `delta_bridge_s`: the bridge hurt because S2-minus is better.

### Seed Design

Run five paired seeds up front:

```text
S2_seed001       paired with S2_MINUS_seed001
S2_seed002       paired with S2_MINUS_seed002
S2_seed003       paired with S2_MINUS_seed003
S2_seed004       paired with S2_MINUS_seed004
S2_seed005       paired with S2_MINUS_seed005
```

Pairing means the same seed id, same extension configuration, same data roots,
same train/eval fingerprints, same eval batches, and the same intentional
randomness policy. Historical canonical `S2_125M` can be reported as context,
but it counts in the primary paired analysis only if its seed/config are fully
documented and a paired S2-minus run exists.

Extension-stage randomness must be seeded independently of upstream stage RNG
consumption. In particular, the S2 bridge stage consumes model/data randomness
before extension while S2-minus does not. Therefore each paired E1 extension run
must explicitly set `training.model_seed` and `training.data_seed` for the
extension stage, and the values must match within each pair:

```text
S2_seed003 extension:       training.model_seed=3, training.data_seed=3
S2_MINUS_seed003 extension: training.model_seed=3, training.data_seed=3
```

Do not rely on implicit defaults or on a global RNG stream advanced by upstream
training. The E1 launcher/registry overrides must record the explicit extension
seeds in the resolved config or run manifest.

### Required Current-Pipeline Re-Evaluation

Before interpreting E1, re-evaluate the canonical comparison checkpoints
through the same current float32 eval pipeline:

- `S1_125M`
- `S2_125M`
- `S3_125M`

Do not compare a fresh `S2_MINUS_125M` result against only a historical ledger
value.

### Seed-Level Decision Rule

Let:

```text
mean_delta = mean(delta_bridge_s for s in paired_seeds)
```

Compute a 95% paired seed-level bootstrap confidence interval by resampling the
seed-paired deltas. Also report raw per-seed deltas and a paired t-style
interval as sensitivity context.

Preregistered categories:

| Category | Rule |
| --- | --- |
| Bridge helps | `mean_delta >= 0.10` and 95% seed-level CI lower bound `> 0` |
| Bridge harmful | `mean_delta <= -0.10` and 95% seed-level CI upper bound `< 0` |
| Bridge negligible | 95% seed-level CI lies fully inside `[-0.10, +0.10]` |
| Inconclusive | anything else |

The `0.10` margin is chosen because it is larger than the observed 125M
continuation-gain scale of about `0.05`, but much smaller than the historical
125M warm-start tax of about `0.644`.

If five paired seeds are inconclusive and the cost ledger remains healthy,
expand to ten paired seeds. If ten paired seeds remain inconclusive, report the
bridge effect as unresolved and do not make it a headline contribution.

### Eval-Batch Bootstrap

Eval-batch bootstrap estimates eval-set sampling variability only. It does not
estimate training-seed noise and cannot settle the bridge question by itself.

For each paired seed, evaluate S2 and S2-minus on the same ordered Books32K eval
batches. For batch `i`:

```text
delta_batch_{s,i} = loss_batch(S2_MINUS_s, i) - loss_batch(S2_s, i)
```

Bootstrap the paired per-batch differences directly. Do not independently
bootstrap S2 and S2-minus losses and difference the intervals.

## E1 Internal-Validity Checklist

Before any E1 compute, generate and archive an internal-validity report. E1 is
valid only if all required checks pass.

### Resolved Config Equality

`S2_125M` extension and `S2_MINUS_125M` extension resolved configs must be
identical after excluding only intentional lineage/output fields:

- stage id
- run id
- experiment name
- output directory
- paper run id
- parent checkpoint or resume checkpoint
- lineage metadata
- free-text notes/tags

The following must match exactly:

- learning-rate schedule
- warmup
- optimizer type and hyperparameters
- inner optimizer config
- batch size
- accumulation steps
- total steps
- sequence length
- dtype
- sharding settings
- model architecture except the parent checkpoint lineage
- eval config

### Dataset Fingerprints

The Books32K training and evaluation fingerprints must match across all
conditions in the E1 comparison. A comparison must fail if any condition was
trained or evaluated on a different fingerprint.

### Optimizer And Restore Policy

- `S2_125M` extension must start with fresh optimizer state rather than
  inheriting bridge optimizer moments.
- `S2_MINUS_125M` must use params-only, shape-aware partial restore from the FA
  seed.
- If either policy cannot be verified from the resolved config and manifests,
  pause before training.

### S2-Minus Existing Config Candidate

The repository already contains:

```text
configs/experiment/125m/pretrained/ext-125m-e2e-32K-from-fa-direct.yaml
```

This is the expected config candidate for `S2_MINUS_125M`. The next registry
change should promote it to a first-class stage only after the internal-validity
report confirms it matches `S2_125M` except for allowed lineage fields.

The required lineage direction is:

```text
S2_125M extension:       training.resume_exp_name=adapt-125m-e2e-8K-from-fa
S2_MINUS_125M extension: training.resume_exp_name=pretrain-125m-fa
```

This single resume-parent difference is the bridge isolation. If S2-minus ever
resumes from the bridge output, it is not S2-minus. If S2 fails to resume from
the bridge output, the comparison no longer isolates the bridge.

## E2a: Paired Retrieval/NIAH Proxy Evaluation

The existing retrieval/NIAH proxy result is preliminary and underpowered. E2a
replaces it with a paired proxy evaluation.

### Confirmatory Test

Primary confirmatory proxy test:

```text
S2_125M vs S3_125M at 32K
```

The primary scale/context is fixed before evaluation:

```text
scale = 125M
context_length = 32768
```

Hypothesis:

```text
accuracy(S2_125M) > accuracy(S3_125M)
```

This is the only uncorrected confirmatory retrieval-proxy test.

Evidence grade:

- A significant proxy result can support a finding or hypothesis of retrieval
  complementarity.
- It cannot become the paper's headline claim unless a validated external
  benchmark in E2b corroborates it.

### Paired Design

Use the same examples across all compared conditions. Every example must have a
stable `example_id`.

For each paired comparison and example:

```text
model_a_correct(example_id)
model_b_correct(example_id)
```

Report:

- Wilson binomial intervals for individual accuracies.
- Paired bootstrap confidence intervals over example IDs for accuracy
  differences.
- Exact McNemar test for paired binary comparisons.

If the proxy produces graded output rather than binary correctness, define the
binarization rule before evaluation.

Committed E2a binarization rule:

```text
correct(example_id) = argmax(candidate_token_logits) == needle_token
```

The E2a proxy is a candidate-set NIAH/RULER-style recall task. Each manifest
example contains one `needle` token, a fixed candidate token set containing the
needle, and a placeholder final query token. The scorer evaluates the restored
checkpoint on the full context plus placeholder, reads the final-position logits
only over the committed candidate set, and marks the example correct iff the
highest-logit candidate is the needle. This is already binary; no post-hoc
graded threshold is allowed. Reports must still store the raw predicted token
and candidate set for audit.

Committed E2a example unit:

```text
num_examples = total examples across the 32K manifest
positions = 0.1, 0.5, 0.9 assigned cyclically across example_id order
```

Thus `n=500` means 500 total paired examples at 32K, not 500 examples per
needle-depth stratum. Stratum-level reads are secondary/exploratory.

Committed 125M checkpoint roster:

```text
S0_125M        = ext-125m-fa-32K
S1_125M        = ext-125m-swa-32K-from-fa
S2_125M        = ext-125m-e2e-32K-from-fa-bridge
S3_125M        = ext-125m-e2e-32K
S2_MINUS_125M = ext-125m-e2e-32K-from-fa-direct-seed001
```

The primary confirmatory comparison uses the canonical `S2_125M` and
`S3_125M` checkpoints above. The `S2_MINUS_125M` checkpoint is included only in
secondary characterization comparisons. Seed001 is fixed before scoring as the
representative no-bridge checkpoint because E1 showed tight loss variance
across all five S2-minus seeds; if future work evaluates all five no-bridge
seeds, that must be labeled an additional exploratory robustness read rather
than changing the E2a confirmatory unit.

### Discordant-Pair Power Rule

McNemar uses only discordant pairs:

```text
b = count(S2 correct, S3 wrong)
c = count(S2 wrong, S3 correct)
effective_n = b + c
```

Start with at least 500 examples. If `effective_n < 50`, expand to a
preregistered 1000-example set. If still underpowered, expand to 2000 examples
if budget/time permits; otherwise report the proxy as underpowered rather than
negative.

### Secondary Tests

Secondary or exploratory proxy comparisons:

- `S2_125M` vs `S2_MINUS_125M`: whether the bridge changes retrieval behavior.
- `S2_MINUS_125M` vs `S1_125M`: whether TTT-E2E conversion alone changes
  retrieval behavior.
- 760M `S2` vs `S3`.
- Context-length or needle-depth strata.

Secondary tests must either be labeled exploratory or use a correction within
the secondary family.

Retrieval results cannot alter the preregistered E1 loss decision. The
`S2_125M` vs `S2_MINUS_125M` retrieval result is characterization only.

Explicit secondary retrieval label: S2_125M vs S2_MINUS_125M.

## Golden-Plan Protection

Adding `S2_MINUS_125M` or internal-validity machinery must not silently change
the canonical stage plans. Before and after registry/orchestrator edits, golden
plan tests must verify command planning for existing canonical stages.

At minimum protect:

- `S1_125M`
- `S2_125M`
- `S3_125M`

The golden tests should ignore only intentionally variable absolute roots, and
should fail on changes to experiment path, stage id, run id, total steps,
runtime mode, dataset roots, checkpoint root, global batch, sequence length, or
stage-specific overrides.

## Cost Discipline

Maintain a revision-v2 cost ledger for all compute runs. Pause if spending
exceeds the expected cost of the active phase by 2x, if any run lacks a
manifest/eval output, or if cumulative spend reaches `$100` before E1 paired
seeds and current-pipeline S1/S2/S3 re-evaluation are complete.
