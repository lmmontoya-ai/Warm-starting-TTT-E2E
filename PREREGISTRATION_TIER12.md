# Tier 1 / Tier 2 / Promoted Experiments Preregistration Addendum

**Preregistered:** 2026-07-14, before Tier-1 implementation produced or exposed
any real-checkpoint results.

**Scope of this addendum:** this revision locks Tier-1 (T1a/T1b/T1c) inputs,
metrics, exclusions, validation gates, and reporting language. Tier-2 and
promoted-experiment training arms remain closed until a later dated addendum
locks their training and decision rules. No Tier-2 or promoted training may be
launched from this document alone.

The repository and `CANONICAL_RESULTS.md` remain the source of truth. If a
checkpoint, dataset, or scalar named here conflicts with the repository, the
run fails closed until the conflict is reconciled in a committed amendment.

## Global Tier-1 rules

1. Tier-1 uses `paper_run_id=revision_v2_t1_eval_v1` and the exact condition
   map in `configs/research/tier1_eval.yaml`.
2. All compared conditions use identical rendered examples and scorer code.
   A scorer change invalidates every condition scored by the earlier version.
3. Full manifests are generated before real checkpoints are loaded. Each
   manifest records source revision, tokenizer revision, selection seed, and a
   SHA-256 content hash.
4. A production result is invalid if its checkpoint lineage, manifest hash,
   tokenizer identity, git state, or expected example count is absent.
5. Tiny synthetic/known-case harness validation is required before any real
   checkpoint is scored. Smoke outputs are never paper evidence.
6. All task attempts and failures are reported. No failed or low-scoring task
   may be silently dropped.
7. Tier-1 is descriptive/exploratory. It does not reopen the confirmatory E2a
   retrieval-complementarity hypothesis, which remains null and closed.
8. The T1c engineering time-box is three working days. If validation has not
   passed by the deadline, T1c is deferred and a dated decision-log entry is
   produced; T1a/T1b and the manuscript continue.

## T1a — short-context downstream sanity battery

### Conditions

- `FA_SEED_125M`: locally pretrained 8K full-attention seed.
- `S2_125M_SEED001`: fresh bridged E1 seed001 final checkpoint.
- `S2_MINUS_125M_SEED001`: paired no-bridge E1 seed001 final checkpoint.
- `S3_125M`: scratch TTT-E2E final checkpoint.
- `S2_BRIDGE_40PCT_125M`: 40% bridge-budget final checkpoint, single seed.

### Tasks and examples

- HellaSwag, ARC-Easy, PIQA, and WinoGrande.
- 500 validation examples per task.
- Selection is deterministic: stable example identifiers are ordered by
  `sha256("tier1-t1a-v1|<task>|<id>")`; the first 500 examples whose complete
  prompt and all choices fit the 1,024-token limit are selected.
- Examples are not truncated. All skipped examples and reasons are recorded.
- Dataset and tokenizer revisions are exactly those in
  `configs/research/tier1_eval.yaml`.

### Scoring and reporting

- Prompts are rendered once into a token manifest and reused byte-for-byte for
  every condition.
- Each candidate is scored independently. Only continuation tokens contribute
  to the score.
- For TTT-E2E conditions, every non-padding causal-LM target in the rendered
  prompt-plus-candidate sequence contributes to the active inner-loop update;
  the continuation-only mask is a separate readout mask. Thus the evaluation
  metric does not alter which tokens drive test-time adaptation. Chunk losses
  are read before their corresponding update, as in the native runtime.
- Primary prediction: maximum mean continuation log-likelihood (`acc_norm`).
- Secondary prediction: maximum summed continuation log-likelihood (`acc`).
- Report every task plus an unweighted macro-average. Report exact sample
  counts and paired 95% bootstrap intervals (10,000 resamples, seed 0) for
  contrasts against the FA seed and S3.
- Reporting rule: describe S2 as higher/lower than a comparator only when the
  paired 95% interval excludes zero; otherwise write "not resolved on this
  battery." No equivalence or non-inferiority claim is licensed.
- The method text must state that TTT-E2E conditions use the active inner-loop
  inference path while the FA seed is a plain Transformer. This asymmetry is
  inherent to the compared systems.

## T1b — fast-weight instrumentation

### Conditions and surface

- Primary pair: `S2_125M_SEED001` and `S2_MINUS_125M_SEED001`.
- Optional descriptive arms may include 5% and 40% bridge checkpoints only if
  both use the identical manifest and instrumentation build. Their absence may
  not block the primary pair.
- Use 32 deterministic Books32K validation sequences at 32,768 tokens. The
  Books3 fingerprint must match across conditions. Sequence identities are
  fixed in a pre-run manifest.

### Locked diagnostics

For every example, 1,024-token chunk, and prime-MLP suffix layer, record:

- pre-update fast-weight L2 norm;
- raw inner-gradient L2 norm;
- applied optimizer-update L2 norm;
- relative update norm `||delta|| / max(||weight||, 1e-12)`;
- gradient-clip threshold crossing;
- finite/non-finite status.

If intermediate extension checkpoints are available, run the same probe at
every declared checkpoint step and label the result a training trajectory. If
only the final exported checkpoint is available, report final-checkpoint
sequence-processing diagnostics and explicitly state that a training-time
trajectory could not be reconstructed. Missing intermediate checkpoints must
not be imputed.

### Reporting

- Primary figure: mean per-chunk applied-update norm with a 95% bootstrap band
  over the 32 paired sequences, S2 versus S2-minus.
- Secondary tables: per-layer mean raw-gradient, update, and relative-update
  norms; clip-crossing rate; non-finite count.
- All interpretation is descriptive. The phrase "activates the fast-weight
  pathway" is allowed only if the preregistered applied-update statistic is
  consistently nonzero and materially separated; otherwise report the
  observed similarity/difference without mechanistic causal language.
- No diagnostic NLL is a manuscript LM-loss result. Any incidental diagnostic
  loss remains in raw artifacts and may not be mixed with the canonical
  64-batch float32 loss table.

## T1c — external RULER-v1 subset

### Harness and tasks

- Source data protocol: NVIDIA/RULER generator pinned at commit
  `38da79d79519ef87aa46ae804f838e1eab7f86d7`. The NeMo-Skills integration is
  pinned at `f4a3fd8e524acd9abd1fea4387e8f179f6d51cf3`; the published
  `rulerv1-ns` pointer is pinned at
  `e8bbff677ca2c239640dc90f93310dcf32408c93`. All three revisions are
  recorded in `configs/research/tier1_eval.yaml`.
- Base-model prompt template; the canonical Llama-3 tokenizer is taken from
  the public `NousResearch/Meta-Llama-3-8B` mirror at pinned commit
  `315b20096dc791d381d514deb5f8bd9c8d6d3061`. The four tokenizer/config file
  SHA-256 fingerprints are locked in `configs/research/tier1_eval.yaml` and are
  shared with T1a. This avoids dependence on gated model-weight access.
- RULER's essay/word-list auxiliary inputs and NLTK sentence-tokenizer data
  are also SHA-256 pinned in the Tier-1 config. A live upstream download that
  no longer reproduces those hashes is a harness-gate failure, not an implicit
  dataset revision.
- Official generator task identifiers: `niah_multikey_1`, `vt`, and `cwe`.
- 200 examples per task at 32,768 tokens, using official task construction,
  answer prefix, task-specific generation limit, and official normalization.
- Every unpadded official prompt must occupy at least 95% of the 32,768-token
  surface. Falling below this floor is a harness failure, not a short example
  that may be silently promoted to the 32K table.
- Because the native TTT runtime consumes complete 1,024-token chunks, the
  unmodified official prompt token sequence is left-padded with the tokenizer
  EOS/pad id to exactly 32,768 prompt tokens. Padding targets have zero loss
  weight; prompt content is never truncated, and the padding count and original
  prompt length are retained per example. The generated answer is not counted
  as another adaptation chunk.
- Greedy decoding; no sampling. Fast weights adapt while consuming the prompt
  in 1,024-token chunks and are frozen during the short answer decode.

### Conditions

- 125M: `S2_125M_SEED001`, `S2_MINUS_125M_SEED001`, `S3_125M`.
- 760M exploratory: `S2_760M`, `S3_760M`, only after the identical 125M
  harness path and a 760M smoke both pass.

### Validation and reporting

- Before real scoring: verify the source commit, tokenizer hash, deterministic
  generation, prompt token length, known-case metric fixture, and a tiny
  synthetic model smoke. Every check must execute; skipped means FAIL.
- Report per-task exact-match/official accuracy, the unweighted macro-average,
  raw counts, and paired disagreements. Use paired bootstrap intervals for
  descriptive differences and exact two-sided McNemar p-values as secondary
  diagnostics only.
- Primary readout: whether RULER agrees or disagrees descriptively with the E2a
  finding that warm-starting has no demonstrated retrieval complementarity.
  T1c cannot overturn or replace E2a's confirmatory decision.
- 760M remains inherited quality evidence and must be labeled exploratory.

## Production stop conditions

A Tier-1 production run stops before scoring if any of these occur:

- preregistration or config hash mismatch;
- dirty or unidentified git state;
- missing checkpoint, lineage manifest, or checkpoint payload;
- missing/mismatched dataset fingerprint;
- unpinned dataset/tokenizer/harness revision;
- known-case smoke failure;
- condition-specific scorer/config divergence;
- non-finite model output;
- RULER time-box expiration.

After a successful session, raw results and reports are mirrored to the public
result repository, the claims audit and session cost ledger are updated, and
the compute box is terminated the same day.
