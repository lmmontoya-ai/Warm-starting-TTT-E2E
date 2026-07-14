# E2a Execution Plan — Retrieval Proxy at n≥500, Preregistered Read, Advisor Memo

**Goal.** Replace the n=16 retrieval anecdote with a powered, paired, preregistered proxy evaluation; run all conditions in one GPU session; interpret strictly through the committed hierarchy; deliver one advisor memo covering E1-final + E2a together.

**Posture.** The Mac is the control plane (harness build, tests, stats, memo). The GPU box exists only for inference and is provisioned once, after the harness is fully tested locally. No training anywhere in this plan.

**Non-negotiables (inherited, restated):**
- The preregistered hierarchy is fixed: **primary confirmatory = S2_125M vs S3_125M at 32K, uncorrected.** Everything else is secondary/exploratory.
- Same example set, stable `example_id`s, across all conditions within a scale.
- Wilson intervals for accuracies; exact McNemar for paired comparisons; discordant-pair count reported prominently.
- Auto-expansion ladder: if discordant pairs `b + c < 50` at n=500 → expand to n=1000; if still underpowered → n=2000 if budget/time permits; else report **underpowered, not negative**.
- If probe output is graded, the binarization rule is committed in `PREREGISTRATION_REVISION_V2.md` **before** any condition is scored.
- Retrieval results cannot retroactively alter the E1 loss decision (already final: helps, 2.0767 [2.0500, 2.0970] on the 64-batch surface).
- Proxy-only evidence supports a complementarity **finding/hypothesis**; it cannot be the paper's headline claim without E2b corroboration.
- Every reported number comes from checkpoint-restored models through the current pipeline; results live under a new `paper_run_id` (`revision_v2_e2a_proxy_v1`).

---

## Phase 1 — Build the E2a harness

**Objective:** extend the existing NIAH/RULER-style proxy machinery (around `scripts/24_eval_ruler.py` / `ttt/research/ruler_runner.py`) into a paired, powered, statistics-complete evaluator.

**Implementation tasks:**
1. **Example generation at scale.** `--num-examples N` (default 500), deterministic from a fixed committed seed; each example carries a stable `example_id`, needle position/depth metadata, and context-length tag. The *same* generated set is reused across every condition at a given scale — generate once, serialize to a manifest, evaluate many.
2. **Paired evaluation output.** Per condition: per-example binary outcome keyed by `example_id` (plus raw model output for audit). Per comparison: paired table joining on `example_id`, refusing to compare runs whose example manifests differ (hash the manifest; assert equality).
3. **Statistics wiring.** Use the already-tested `ttt/research/preregistration.py` functions — Wilson interval, exact McNemar, paired counts — do not reimplement. Add a thin aggregator that emits, per comparison: accuracies + Wilson CIs, b/c discordant counts, McNemar p, and the expansion-ladder verdict (`adequate` / `expand_1000` / `expand_2000` / `underpowered`).
4. **Binarization rule.** Inspect probe output format. If graded, write the exact rule (e.g., exact-match after normalization) into `PREREGISTRATION_REVISION_V2.md` **now**, commit, and have the scorer cite it. If already binary, record that fact in the prereg for completeness.
5. **Hierarchy enforcement in output.** The summary report template hard-labels each comparison: `PRIMARY (uncorrected)` for S2_125M vs S3_125M; `SECONDARY` for S2 vs S2-minus, S2-minus vs S1, 760M S2 vs S3, and any strata. Secondary results carry Holm-corrected and uncorrected p side by side, labeled exploratory.
6. **Condition roster.** 125M: S0, S1, S2, S3, S2_MINUS (one representative seed checkpoint — preregister *which* seed, e.g. seed001, before running; or evaluate all five S2/S2-minus seeds and preregister that the seed-mean is the comparison unit. Pick one, commit it, write it down). 760M: S2, S3.

**DoD:**
- Harness runs end-to-end on a mock/tiny model locally (CPU ok) producing schema-valid outputs.
- Example manifest is deterministic: two generations with the same seed are byte-identical.
- Prereg file updated (binarization rule + S2/S2-minus seed-unit decision) and committed **before** any real checkpoint is scored.

**Tests (add `tests/test_e2a_harness.py`):**
- Determinism: same seed → identical example manifest hash.
- Pairing: evaluator refuses two conditions with mismatched manifest hashes.
- Stats golden values: Wilson and McNemar against known fixtures (reuse/extend existing prereg tests).
- Expansion ladder: synthetic results with b+c = 49 → `expand_1000`; b+c = 50 → `adequate`.
- Hierarchy labels: primary/secondary labeling is emitted correctly and S2_125M-vs-S3_125M is the only `PRIMARY`.
- Schema: per-example outputs contain example_id, outcome, checkpoint id, context length, seed.
- `uv run pytest -q` fully green, including golden-plan and ledger tests.

---

## Phase 2 — Single GPU session

**Objective:** one box, one session, every condition at both scales.

**Implementation sequence:**
1. Use the same Prime Intellect 2×H100.
2. Check or Restore checkpoints from HF: 125M S0/S1/S2/S3 finals, the chosen E1 seed checkpoints (S2_seedXXX, S2_MINUS_seedXXX per the committed roster), 760M S2/S3.
3. Verify fingerprints/restore manifests for every checkpoint (reuse preflight machinery; all checks must execute).
4. Run smoke: n=8 examples on one 125M condition; confirm output schema and a sane accuracy before committing to full runs.
5. Run full n=500 across all 125M conditions, then 760M S2/S3, all against the same per-scale example manifests.
6. Compute the aggregator report on-box; check the expansion ladder. If `expand_1000` triggers for the **primary** comparison, run the expansion in the same session (the +500 examples come from extending the same seeded generator — preregistered as the committed ladder, not a new decision).
7. Sync all outputs + manifests + report to HF under `revision_v2_e2a_proxy_v1`; verify upload; **terminate the box.**

**DoD:**
- Every rostered condition has per-example outputs against the committed manifest.
- Expansion ladder resolved for the primary comparison (adequate, expanded, or documented underpowered).
- All artifacts on HF; box terminated; session cost logged in the cost ledger.

**Verification:**
- Spot-check 10 raw model outputs per scale by eye for scoring sanity (needle actually present/absent as scored).
- Aggregator refuses any cross-manifest comparison (test fires if misconfigured).
- Cost ledger entry: expected ~$25–50 all-in (inference + possible expansion); flag if exceeded.

---

## Phase 3 — Preregistered read

**Objective:** interpret strictly through the committed hierarchy; no improvisation.

**Procedure:**
1. Read the **primary** result first and alone: S2_125M vs S3_125M, exact McNemar, uncorrected. Outcomes:
   - **S2 > S3, significant, adequate power** → complementarity *finding* (proxy-grade). Paper may report it as evidence; headline status remains gated on E2b.
   - **No significant difference, adequate power** → no complementarity claim; paper is the characterization study with E1 as the isolated-mechanism headline. This is a fine outcome.
   - **S3 > S2, significant** → report it straight; it sharpens the trade-off framing (scratch wins quality *and* retrieval; warm-start wins reuse).
   - **Underpowered after ladder** → report as underpowered; no directional claim either way.
2. Then read secondaries with Holm correction within the family, labeled exploratory: does the bridge change retrieval (S2 vs S2-minus)? does conversion alone change it (S2-minus vs S1)? does the 760M direction replicate 125M?
3. Write the one-paragraph interpretation per the labels above — the language constraints were committed in advance; fill in numbers, don't compose framing.

**DoD:** an `e2a_analysis.md` in `reports/revision_v2/` containing: per-condition accuracies + Wilson CIs, paired tables, discordant counts, primary verdict with preregistered label, corrected secondary results, and the explicit sentence on what the paper may and may not claim from proxy-grade evidence.

---

## Phase 4 — Combined decision memo to advisor

**Objective:** one page; discharge the advisor checkpoint for both gates before any further spend (E3) or writing.

**Contents (in order):**
1. **E1 final:** preregistered statistic, 8-batch gate decision (helps), 64-batch manuscript-grade confirmation (2.0767 [2.0500, 2.0970]), mechanism note (S2-minus trains but plateaus ~6.0; gap uniform across all positions), replication note (fresh S2 ≈ historical S2 within ~0.007 across hardware/topology), A0 escalation closed (8-batch drift = sampling variance; 64-batch values reproduce historical ledger).
2. **E2a result:** primary verdict verbatim from the preregistered category; secondary highlights; power status.
3. **Manuscript implication:** which thesis variant the data now supports (bridge-validated reuse framing; complementarity included/excluded/suggestive per Phase 3), what the abstract will and won't claim.
4. **Asks:** (a) sign-off on the thesis variant; (b) authorize E3 (bridge helps → unlocked; 5% run first, curve capture per plan, ~$55–70); (c) confirm whether E2b (one external benchmark, time-boxed) is wanted given the E2a outcome — it matters most if complementarity is suggestive-but-proxy-only; (d) confirm both prior submissions are formally closed before the next venue submission.
5. **Budget line:** total spent to date vs $1k, projected remaining for E3 (+E2b if authorized).

**DoD:** memo sent; advisor responses recorded as decisions in the repo (`docs/` decision log); E3 spend does not start before the authorization in (b).

---

## Sequence summary
1. Build harness + tests locally; commit prereg additions (binarization, seed-unit) **first**.
2. Full local test suite green.
3. Provision → restore → fingerprints → smoke → n=500 all conditions → ladder if triggered → sync → terminate.
4. Preregistered read; `e2a_analysis.md`.
5. Combined E1+E2a memo to Vijay; await E3/E2b authorization.

**Failure handling:** any fingerprint mismatch, manifest-hash mismatch, or skipped check stops the run (same semantics as E1 preflight). A harness bug discovered mid-session → fix locally, re-test, re-provision; do not hot-patch the scorer on the box between conditions, since all conditions must be scored by identical code.