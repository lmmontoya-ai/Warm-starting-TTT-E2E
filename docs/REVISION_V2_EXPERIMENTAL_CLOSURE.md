# Revision V2 Experimental Closure Note

Date: 2026-06-12.

The in-house revision-v2 experimental program is closed. A0, E1, E2a, E3, and
the S2-minus continuation have all been run, evaluated on the current
64-batch/float32 checkpoint-restore surface, documented in the canonical ledger,
and tied to the claims audit.

## Closed Experimental Gates

| Gate | Status | Result |
| --- | --- | --- |
| A0 canonical ledger | complete | Repo ledger is source of truth; stale surfaces identified. |
| E1 bridge isolation | complete | Bridge helps: mean `S2_MINUS - S2 = 2.0767`, 95% paired seed-bootstrap CI `[2.0500, 2.0970]`. |
| Current-pipeline baseline reconciliation | complete | 64-batch losses: S0 `6.5879`, S1 `6.5423`, S2 `3.9169`, S3 `3.2722`. |
| E2a retrieval proxy | complete | No warm-start retrieval complementarity; primary 125M S2 vs S3 is `0.060` vs `0.084`, not significant. |
| E3 bridge-budget frontier | complete | 5% bridge narrowly recovers `70.20%` of the E1 bridge effect; 20% and 40% continue improving. |
| S2-minus continuation | complete | +1440 steps improves loss by `0.5102` to `5.4874`, weakening a hard plateau claim but leaving a large bridge advantage. |

## Verification

- Final focused E3/runtime-profile tests: `4 passed`.
- Final full local test suite: `86 passed`.
- No Vast instances remained after teardown (`vastai show instances --raw`
  returned `[]`).
- Final E3/continuation cost ledger:
  `docs/REVISION_V2_COST_LEDGER.md`.

## Current Paper Spine

The revised paper should now be written as a rigorous characterization study:

1. Warm-starting is a reuse/entry-path method for adapting an existing FA
   checkpoint into long-context TTT-E2E.
2. The bridge stage is isolated and necessary under the matched standard
   extension budget.
3. Scratch remains better on LM loss; the paper reports the quality tax instead
   of hiding it.
4. Powered retrieval-proxy evaluation does not support a complementarity
   headline.
5. Bridge budget is a practical cost-quality knob: a small bridge recovers most
   of the matched-budget bridge effect, while larger bridges continue improving
   final LM loss.

## Remaining Non-Compute Dependencies

These are manuscript/process items, not blockers for additional in-house
experiments:

- Advisor/venue request: send-ready draft in
  `docs/REVISION_V2_ADVISOR_REQUEST.md`. It asks Vijay/advisor to confirm the
  target venue path, prior-submission closure, and any TTT-E2E publication
  status updates.
- Related-work currency scan: initial scan completed in
  `docs/REVISION_V2_RELATED_WORK_SCAN.md`. Re-run immediately before final
  submission because TTT/In-Place TTT/LaCT work is moving quickly.

## E5 Status

External-model generality (E5) is bonus-tier only. It should not delay the
revised manuscript. If opened, it must be time-boxed to 3-4 engineering days and
must follow the profile -> import -> audit -> initial-loss probe sequence before
any training spend.
