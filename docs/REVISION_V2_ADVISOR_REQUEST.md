# Revision V2 Advisor Request

Status: send-ready draft, 2026-06-12.

Purpose: Phase 5 requires the venue/status dependency to be in motion before
the revision-v2 in-house experimental program is treated as closed. This note
is the exact request to send to Vijay/advisor.

## Message

Subject: Warm-start TTT-E2E revision-v2 results and venue decision

Hi Professor Madisetti,

The revision-v2 in-house experimental program is now complete. The core result
is strong: the preregistered no-bridge control validates the bridge contribution
directly. Across five paired seeds on the 125M Books32K 64-batch surface, the
mean bridge effect is:

```text
loss(S2-minus) - loss(S2) = 2.0767
95% paired seed-bootstrap CI = [2.0500, 2.0970]
```

The powered retrieval proxy did not support the earlier complementarity
hypothesis, so the revised paper should stay with the cleaner characterization
framing: warm-starting is a reuse/lower-barrier path into long-context TTT-E2E,
not an absolute quality win over scratch. Scratch remains better on LM loss.

E3 is also complete. The bridge-budget frontier shows:

- 5% bridge budget: loss 4.5396, recovering 70.20% of the E1 bridge effect
  (narrowly meeting the preregistered threshold-like label).
- 20% bridge budget: loss 3.6460.
- 40% bridge budget: loss 3.4462.
- S2-minus +1440 continuation improves to 5.4874, so extra direct-extension
  training helps, but it remains far behind the bridged paths.

All artifacts are documented in the repo:

- `CANONICAL_RESULTS.md`
- `docs/CLAIMS_AUDIT.md`
- `docs/REVISION_V2_E3_FRONTIER_ANALYSIS.md`
- `docs/REVISION_V2_EXPERIMENTAL_CLOSURE.md`

Could you confirm the next submission target and status constraints?

1. Should we target IEEE OJ-CS, IEEE Access, or TMLR for the revised full
   characterization paper?
2. Are both prior submissions formally closed from your side, so there is no
   dual-submission issue before the next submission?
3. Do you know of any updated publication status for the TTT-E2E paper or
   related unpublished work we should account for before finalizing related
   work?

My recommendation is to write the full revised manuscript around the
characterization spine, with E1 as the headline experimental contribution and
E3 as the practical budget guidance. E5/external-model generality should stay
bonus-tier and should not delay resubmission.

Best,
Luis

## Send Checklist

- Send to Vijay/advisor before manuscript drafting is declared complete.
- Record the sent date and any reply in the manuscript planning notes.
- If the answer selects a venue, update the writing checklist and cover-letter
  outline accordingly.
