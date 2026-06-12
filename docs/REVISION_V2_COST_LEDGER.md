# Revision V2 Cost Ledger

Status: current as of 2026-06-12.

This ledger records the cost evidence needed for the revision-v2 close-out. It
is not a manuscript cost table; manuscript cost claims should use the canonical
training run inventories in `CANONICAL_RESULTS.md`, not the cloud debugging
spend below.

## Final E3 / Continuation Session

| Item | Value |
| --- | ---: |
| Provider | Vast.ai |
| Instance | 8x H200 |
| Instance id | `40609982` |
| Hourly rate observed by Vast | `$28.96929824561404/hr` |
| Duration observed by Vast before teardown | `28948.292427778244` seconds (`8.0412` hours) |
| Approximate session cost | `$232.93` |
| Observed training GPU-hours in run manifests | `25.394241899416234` |
| Instance status after close-out | destroyed; `vastai show instances --raw` returned `[]` |

The final E3 session was above the optimistic `$80-95` estimate in
`goals/goal_part_3.md`, mostly because the validated 8x H200 instance was kept
alive through setup, eval, artifact upload, HF verification, and local artifact
copy. It remained under the `$250` hard ceiling for this final experimental
program.

## Budget Interpretation

- For manuscript economics, do not count cloud debugging/provisioning cost as
  model-training cost. Use canonical run inventories and the quality/cost caveats
  in `CANONICAL_RESULTS.md`.
- For project management, the important close-out fact is that the final
  experimental box was terminated and no Vast instances remain running.
- Earlier revision-v2 sessions are documented in their respective reports and
  memos where available, but this final ledger only proves the Phase 5 ceiling
  for the E3/continuation close-out session.
