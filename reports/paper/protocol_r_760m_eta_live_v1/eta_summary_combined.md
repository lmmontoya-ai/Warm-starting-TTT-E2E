`760M` revised-protocol ETA summary on `8x H200` from real `20`-step benchmarks.

- `S0_760M`: `3833.66s` training ETA, `8.52` GPU-hours, steady-state `1.31s/step`
- `S1_760M`: `9952.17s` training ETA, `22.12` GPU-hours, steady-state `3.39s/step`
- `S2_ADAPT_760M`: `18191.22s` training ETA, `40.42` GPU-hours, steady-state `0.78s/step`
- `S2_760M`: `9049.77s` training ETA, `20.11` GPU-hours, steady-state `3.06s/step`
- `S3_760M`: `9182.90s` training ETA, `20.41` GPU-hours, steady-state `3.11s/step`

Totals, training only:
- full author-seeded `760M` ladder: `50209.71s` (`13.95h`) on `8x H200`
- total estimated compute: `111.58` GPU-hours

These estimates are training-only and do not include parity eval, HF export, retries, or instance bootstrap/sync overhead.
