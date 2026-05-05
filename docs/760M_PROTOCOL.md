# 760M Revised Protocol

The 760M experiments use the same scientific ladder as the 125M runs, but the
executed protocol differs from the original faithful plan because the faithful
batch sizes exceeded available hardware memory.

## Original Faithful Plan

- Bridge: sequence length 8,192, global batch size 64, 2,900 steps.
- 32K extension: sequence length 32,768, global batch size 32, 725 steps.

## Revised Token-Equivalent Protocol

- Bridge: global batch size 8, 23,200 steps, preserving 1.52B tokens.
- 32K extension: global batch size 8, 2,900 steps, preserving 760M tokens.

The revision preserves observed token budgets while changing the number of
optimizer steps. The downstream 32K comparison remains matched between S2 and
S3.

## Reporting Caveat

The 760M warm-start path used mixed hardware surfaces, so the paper reports
token budgets for warm-start cost and uses measured scratch extension GPU-hours
as a reference point. The 760M S0/S1 controls and continuation analyses were not
completed in the paper release.
