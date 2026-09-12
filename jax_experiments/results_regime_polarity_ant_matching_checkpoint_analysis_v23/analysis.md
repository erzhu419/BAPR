# Ant matching-mode checkpoint result

Modes 0/1 are newly selected on matching-mode validation. Modes 2/3 reuse the frozen V22 final policies.

| Variant | Seed | Mode wins | Robust / safe switching | Gain | Termination | Gate |
|---|---:|---:|---:|---:|---:|:---:|
| matching_best | 85003 | 0/4 | 2455.0 / 2793.9 | +13.8% | 6.7% | FAIL |
| matching_best | 85021 | 3/4 | 2739.7 / 4350.5 | +58.8% | 53.3% | FAIL |
| matching_best | 85039 | 3/4 | 2680.5 / 3782.0 | +41.1% | 6.7% | FAIL |
| period2_matching_best | 85003 | 1/4 | 2455.0 / 3793.0 | +54.5% | 0.0% | FAIL |
| period2_matching_best | 85021 | 3/4 | 2739.7 / 4516.5 | +64.9% | 46.7% | FAIL |
| period2_matching_best | 85039 | 3/4 | 2680.5 / 3459.5 | +29.1% | 13.3% | FAIL |

Selected candidate: `None`.

Registered gate: **FAIL**.

Next step: close checkpoint selection and actor-period tuning; train specialists on switch-state rollouts with explicit termination risk while keeping the robust actor as fallback.
