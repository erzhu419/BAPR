# V20 full-state specialist policy-stability result

This fresh-seed screen changes only policy selection and actor update frequency after a full-controller warm start.

| Variant | Seed | Holdout mode wins | Robust / safe switching | Safe gain | Event wins | Cell gate |
|---|---:|---:|---:|---:|---:|:---:|
| full_state_final | 83003 | 4/4 | 1778.1 / 3862.3 | +117.2% | 3/3 | PASS |
| full_state_final | 83021 | 4/4 | 1445.2 / 2834.0 | +96.1% | 3/3 | PASS |
| full_state_final | 83039 | 4/4 | 1309.1 / 2850.6 | +117.8% | 3/3 | PASS |
| full_state_best | 83003 | 0/4 | 1778.1 / 1778.1 | +0.0% | 0/3 | FAIL |
| full_state_best | 83021 | 2/4 | 1445.2 / 2438.6 | +68.7% | 3/3 | FAIL |
| full_state_best | 83039 | 3/4 | 1309.1 / 2749.9 | +110.1% | 3/3 | PASS |
| period2_best | 83003 | 2/4 | 1778.1 / 1852.1 | +4.2% | 3/3 | FAIL |
| period2_best | 83021 | 1/4 | 1445.2 / 1727.2 | +19.5% | 3/3 | FAIL |
| period2_best | 83039 | 3/4 | 1309.1 / 2460.9 | +88.0% | 3/3 | PASS |

| Candidate | Mean delta vs control | Seed wins | Event wins | Min stationary retention | Worst gain candidate / control | Decision |
|---|---:|---:|---:|---:|---:|:---:|
| full_state_best | -860.1 | 0/3 | 0/9 | 33.9% | +0.0% / +96.1% | FAIL |
| period2_best | -1168.9 | 0/3 | 0/9 | 29.2% | +4.2% / +96.1% | FAIL |

Passing candidates: `[]`.

Selected candidate: `None`.

Diagnosis: **policy_selection_and_period_do_not_resolve_bank_variance**.

Next step: close validation checkpoint selection and actor-period thinning; do not alter the frozen posterior or v18/v19 holdouts.
