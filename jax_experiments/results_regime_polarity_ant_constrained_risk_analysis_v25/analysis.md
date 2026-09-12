# Ant constrained termination-risk result

Both variants use a learned discounted termination-risk critic. The absolute arm penalizes candidate risk directly; the relative arm penalizes only risk above the immutable robust action.

| Variant | Seed | Mode wins | Robust / fallback switching | Gain | Termination | Gate |
|---|---:|---:|---:|---:|---:|:---:|
| risk_q_absolute | 85003 | 2/4 | 2439.9 / 2722.1 | +11.6% | 0.0% | FAIL |
| risk_q_absolute | 85021 | 4/4 | 2666.6 / 4612.5 | +73.0% | 33.3% | FAIL |
| risk_q_absolute | 85039 | 1/4 | 2379.1 / 2857.9 | +20.1% | 6.7% | FAIL |
| risk_q_relative | 85003 | 2/4 | 2439.9 / 3265.6 | +33.8% | 0.0% | FAIL |
| risk_q_relative | 85021 | 3/4 | 2666.6 / 4258.7 | +59.7% | 20.0% | FAIL |
| risk_q_relative | 85039 | 3/4 | 2379.1 / 3326.3 | +39.8% | 13.3% | FAIL |

Passing variants: `[]`.

Selected candidate: `None`.

Registered gate: **FAIL**.

Diagnosis: **learned_risk_does_not_stabilize_independent_ant_bank**.

Next step: close independent Ant specialist optimization and train one joint robust-plus-mode conditioned controller with the same relative termination-risk constraint.
