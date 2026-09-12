# Ant joint mode-conditioned risk-controller result

Both budgets use one shared true-mode-conditioned actor, critic, and relative termination-risk critic with an immutable robust fallback. The data-matched arm supplies four times the continuation data so each mode receives the same unique-data scale as a V25 specialist.

| Variant | Seed | Mode wins | Robust / safe joint switching | Gain | Termination | Gate |
|---|---:|---:|---:|---:|---:|:---:|
| joint_equal_budget | 85003 | 1/4 | 2388.5 / 2554.0 | +6.9% | 13.3% | FAIL |
| joint_equal_budget | 85021 | 4/4 | 2707.9 / 3535.9 | +30.6% | 6.7% | FAIL |
| joint_equal_budget | 85039 | 4/4 | 2698.0 / 3261.6 | +20.9% | 0.0% | PASS |
| joint_data_matched | 85003 | 1/4 | 2388.5 / 2718.4 | +13.8% | 13.3% | FAIL |
| joint_data_matched | 85021 | 2/4 | 2707.9 / 3109.4 | +14.8% | 20.0% | FAIL |
| joint_data_matched | 85039 | 3/4 | 2698.0 / 3680.2 | +36.4% | 13.3% | FAIL |

Passing variants: `[]`.

Selected candidate: `None`.

Registered gate: **FAIL**.

Diagnosis: **joint_mode_controller_does_not_stabilize_ant**.

Next step: close Ant controller optimization under actuator polarity and retain it as a negative transfer case; do not train an estimator.
