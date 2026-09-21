# V29 Ant oracle action-compensation audit

This no-training development audit reuses the three frozen V22 Ant policy seeds and new calibration/holdout streams.

| Arm | Switching return | Stationary return | Switch term. | Static term. |
|---|---:|---:|---:|---:|
| Robust SAC | 2588.6 | 2745.3 | 11.1% | 4.4% |
| Reference, no compensation | -266.3 | 24.7 | 97.8% | 59.4% |
| Reference + true-mode compensation | 4510.7 | 4698.5 | 4.4% | 6.7% |
| V22 dynamic specialist oracle | 4464.4 | 4522.0 | 28.9% | 12.2% |

| Seed | Ref. mode | Robust switch | Oracle switch | Gain | Oracle term. | Gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 85003 | 3 | 2422.3 | 3774.5 | +55.8% | 0.0% | FAIL |
| 85021 | 3 | 2613.7 | 5312.6 | +103.3% | 13.3% | FAIL |
| 85039 | 2 | 2730.0 | 4445.0 | +62.8% | 0.0% | PASS |

## Paired switching comparisons

| Comparison | Difference | 95% CI | Seeds | Events |
|---|---:|---:|---:|---:|
| Oracle compensation - robust SAC | +1922.1 | [+191.1, +3653.0] | 3/3 | 9/9 |
| Oracle compensation - no compensation | +4777.0 | [+2640.9, +6913.0] | 3/3 | 9/9 |
| Oracle compensation - dynamic bank | +46.3 | [-891.8, +984.3] | 1/3 | 5/9 |

## Decision

Oracle compensation gate: **FAIL**.
Diagnosis: **ant_reference_policy_is_not_survival_safe**.
Maximum native-equivalence return error: `0`.
Next step: stop estimator work and retain Ant as a safety counterexample.

This is an Ant development upper-bound audit, not an independent cross-environment confirmation. It uses true mode and performs no estimator or policy training.
