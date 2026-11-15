# V28 actuator-polarity action-compensation audit

This is a no-training mechanism audit over the five frozen V21 HalfCheetah policy seeds and new calibration/holdout streams.

| Arm | Switching return | Stationary return |
|---|---:|---:|
| Robust SAC | 1424.8 | 1442.6 |
| Reference, no compensation | 513.3 | 777.4 |
| Reference + true-mode compensation | 4291.8 | 4283.8 |
| Reference + causal v5 compensation | 4076.2 | 4266.1 |
| V21 bank + causal v5 | 3246.0 | 3478.0 |
| SAC5 + causal v5 | 2301.4 | 2519.5 |

## Paired switching comparisons

| Comparison | Difference | 95% CI | Seeds | Events | Pass |
|---|---:|---:|---:|---:|---:|
| Causal compensation - no compensation | +3563.0 | [+2961.1, +4164.9] | 5/5 | 15/15 | PASS |
| Causal compensation - robust SAC | +2651.4 | [+2201.2, +3101.6] | 5/5 | 15/15 | PASS |
| Causal compensation - V21 bank | +830.2 | [+669.2, +991.3] | 5/5 | 15/15 | PASS |
| V21 bank - causal compensation | -830.2 | [-991.3, -669.2] | 0/5 | 0/15 | FAIL |
| Causal compensation - SAC5 | +1774.8 | [+926.8, +2622.7] | 5/5 | 15/15 | PASS |

## Mechanism checks

- End-to-end oracle equivalence: PASS; maximum paired return error `0`.
- Oracle-headroom recovery: `5/5` seeds pass the frozen 70% threshold.
- Causal mode accuracy: `0.9884`; median switch detection delay: `3.0` steps.
- Mean executed-signal error from causal mode mistakes: `0.010726`.

## Decision

Causal compensation claim: **PASS**.
Bank comparison: **single_policy_compensation_is_superior**.
Diagnosis: **causal_action_compensation_supported**.

HalfCheetah has no health termination in this implementation. These results measure return and switching control loss, not safety. The sign-transform result is specific to actuator polarity and is not claimed for bus uncertainty or non-invertible gain loss.

## Accounting

No new policy or estimator training was performed. V21 BAPR reuses five policies trained with 16.8M total interactions; SAC5 reuses five policies trained with 28.0M. Policy count, interactions, and estimator cost are separate quantities.
