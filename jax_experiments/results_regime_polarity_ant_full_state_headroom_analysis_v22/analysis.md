# Ant frozen-recipe policy-bank headroom result

This development screen transfers the frozen V21 robust/full-state specialist recipe to Ant. It does not train or tune a router.

| Seed | Stationary mode wins | Robust / safe-oracle switching | Gain | Event wins | Gate |
|---:|---:|---:|---:|---:|:---:|
| 85003 | 1/4 | 2451.6 / 3048.4 | +24.3% | 3/3 | FAIL |
| 85021 | 1/4 | 2673.5 / 4127.8 | +54.4% | 3/3 | FAIL |
| 85039 | 2/4 | 2748.1 / 3501.5 | +27.4% | 3/3 | FAIL |

Mean robust / safe oracle: 2624.4 / 3559.2.

Registered gate: **FAIL**.

Next step: retain Ant as a negative transfer result and do not train an estimator on this policy bank.

Aggregation amendment: adapted the protocol-specific `validate_audit(seed)` API; audit artifacts and gates are unchanged.
