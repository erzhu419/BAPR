# V32 prospective power confirmation

The V31 five-seed result was used only to choose n=10. This table contains ten new policy seeds and new event streams; V31 outcomes are not pooled.

| Arm | Switching return | Stationary return |
|---|---:|---:|
| Robust SAC, 5.6M | 1650.8 | 1775.0 |
| Robust SAC, 8.4M | 2287.1 | 2382.6 |
| Canonical mode 0, no compensation | 340.3 | 560.3 |
| Canonical + true-mode compensation | 3422.0 | 3432.5 |
| Canonical + frozen v5 compensation | 3297.1 | 3403.5 |
| Recurrent ESCP, 8.4M | 2046.0 | 2117.0 |
| RE-SAC b0, 8.4M | 2360.2 | 2435.9 |

## Paired switching comparisons

| Comparison | Difference | 95% CI | Seeds | Events | Pass |
|---|---:|---:|---:|---:|:---:|
| Causal - no compensation | +2956.8 | [+2500.7, +3412.8] | 10/10 | 30/30 | yes |
| Causal - robust SAC 5.6M | +1646.2 | [+1174.9, +2117.6] | 10/10 | 30/30 | yes |
| Causal - robust SAC 8.4M | +1009.9 | [+391.7, +1628.2] | 7/10 | 23/30 | no |
| Causal - ESCP 8.4M | +1251.1 | [+553.5, +1948.7] | 9/10 | 25/30 | yes |
| Causal - RE-SAC 8.4M | +936.9 | [+337.1, +1536.7] | 9/10 | 27/30 | yes |

## Decision

Prospective power confirmation: **False**.
Diagnosis: **canonical_controller_lacks_equal_budget_oracle_headroom**.
Oracle headroom: 7/10 seeds; causal recovery: 10/10; stationary retention: 10/10.
Frozen-v5 mode accuracy: 0.9855; median switch delay: 3.0 steps.

The decision requires positive paired means with two-sided 95% CI lower bounds above zero against equal-budget SAC, ESCP, and RE-SAC, plus at least 8/10 seed wins and 24/30 event wins.

The claim remains limited to HalfCheetah actuator polarity. The compensation path and all comparators receive 8.4M interactions per policy seed; the frozen v5 estimator is not retrained.
