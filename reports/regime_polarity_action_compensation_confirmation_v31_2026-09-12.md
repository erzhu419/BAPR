# V31 fresh-policy canonical action-compensation confirmation

Mode 0, five policy seeds, training budgets, event streams, and decision gates were frozen before training.

| Arm | Switching return | Stationary return |
|---|---:|---:|
| Robust SAC, 5.6M | 2178.4 | 2353.9 |
| Robust SAC, 8.4M | 2594.3 | 2738.2 |
| Canonical mode 0, no compensation | 325.4 | 623.7 |
| Canonical + true-mode compensation | 3642.5 | 3631.5 |
| Canonical + frozen v5 compensation | 3400.6 | 3508.5 |
| Recurrent ESCP, 8.4M | 1901.8 | 1963.1 |
| RE-SAC b0, 8.4M | 1916.1 | 1957.9 |

## Paired switching comparisons

| Comparison | Difference | 95% CI | Seeds | Events | Pass |
|---|---:|---:|---:|---:|:---:|
| Causal - no compensation | +3075.2 | [+2327.7, +3822.7] | 5/5 | 15/15 | yes |
| Causal - robust SAC 5.6M | +1222.2 | [+666.5, +1777.8] | 5/5 | 15/15 | yes |
| Causal - robust SAC 8.4M | +806.3 | [-30.2, +1642.8] | 4/5 | 12/15 | no |
| Causal - ESCP 8.4M | +1498.8 | [+611.8, +2385.7] | 5/5 | 15/15 | yes |
| Causal - RE-SAC 8.4M | +1484.5 | [-23.8, +2992.8] | 4/5 | 12/15 | no |

## Decision

Fresh-policy confirmation: **False**.
Diagnosis: **canonical_compensation_fails_comparators:causal_vs_equal_budget_sac,causal_vs_equal_budget_resac**.
Oracle headroom: 4/5 seeds; causal recovery: 5/5; stationary retention: 4/5.
Frozen-v5 mode accuracy: 0.9826; median switch delay: 3.0 steps.

The claim remains limited to HalfCheetah actuator-polarity, whose sign transform is exactly invertible. HalfCheetah has no health termination here, so this does not establish safety.

## Accounting

The compensation path uses one 5.6M robust source plus 2.8M fixed-mode fine-tuning per seed. SAC, ESCP, and RE-SAC comparators each receive 8.4M interactions. The frozen v5 estimator is not retrained.

## Execution amendment

All 25 GPU training bundles were valid. The frozen audit runner had three
execution-only wiring errors: it rejected the publisher's `nnx.State`, used the
old `specialist_0` key instead of `canonical_reference`, and omitted V28's v5
estimator-module binding. Amendment 1 records these corrections; no failed
attempt wrote an audit artifact, and no seed, policy, estimator, rollout, or
decision gate changed. Valid audits are `t93224-t93228`; aggregate `t93155`
completed from their synchronized JSON manifests.
