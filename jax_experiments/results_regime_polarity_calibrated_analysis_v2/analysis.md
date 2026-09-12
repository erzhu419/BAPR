# Polarity posterior screen

Decision: **estimator_failed**.

| Arm | Switching mean | Delta vs robust | Seed wins | Oracle headroom recovered |
|---|---:|---:|---:|---:|
| robust | 1023.4 | +0.0 | - | - |
| oracle | 2327.7 | +1304.3 | - | - |
| learned_soft | 746.3 | -277.2 | 1/5 | -25.8% |
| learned_map | 602.8 | -420.7 | 0/5 | -38.6% |

## Causal inference

| Metric | Learned soft | Gate |
|---|---:|---:|
| Mode accuracy | 0.537 | >= 0.85 |
| Median switch delay | 23.0 | <= 25 |
| P90 switch delay | 250.0 | <= 50 |
| Brier score | 0.530 | <= 0.25 |

Inference gate: **FAIL**. Posterior-conditioned SAC training: **BLOCKED**.
