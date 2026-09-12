# Polarity posterior screen

Decision: **estimator_failed**.

| Arm | Switching mean | Delta vs robust | Seed wins | Oracle headroom recovered |
|---|---:|---:|---:|---:|
| robust | 1023.4 | +0.0 | - | - |
| oracle | 2327.7 | +1304.3 | - | - |
| learned_soft | 799.4 | -224.0 | 1/5 | -22.6% |
| learned_map | 727.8 | -295.6 | 0/5 | -27.4% |

## Causal inference

| Metric | Learned soft | Gate |
|---|---:|---:|
| Mode accuracy | 0.537 | >= 0.85 |
| Median switch delay | 10.0 | <= 25 |
| P90 switch delay | 369.4 | <= 50 |
| Brier score | 0.839 | <= 0.25 |

Inference gate: **FAIL**. Posterior-conditioned SAC training: **BLOCKED**.
