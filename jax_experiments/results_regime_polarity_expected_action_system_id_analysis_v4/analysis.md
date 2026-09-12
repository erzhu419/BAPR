# Polarity posterior screen

Decision: **direct_posterior_policy_training_authorized**.

| Arm | Switching mean | Delta vs robust | Seed wins | Oracle headroom recovered |
|---|---:|---:|---:|---:|
| robust | 1023.4 | +0.0 | - | - |
| oracle | 2327.7 | +1304.3 | - | - |
| learned_soft | 2171.3 | +1147.9 | 5/5 | 86.2% |
| learned_map | 2112.1 | +1088.7 | 5/5 | 82.7% |

## Causal inference

| Metric | Learned soft | Gate |
|---|---:|---:|
| Mode accuracy | 0.998 | >= 0.85 |
| Median switch delay | 5.0 | <= 25 |
| P90 switch delay | 12.0 | <= 50 |
| Brier score | 0.004 | <= 0.25 |

Inference gate: **PASS**. Posterior-conditioned SAC training: **AUTHORIZED**.
