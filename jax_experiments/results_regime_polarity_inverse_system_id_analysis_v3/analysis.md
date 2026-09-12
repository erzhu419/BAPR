# Polarity posterior screen

Decision: **direct_posterior_policy_training_authorized**.

| Arm | Switching mean | Delta vs robust | Seed wins | Oracle headroom recovered |
|---|---:|---:|---:|---:|
| robust | 1023.4 | +0.0 | - | - |
| oracle | 2327.7 | +1304.3 | - | - |
| learned_soft | 2190.5 | +1167.0 | 5/5 | 89.3% |
| learned_map | 2135.1 | +1111.6 | 5/5 | 84.5% |

## Causal inference

| Metric | Learned soft | Gate |
|---|---:|---:|
| Mode accuracy | 0.999 | >= 0.85 |
| Median switch delay | 5.0 | <= 25 |
| P90 switch delay | 11.0 | <= 50 |
| Brier score | 0.001 | <= 0.25 |

Inference gate: **PASS**. Posterior-conditioned SAC training: **AUTHORIZED**.
