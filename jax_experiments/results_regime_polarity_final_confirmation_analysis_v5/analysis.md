# Polarity posterior screen

Decision: **estimator_passed_but_frozen_control_failed**.

| Arm | Switching mean | Delta vs robust | Seed wins | Oracle headroom recovered |
|---|---:|---:|---:|---:|
| robust | 1340.1 | +0.0 | - | - |
| oracle | 2283.5 | +943.3 | - | - |
| learned_soft | 2161.5 | +821.4 | 3/5 | undefined |
| learned_map | 2156.1 | +816.0 | 3/5 | undefined |

Oracle-headroom recovery is undefined for the full sealed split because oracle does not beat robust on seed(s) 1031. These seeds remain in every paired control gate.

## Causal inference

| Metric | Learned soft | Gate |
|---|---:|---:|
| Mode accuracy | 0.995 | >= 0.85 |
| Median switch delay | 4.0 | <= 25 |
| P90 switch delay | 10.0 | <= 50 |
| Brier score | 0.008 | <= 0.25 |

Inference gate: **PASS**. Posterior-conditioned SAC training: **BLOCKED**.
