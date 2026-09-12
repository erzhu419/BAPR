# Independent-specialist router failure decomposition

This switching-only checkpoint audit separates estimator mode errors, one-step gate thrashing, and robust-to-specialist handoff.

## Source seed 4021

| Arm | Return | Oracle recovery | Fallback | Mode accuracy |
|---|---:|---:|---:|---:|
| `robust_sac` | 1616.5 | 0.0% | 100.00% | - |
| `dynamic_oracle` | 2792.9 | 100.0% | 0.00% | 100.00% |
| `posterior_map_fallback` | 642.9 | -82.8% | 25.91% | 96.79% |
| `true_mode_current_gate` | 810.3 | -68.5% | 28.39% | 100.00% |
| `true_mode_robust10` | 2342.6 | 61.7% | 4.00% | 100.00% |
| `posterior_map_no_gate` | 1983.7 | 31.2% | 0.00% | 88.43% |
| `posterior_debounced_option` | 911.3 | -60.0% | 34.76% | 97.84% |

## Source seed 4049

| Arm | Return | Oracle recovery | Fallback | Mode accuracy |
|---|---:|---:|---:|---:|
| `robust_sac` | 1669.0 | 0.0% | 100.00% | - |
| `dynamic_oracle` | 3001.7 | 100.0% | 0.00% | 100.00% |
| `posterior_map_fallback` | 1154.8 | -38.6% | 24.58% | 97.91% |
| `true_mode_current_gate` | 1300.5 | -27.7% | 24.98% | 100.00% |
| `true_mode_robust10` | 2885.4 | 91.3% | 4.00% | 100.00% |
| `posterior_map_no_gate` | 2213.5 | 40.9% | 0.00% | 90.14% |
| `posterior_debounced_option` | 1278.7 | -29.3% | 23.94% | 98.83% |

Handoff viable: **False**

Debounced router pass: **False**

Diagnosis: independent specialists are not robust to causal handoff states.

Decision: train switch-matched specialists from robust-state handoff curricula.
