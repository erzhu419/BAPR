# Sticky independent-specialist router screen

The primary arm confirms a new posterior mode for three consecutive transitions and then switches specialists atomically. It uses the robust controller only before the first option is identified.

## Source seed 4021

| Arm | Return | Gain | Oracle recovery | Robust actions | Mode accuracy | Event wins |
|---|---:|---:|---:|---:|---:|---:|
| `robust_sac` | 1550.8 | 0.0% | 0.0% | 100.00% | - | 0/3 |
| `dynamic_oracle` | 2764.3 | 78.3% | 100.0% | 0.00% | 100.00% | 3/3 |
| `posterior_map_no_gate` | 2113.1 | 36.3% | 46.3% | 0.00% | 89.03% | 3/3 |
| `posterior_sticky_confirm2` | 2265.1 | 46.1% | 58.9% | 0.49% | 90.08% | 3/3 |
| `posterior_sticky_confirm3` | 2152.4 | 38.8% | 49.6% | 0.49% | 92.27% | 3/3 |
| `posterior_sticky_confirm5` | 2061.3 | 32.9% | 42.1% | 0.49% | 93.86% | 3/3 |

## Source seed 4049

| Arm | Return | Gain | Oracle recovery | Robust actions | Mode accuracy | Event wins |
|---|---:|---:|---:|---:|---:|---:|
| `robust_sac` | 1546.7 | 0.0% | 0.0% | 100.00% | - | 0/3 |
| `dynamic_oracle` | 3149.6 | 103.6% | 100.0% | 0.00% | 100.00% | 3/3 |
| `posterior_map_no_gate` | 2251.0 | 45.5% | 43.9% | 0.00% | 90.02% | 3/3 |
| `posterior_sticky_confirm2` | 2329.5 | 50.6% | 48.8% | 0.44% | 92.60% | 3/3 |
| `posterior_sticky_confirm3` | 2309.9 | 49.3% | 47.6% | 0.44% | 93.01% | 3/3 |
| `posterior_sticky_confirm5` | 2092.7 | 35.3% | 34.1% | 0.44% | 93.42% | 3/3 |

Primary confirm3 pass: **False**

Ungated MAP useful on both source seeds: **True**

Diagnosis: mode information is useful, but frozen estimator trajectories remain insufficient for the registered sticky router.

Decision: fit the estimator on specialist-policy trajectories without changing the frozen confirm3 option.
