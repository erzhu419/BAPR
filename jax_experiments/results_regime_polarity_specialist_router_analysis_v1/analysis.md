# Causal independent-specialist router development screen

This checkpoint-only screen reuses the two independent specialist banks, the frozen expected-action estimator, and the frozen causal fallback. True mode is available only to `dynamic_oracle`.

## Source seed 4021

| Arm | Stationary | Worst mode | Switching | Fallback |
|---|---:|---:|---:|---:|
| `robust_sac` | 1708.7 | 1296.3 | 1501.4 | - |
| `dynamic_oracle` | 2578.9 | 1181.4 | 2595.2 | - |
| `posterior_map_fallback` | 1180.8 | 408.2 | 984.0 | 29.51% |
| `posterior_soft_fallback` | 1086.6 | 287.9 | 862.5 | 31.45% |

- `posterior_map_fallback`: switching recovery -47.3%, stationary recovery -60.7%, event wins 0/3, pass=False.
- `posterior_soft_fallback`: switching recovery -58.4%, stationary recovery -71.5%, event wins 0/3, pass=False.

## Source seed 4049

| Arm | Stationary | Worst mode | Switching | Fallback |
|---|---:|---:|---:|---:|
| `robust_sac` | 1605.7 | 1499.2 | 1679.6 | - |
| `dynamic_oracle` | 2788.5 | 2323.7 | 2989.0 | - |
| `posterior_map_fallback` | 1072.4 | 672.1 | 1085.0 | 23.30% |
| `posterior_soft_fallback` | 1102.9 | 554.1 | 1200.7 | 24.30% |

- `posterior_map_fallback`: switching recovery -45.4%, stationary recovery -45.1%, event wins 0/3, pass=False.
- `posterior_soft_fallback`: switching recovery -36.6%, stationary recovery -42.5%, event wins 0/3, pass=False.

Router gate pass: **False**

Selected development arm: `None`.

Decision: do not spend GPU budget on this specialist-router construction.
