# Specialist-trajectory expected-action system ID v5

The confirm3 router is frozen from v3. Only the expected-action model and posterior filter are fitted on independent-specialist trajectories.

## Source seed 4021 (training policy bank)

| Arm | Return | Gain | Oracle recovery | Robust actions | Mode accuracy | Event wins |
|---|---:|---:|---:|---:|---:|---:|
| `robust_sac` | 1620.9 | 0.0% | 0.0% | 100.00% | - | 0/3 |
| `dynamic_oracle` | 2587.2 | 59.6% | 100.0% | 0.00% | 100.00% | 3/3 |
| `posterior_map_no_gate` | 2717.4 | 67.7% | 113.5% | 0.00% | 99.11% | 3/3 |
| `posterior_sticky_confirm3` | 2584.1 | 59.4% | 99.7% | 0.30% | 98.42% | 3/3 |

Seed pass: **True**

## Source seed 4049 (held-out policy bank)

| Arm | Return | Gain | Oracle recovery | Robust actions | Mode accuracy | Event wins |
|---|---:|---:|---:|---:|---:|---:|
| `robust_sac` | 1467.7 | 0.0% | 0.0% | 100.00% | - | 0/3 |
| `dynamic_oracle` | 3116.6 | 112.3% | 100.0% | 0.00% | 100.00% | 3/3 |
| `posterior_map_no_gate` | 2989.4 | 103.7% | 92.3% | 0.00% | 99.09% | 3/3 |
| `posterior_sticky_confirm3` | 2746.1 | 87.1% | 77.5% | 0.40% | 98.35% | 3/3 |

Seed pass: **True**

Held-out policy-bank pass: **True**

Primary pass: **True**

Diagnosis: specialist-trajectory system ID closes the frozen confirm3 loop.

Decision: freeze v5 and train fresh independent source banks.
