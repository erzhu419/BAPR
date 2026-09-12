# Frozen-base independent-adapter development screen

This is a one-training-seed development screen. The five sealed event streams are paired disturbance replicates, not independent policy-training seeds; their intervals are descriptive only.

| Delta | Utility map | Robust switch | Utility switch | Delta | Relative | Wins | Gate |
|---:|---|---:|---:|---:|---:|---:|:---:|
| 0.25 | `[0, 0, 2, -1]` | 2212.0 | 2558.9 | +346.9 | +15.7% | 5/5 | fail |
| 0.50 | `[0, 1, 2, 3]` | 2212.0 | 2569.0 | +357.0 | +16.1% | 5/5 | pass |
| 1.00 | `[0, 1, 2, 3]` | 2212.0 | 2544.2 | +332.2 | +15.0% | 5/5 | pass |

Decision: `expand_selected_delta_to_five_training_seeds`.

Promotion requires switching gain >=10%, stationary gain >=5%, at least 4/5 paired event wins, utility routing above every fixed adapter, and termination no worse than robust by more than 0.05. A passing delta must still be rerun over five independent training seeds before any paper claim or learned-estimator training.
