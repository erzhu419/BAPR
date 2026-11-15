# V19 specialist training stability result

This fresh-seed screen changes only specialist initialization. The environment, posterior, gate, routing policy, and v18 holdout remain frozen.

| Variant | Seed | Holdout mode wins | Robust / safe switching | Safe gain | Event wins | Cell gate |
|---|---:|---:|---:|---:|---:|:---:|
| actor_only_control | 82003 | 3/4 | 1736.8 / 2489.2 | +43.3% | 3/3 | PASS |
| actor_only_control | 82021 | 4/4 | 1847.0 / 3163.8 | +71.3% | 3/3 | PASS |
| actor_only_control | 82039 | 4/4 | 1677.5 / 2434.0 | +45.1% | 3/3 | PASS |
| full_state | 82003 | 4/4 | 1736.8 / 3192.7 | +83.8% | 3/3 | PASS |
| full_state | 82021 | 4/4 | 1847.0 / 3101.0 | +67.9% | 3/3 | PASS |
| full_state | 82039 | 4/4 | 1677.5 / 2579.9 | +53.8% | 3/3 | PASS |
| critic_warmup | 82003 | 4/4 | 1736.8 / 3016.9 | +73.7% | 3/3 | PASS |
| critic_warmup | 82021 | 4/4 | 1847.0 / 2992.3 | +62.0% | 3/3 | PASS |
| critic_warmup | 82039 | 4/4 | 1677.5 / 2268.4 | +35.2% | 3/3 | PASS |

| Candidate | Mean delta vs control | Seed wins | Event wins | Min stationary retention | Worst gain candidate / control | Decision |
|---|---:|---:|---:|---:|---:|:---:|
| full_state | +262.2 | 2/3 | 7/9 | 85.2% | +53.8% / +43.3% | FAIL |
| critic_warmup | +63.5 | 1/3 | 3/9 | 85.0% | +35.2% / +43.3% | FAIL |

Passing candidates: `[]`.

Selected candidate: `None`.

Diagnosis: **initialization_does_not_resolve_policy_bank_variance**.

Next step: do not alter the frozen posterior or v18 holdout; inspect per-mode optimization traces and close this initialization family.
