# Ant switch-recovery specialist result

The robust prefix generates real switch states. Only post-switch target-mode transitions train each specialist; the risk arm adds a fixed terminal penalty. Evaluation uses an immutable eight-step robust fallback after every switch.

| Variant | Seed | Mode wins | Robust / fallback switching | Gain | Termination | Gate |
|---|---:|---:|---:|---:|---:|:---:|
| switch_state | 85003 | 1/4 | 2340.2 / 2901.8 | +24.0% | 13.3% | FAIL |
| switch_state | 85021 | 4/4 | 2682.0 / 5405.2 | +101.5% | 40.0% | FAIL |
| switch_state | 85039 | 2/4 | 2618.3 / 3242.7 | +23.8% | 53.3% | FAIL |
| switch_state_risk | 85003 | 1/4 | 2340.2 / 2884.9 | +23.3% | 26.7% | FAIL |
| switch_state_risk | 85021 | 2/4 | 2682.0 / 3972.4 | +48.1% | 26.7% | FAIL |
| switch_state_risk | 85039 | 3/4 | 2618.3 / 3005.8 | +14.8% | 6.7% | FAIL |

Passing variants: `[]`.

Selected candidate: `None`.

Registered gate: **FAIL**.

Diagnosis: **termination_risk_helps_but_ant_bank_remains_unstable**.

Next step: retain the immutable robust fallback and replace scalar terminal penalty with a learned constrained termination-risk critic.
