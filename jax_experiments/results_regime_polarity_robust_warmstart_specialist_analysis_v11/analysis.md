# Robust-warm-started specialist development result

The matched robust actor is the only shared initialization. No estimator, gate, residual, or environment parameter is changed.

| Variant | Seed | Holdout mode wins | Robust / safe switching | Safe gain | Switching event wins | Cell gate |
|---|---:|---:|---:|---:|---:|:---:|
| full_state | 61039 | 4/4 | 2429.6 / 3699.4 | +52.3% | 3/3 | PASS |
| full_state | 61057 | 4/4 | 2730.9 / 4028.7 | +47.5% | 3/3 | PASS |
| actor_only | 61039 | 4/4 | 2429.6 / 3184.6 | +31.1% | 3/3 | PASS |
| actor_only | 61057 | 4/4 | 2730.9 / 3952.9 | +44.7% | 3/3 | PASS |

Variant decisions: `{"actor_only": true, "full_state": true}`.

Diagnosis: **robust_warmstart_candidate_supported**.

Next step: freeze the simpler passing initialization and run a completely new five-seed policy-bank confirmation.

The gate requires at least 3/4 independently held-out stationary mode wins and at least 10% safe-oracle switching gain on all three switching streams for both development seeds.
