# Actor-only robust-warm-start policy-bank confirmation

All controllers use fresh confirmation seeds. The robust source and specialists are compared under matched budgets, and each switching event uses a distinct frozen four-mode cycle.

| Seed | Holdout mode wins | Robust / safe switching | Safe gain | Switch schedules won | Seed gate |
|---:|---:|---:|---:|---:|:---:|
| 71003 | 1/4 | 2876.1 / 3553.0 | +23.5% | 3/3 | FAIL |
| 71021 | 4/4 | 2457.9 / 3856.3 | +56.9% | 3/3 | PASS |
| 71039 | 4/4 | 1155.4 / 2761.8 | +139.0% | 3/3 | PASS |
| 71057 | 4/4 | 2284.7 / 3777.0 | +65.3% | 3/3 | PASS |
| 71079 | 4/4 | 1797.2 / 2450.0 | +36.3% | 3/3 | PASS |

Seed decisions: **4/5 pass** (required 4/5).

Per-mode replication: `{"0": 4, "1": 4, "2": 5, "3": 4}`.

Mean/median switching gain: +64.2% / +56.9%.

Diagnosis: **actor_only_warmstart_policy_bank_confirmed**.

Next step: freeze these policy banks and train a causal mode estimator; evaluate it with the previously established delayed-oracle budget.
