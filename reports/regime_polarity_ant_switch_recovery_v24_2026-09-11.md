# Ant switch-state recovery v24

V24 completed all 24 unique GPU training cells, six CPU audits, and the
registered aggregate. Each final bundle is at iteration 2099 with 11.2M
physical environment steps and 525k updates. Frozen robust-policy equivalence
passed in every bundle with zero action error.

| Variant | Safe switching | Robust | Relative gain | Stationary cells | Switching termination | Seed gates |
|---|---:|---:|---:|---:|---:|---:|
| `switch_state` | 3849.9 | 2546.9 | +49.8% | 7/12 | 35.6% | 0/3 |
| `switch_state_risk` | 3287.7 | 2546.9 | +28.7% | 6/12 | 20.0% | 0/3 |

The registered decision is **FAIL** for both variants. Switch-state training
creates large return gains, but the resulting Ant specialists remain unsafe
and seed-dependent. A scalar 500-point terminal penalty reduces aggregate
termination but does not consistently rank or suppress risky actions before
failure.

The next controlled screen should retain the immutable robust actor and train
an explicit discounted termination-risk critic. The actor constraint should
penalize only predicted risk above the robust action at the same state, so the
objective targets incremental catastrophic risk instead of globally depressing
return. This is a controller experiment; posterior or gate tuning remains out
of scope until the true-mode policy bank is stable.
