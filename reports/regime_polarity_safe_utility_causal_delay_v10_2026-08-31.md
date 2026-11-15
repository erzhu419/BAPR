# Safe-utility causal-delay result

Tasks `t85444-t85449` completed. Five checkpoint-only audits evaluated the
frozen v9 controller banks on unused event seeds `156201`, `156217`, and
`156233`; the sixth task aggregated only the synchronized JSON manifests. No
model was trained and no checkpoint was downloaded for this diagnostic.

| Training seed | Robust | Zero-delay safe oracle | Posterior MAP | Oracle headroom | MAP recovery | Best delay-4 retention |
|---:|---:|---:|---:|---:|---:|---:|
| 61003 | 897.9 | 3085.5 | 2848.6 | +243.6% | 89.2% | 94.7% |
| 61021 | 2236.1 | 2564.7 | 2374.2 | +14.7% | 42.0% | 85.2% |
| 61039 | 2367.5 | 2367.5 | 2367.5 | 0.0% | n/a | n/a |
| 61057 | 2739.2 | 2739.2 | 2739.2 | 0.0% | n/a | n/a |
| 61079 | 1312.8 | 1704.2 | 1983.3 | +29.8% | 171.3% | 121.2% |

The preregistered headroom gate fails: only `3/5` policy banks have at least
10% safe-oracle headroom, rather than the required `4/5`. Delay-4 retains the
required 70% of the safe-oracle margin on all three positive-headroom banks,
so a realistic short inference delay is not the common failure. Posterior MAP
recovers the required margin on only `2/5` banks, but estimator retraining is
not authorized because the policy-bank headroom prerequisite failed first.

The two exact zero-gain rows are intentional robust fallbacks, not an audit
bug. Their frozen v9 calibration maps are `[R,R,R,R]`: on seeds `61039` and
`61057`, every matching stationary specialist lost to its matched robust SAC
controller under the independent calibration streams. Training-log inspection
also found no simple final-checkpoint collapse that could be repaired by
selecting the preceding save. The ordinary online `Eval` metric is not a
fixed-mode validation metric and is low even for the successful seed `61003`,
so it cannot be used for specialist checkpoint selection.

Diagnosis: **policy-bank headroom is not reproducible**. The next development
screen must target controller optimization rather than the estimator, gate, or
environment: start each fixed-mode controller from its matched robust actor,
allow the full controller to adapt, and compare full-state versus actor-only
initialization on the failed development banks. Any continuation requires at
least `3/4` stationary mode wins over robust on both banks before switching or
learned-posterior evaluation.

The machine-readable source of record is
`jax_experiments/results_regime_polarity_safe_utility_causal_delay_analysis_v10/analysis.json`.
