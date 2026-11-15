# V33 canonical-controller training ablation

Five new development seeds completed four equal-budget HalfCheetah actuator-polarity recipes (8.4M policy interactions and 525k updates per seed). The frozen v5 estimator and three stationary plus three switching event streams were used for every audit. All five audit manifests validate locally, and local aggregation reproduces the synced analysis exactly.

| Training recipe | Switching oracle | Switching causal | Stationary causal | Causal vs robust | Causal vs warm-start |
|---|---:|---:|---:|---:|---:|
| Robust multimode SAC | -- | 2085.5 | 2192.0 | reference | -- |
| 5.6M multimode + 2.8M mode-0 warm-start | 3296.4 | 3161.3 | 3264.6 | +1075.8 (5/5 seeds) | reference |
| Direct mode-0, 8.4M | 3459.4 | 3323.2 | 3392.3 | +1237.8 (5/5) | +162.0 (3/5) |
| Canonicalized multimode, 8.4M | 2464.1 | 2440.1 | 2438.2 | +354.7 (4/5) | -721.1 (0/5) |

Direct mode-0 improves the causal mean by 162.0 over warm-start, but the paired 95% interval is [-480.3, 804.3] and it wins only 3/5 seeds. Its oracle gain over warm-start is similar to its causal gain; mean inference loss is 136.1 versus 135.2 for warm-start. It therefore shows possible controller headroom, not a consistent recipe improvement. Canonicalized multimode loses to warm-start in every seed. Oracle trajectory equivalence passes for all three compensated recipes; stationary causal-to-oracle retention passes in 5/5 seeds for direct and warm-start, and 4/5 for canonicalized.

Under the frozen rule requiring at least 4/5 paired causal wins, the selected development recipe remains **warm-start mode 0**. This exploratory selection does not modify V32's positive average effects or its failed composite confirmation.

In this actuator-polarity family, all modes share gravity, noise scale, and other physical parameters. Canonicalized training replaces each mode's actuator gain with the reference gain, so the post-transform physical control problem is the same as fixed mode 0; the hidden mode clock and task labels remain different. Direct and canonicalized policy files are identical for seeds 89021 and 89039. Their training-time `Reward/Eval` logs use different recipe-specific evaluation interfaces; the table above uses the common strict audit, which disables both fixed-mode and canonical-training interfaces. This arm does not provide additional physical-task diversity, and its lower mean should not be interpreted as evidence against canonicalization in environments with other mode-dependent physics.

The original CPU audits for seeds 89003 and 89039 failed before evaluation because their scheduler specs omitted the source policy bundle from remote staging. Corrected retry specs t100543 and t100544 included that compact bundle and completed; aggregate t100363 then completed. No GPU training was repeated. Future V33 audit submissions must stage the source bundle used by warm-start provenance validation.

Compact evidence: `jax_experiments/results_regime_polarity_canonical_training_audit_v33/seed_*/audit.json` and `jax_experiments/results_regime_polarity_canonical_training_analysis_v33/analysis.json`. No full checkpoint or replay buffer is needed for these comparisons.
