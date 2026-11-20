# BAPR mechanism proofs

This standalone Lean 4.28.0 / Mathlib 4.28.0 package contains the fifteen
environment-independent mechanism modules used by the manuscript and extended
theory notes. It does not depend on the sibling proof repository or include
Spring/HalfCheetah simulator certificates. Sources retain their original
`BAPR.ActionCompensation` namespace. The dependency manifest pins Mathlib and
its transitive dependencies; build caches are not part of the release.

With Lean's Elan toolchain manager installed, run from this directory:

```bash
lake exe cache get
lake build
```

The default build checks every included module. The manuscript's core results
correspond to:

| Manuscript result | Module and theorem |
| --- | --- |
| Feasible compensation and pathwise equivalence | `ActionCompensation`: `exact_compensation_iff_feasible`, `arbitrary_polarity_switching` |
| Action-weighted mode error and clipping | `ActionCompensation`: `action_weighted_mode_error`, `modeError_l2Norm`, `executed_l2_nonexpansive` |
| Causal action selection | `ActionCompensation`: `causalCommand_prefix`; `ActionCompensationMeasure`: `measurable_causal_action` |
| Finite-horizon return identity on causal occupancies | `ActionCompensationClosedLoop`: `occupancy_return_difference` |
| Conditional polarity return bound | `ActionCompensationMeasure`: `measure_polarity_return_bound`, `measure_initial_return_bound` |
| Tempered sticky filter stability, finite-dwell switch recovery, and quiet retention | `ActionCompensationFilter`: `temperedFilter_score_stability`, `temperedFilter_switch_recovery`; `ActionCompensationEvidence`: `inverseExcitationMargin_polarity`; `ActionCompensationRetention`: `temperedFilter_quiet_window` |
| Lossy gain box projection, infeasibility-plus-inference decomposition, `δ < 1` log-odds cap, and `O(d)` MAP exactness for the separable channel-product filter | `LossyProjectionAndFactoredFilter`: `weightedBoxProjection_optimal`, `lossyProjection_infeasibility_residual`, `lossy_two_term_return_decomposition`, `untempered_switch_delay_scales_with_dwell`, `tempered_log_odds_uniform_cap`, `channelFactored_map_maximizes_joint`, `channelFactored_map_unique_recovery` |

The other modules give conditional filter, evidence, recovery, calibrated-belief,
KL and Gaussian-smoothing extensions. They are supplementary results, not new
empirical claims or prerequisites for the six numbered manuscript results.

## Scope

Compensation assumes known feasible transforms and common transition/reward
maps. Return results retain measurability, integrability and action-value
sensitivity premises. The recursive Lean Bellman API uses a sufficient
`BellmanIntegrable` condition over all states, counterfactual signals and finite
remaining horizons, stronger than the manuscript's visited-law conditions.
The formalized sufficient case therefore does not claim to check the weakest
version of every manuscript assumption. Recovery extensions additionally assume
prediction-error moments and excitation on the causal rollout law. Gaussian
noise alone does not establish these premises, calibrated beliefs, or superiority
over another learned controller. This package does not certify Brax's floating-
point implementation or infer experimental success from kernel checking.

The full theorem-to-code inventory is in
[`../v32/PROOF_SCOPE.md`](../v32/PROOF_SCOPE.md).
