# Reacher Same-Information Independent Confirmation

Freeze the V37 implementation and both comparisons before launch. No model or
filter tuning. New policy seeds: 99007/99027/99047/99067/99087; switching events
209601/209617/209633, five paired episodes each; stationary base 209701, five
episodes per mode. Earlier development/confirmation seeds are not pooled.

- Four controllers per seed: native-reference, memoryless robust, filtered-belief
  conditioned SAC, instantaneous-belief conditioned SAC. Twenty independent GPU
  trainings, one remote smoke, five CPU audits and one aggregate (27 tasks).
- Keep 400 iterations/1.6M interactions/100,000 updates, hidden width 256,
  native generalized Reacher reward/termination, horizon 200, dwell 50, noise 0.02.
  Use the admitted CPU sampler with GPU updates and frozen JAX 0.4.38,
  Brax 0.10.5, Flax 0.10.4, MuJoCo 3.1.6 runtime.
- Reuse inverse 95103 and filter controls (0.002,1.0,0.98). Both matched routes
  have the same estimator information and extra historical provider/calibration
  cost. Conditioned SAC uses pre-action belief and outputs the command directly.
- Primary outcome: switching return difference, filtered compensation minus
  filtered conditioned SAC. A positive paired 95% CI across five policy seeds
  confirms this comparison; report wins without imposing an all-five-win rule.
  Instantaneous compensation minus instantaneous conditioned SAC is the predefined
  secondary comparison, reported separately. No post-hoc promotion of a secondary
  to primary. Retain stationary, oracle/native, fixed 0-3, robust and head-0 controls.
- All switching evaluations must retain 200 steps/three switches; controller
  budgets and training/evaluation belief definitions must match. Episodes are
  repeated measurements within policy seeds, not extra independent samples.
- Scheduler only, no local; use the already matched GPU servers and CPU nodes.
  Each seed/arm has its own checkpoint and task. Resume full state safely.
  Synchronize compact actor/estimator exports and JSON only, never replay/checkpoints.

Next: report the untouched five-seed outcome and integrate a positive matched
comparison; retain negative/inconclusive results without retuning this cohort.
V32, V36, bus, manuscript and Lean remain unchanged.
