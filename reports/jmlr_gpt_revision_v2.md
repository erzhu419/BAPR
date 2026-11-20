# BAPR JMLR Revision

This revision addresses the internal review of `f2cad34`. All new rollout
diagnostics reuse frozen checkpoints. None adds a trained policy or estimator.

1. **Contribution.** The paper now relates BAPR to virtual actuators,
   Grounded Action Transformation and TeAR. Its main question is when
   reference-only identification and analytic action compensation can reuse
   a learned controller. Reacher and MuJoCo shared-estimator comparisons lead
   the evidence. Those comparisons still differ in policy training regime.
2. **Frozen interface.** Five paired seeds on each of HalfCheetah and Walker2d
   compare hard MAP, posterior-mean command and true-mode compensation.
   HalfCheetah soft minus hard is +48.4 with interval [-16.8, 113.6].
   Walker2d soft leaves 41 of 70 reached switches without sustained correct
   estimates, versus 11 of 94 for hard. True mode itself falls in 64 of 75
   episodes. This separates reference-policy fragility from imperfect inference
   without claiming a causal explanation of individual falls.
3. **Mismatch and theory.** With nominal gain 0.50, causal compensation
   beats the uncorrected actor on all five HalfCheetah seeds at actual gains
   0.35, 0.50 and 0.65. Projection optimality is about signal tracking, not
   optimal task return. The ideal zero-hazard recurrence is distinct from
   the implemented probability floor. The eight-step empirical sufficient
   event was not met on any of the 450 tested Brax windows at epsilon 0.1.
4. **Filter and estimator.** With matched marginal bit hazard and the 1/6
   score normalization, factored minus joint return in the Brax replay is
   +0.3 [-151.7, 152.2], not a resolved deficit. This is a sensitivity result,
   not a tuned replacement default. The original shared-forward contrasts
   do not establish inverse-model return superiority; Ant changes with
   numerical replay.
5. **Verification.** The versioned supplement separates original seed
   summaries from 20 MuJoCo and 10 Brax replay audits plus the frozen
   interface experiments. Only compact returns, episode diagnostics and
   scientific metadata are released. The checkers independently recompute
   counts and paired statistics from those records.

## Second Review Follow-Up

The abstract and introduction now distinguish reference-only MuJoCo
calibration from the four-mode Reacher datasets. The MuJoCo budget matches
direct collection, not access to the shared estimator's fitting data.
The episode checker reproduces the same-opportunity Walker2d counts
(5/59 vs. 35/59 unresolved), equal-window recovery (20/20 vs. 13/20), and
matched-minus-nominal oracle intervals at gains 0.35 and 0.65. These are
reanalyses of the unchanged versioned records. The public audit environment
pins SciPy 1.15.3 and the other direct analysis dependencies.
