# V32 theory, implementation, and evidence alignment

Updated 2026-10-05. The paper uses the frozen single-reference actor, transition-only
inverse ensemble, tempered sticky belief, and causal polarity compensation.

The submission now uses `paper/appendix_submission.tex` and retains only
pathwise equivalence, action-weighted mode error, and the conditional return
bound as numbered theoretical results. Their proofs and the full experimental
protocol remain in the PDF. The original `paper/appendix_v32.tex` preserves the
extended filter, measure-theoretic, and simulator-source development outside
the submission. The inventory below records those supplementary artifacts;
it is not a list of prerequisites for submitting the RL paper.

The fourteen environment-independent sources are now shipped in the standalone
[`../lean/`](../lean/) package with a Lean toolchain, Lake configuration and
dependency manifest. From that directory, `lake exe cache get` obtains the
Mathlib cache and `lake build` checks all fourteen targets. The package was
independently rebuilt on 2026-10-05 with fresh BAPR build outputs and a reused
third-party dependency cache. It needs no sibling-project source or compiled
BAPR artifact. Its return formalization uses sufficient all-state
`BellmanIntegrable` premises, stronger than the manuscript's visited-law
integrability statement; see the package README. No simulator certificate is
included in this release.

The original extended sources in sibling project `proof/BAPR` are `ActionCompensation.lean`,
`ActionCompensationBelief.lean`,
`ActionCompensationClosedLoop.lean`,
`ActionCompensationFilter.lean`,
`ActionCompensationEvidence.lean`,
`ActionCompensationRecovery.lean`,
`ActionCompensationReturn.lean`,
`ActionCompensationWindows.lean`,
`ActionCompensationCumulative.lean`,
`ActionCompensationRetention.lean`,
`ActionCompensationMeasure.lean`, `ActionCompensationKL.lean`,
`ActionCompensationKLExtended.lean`, `HalfCheetahReward.lean`,
`GaussianSmoothing.lean`, `SpringReachability.lean`, `SpringSourceComponents.lean`,
`SpringSevenLink.lean`, `SpringContactClosure.lean`, `SpringObservation.lean`,
`SpringMatrixStep.lean`, `SpringMatrixMeasurable.lean`, `SpringScalarMargins.lean`,
`SpringScalarStep.lean`, `SpringReset.lean`, `SpringContactBarrier.lean`,
`SpringPlanarInvariant.lean`, `SpringPairedFoot.lean`, `SpringFootJoint.lean`,
`SpringFootRecurrence.lean`, `SpringGlobalMomentum.lean`, `SpringCoupledEdges.lean`,
`SpringForcedResponse.lean`, `SpringSignedResponse.lean`, `SpringJointWork.lean`,
`SpringEnergyBounds.lean`, `SpringCurrentEnergy.lean`, `SpringTransientLyapunov.lean`,
`SpringModalFoot.lean`, `SpringModalDrift.lean`, `SpringAngularPrimitive.lean`,
and `SpringPitchDamping.lean`,
namespace `BAPR.ActionCompensation` (Lean/Mathlib 4.28.0).
There are forty-two named Lake targets. The environment-independent mechanism
group is `ActionCompensation`, `ActionCompensationBelief`, `ActionCompensationClosedLoop`,
`ActionCompensationFilter`, `ActionCompensationEvidence`, `ActionCompensationRecovery`,
`ActionCompensationReturn`, `ActionCompensationWindows`, `ActionCompensationCumulative`,
`ActionCompensationRetention`,
`ActionCompensationMeasure`, `ActionCompensationKL`,
`ActionCompensationKLExtended`, and `GaussianSmoothing`:

```bash
lake build ActionCompensation ActionCompensationBelief ActionCompensationClosedLoop ActionCompensationFilter ActionCompensationEvidence ActionCompensationRecovery ActionCompensationReturn ActionCompensationWindows ActionCompensationCumulative ActionCompensationRetention ActionCompensationMeasure ActionCompensationKL ActionCompensationKLExtended GaussianSmoothing
```

These fourteen targets import no HalfCheetah or spring source certificate. The remaining
twenty-eight targets retain the environment-specific sufficient-condition work.
They are auxiliary certificates, not premises of compensation equivalence or
causal deployment. General return bounds still expose integrability and
action-value sensitivity assumptions; separating the modules does not prove
those assumptions for the experimental simulator. The submission includes the
core mechanism proofs; the extended notes retain the source certificates.

- Feasible command: `exact_compensation_iff_feasible` covers invertible diagonal
  gains and command clipping; the experiment specializes these gains to signs.
- Exact oracle: `arbitrary_polarity_switching` covers arbitrary real noise paths
  with common physics, reward, and initial state. All ten audits have zero
  paired native/oracle return error.
- Causal action: `causalCommand_prefix` proves prefix non-anticipation. The V32
  evaluator selects from the previous belief before executing and observing;
  a regression test confirms that hidden labels cannot alter this selection.
  `measurable_beliefProcess` and `measurable_causal_action` additionally prove
  adaptedness to pre-action information under measurable updates and selectors.
- Mode error: `action_weighted_mode_error`, `modeError_l2Norm`, and
  `executed_l2_nonexpansive` give the same-state error and clipping bound.
- Belief calibration: `belief_squared_signal_error` gives the exact conditional
  mean squared error under any finite true mode mass. Its executed-action
  counterpart includes common noise and clipping. `beliefActionRisk_calibration`
  bounds the difference between true and belief-weighted action risk by
  `sum(q_j^2) * TV(p,b)`. The confidence corollary retains that calibration term;
  it does not identify the tempered belief with the true posterior.
  `belief_action_value_bound` derives a local, mode-uniform sensitivity bound
  `2 * L * sqrt(R_b + sum(q_j^2) * TV(p,b))` by weighted Cauchy--Schwarz.
  Continuous states/history are allowed; only the conditional mode sum is finite.
  This is not a calibration certificate inferred from aggregate Brier/accuracy.
- Closed-loop accumulation: `causal_occupancy_transport` connects the next-state
  law to the causal update by pushforward and Fubini. `occupancy_return_difference`
  derives the cumulative oracle-Q gap by telescoping; the performance-difference
  sum is not a hypothesis. `kernelModeMass_isModeMass` and
  `conditional_mode_integral` connect a genuine finite conditional kernel to the
  normalized mode sum. `closed_loop_belief_return_bound` then integrates
  `2 * L * sqrt(R_b + sum(q_j^2) * TV(p,b))` over the actual causal occupancies,
  covering switch transients and continuous state/history/noise. Its true kernel
  must be consistent with the same forward occupancy flow, and remains separate
  from the implemented belief. No simulator-specific module is imported.
- Filter stability: `temperedFilter_score_stability` proves the one-step
  log-odds discrepancy bound `delta * D + 2 * c * eta` for the actual common
  sticky mixer. `temperedFilter_error_envelope` derives its geometric sum;
  `temperedFilter_error_floor` gives `2*c*eta/(1-delta)`. The compared filters
  receive the same completed-transition/command sequence. The reference uses
  the same forgetting, not exact Bayes. `calibration_of_reference_filter` and
  `beliefActionRisk_reference_filter` retain its bias relative to the true
  conditional mode law while connecting the score bound to TV and action risk.
  `temperedFilter_switch_recovery` proves unique true-mode dominance after a
  finite dwell under a persistent positive score margin, using the prior floor
  rather than assuming an unstale incoming belief. The common softmax shifts
  and inactive frozen prior clamp are also proved in real arithmetic. Learned
  score accuracy, excitation and reference-filter calibration remain inputs.
  The recovery-probability extension below supplies a conditional probability
  bound, not certification from aggregate estimator metrics or Gaussian noise.
- Inverse evidence: `inverseEvidence_prediction_error` derives the score budget
  `2/d * sum(e_i / variance_i)` from coordinatewise head-prediction errors;
  `logMeanExp_lipschitz` preserves it through the actual log-mixture aggregation.
  `inverse_filter_prediction_error` connects this to the filter error envelope.
  `inverseEvidence_signal_margin` derives mode separation from candidate-signal
  distance and prediction error. `inverseExcitationMargin_polarity` reduces it to
  differing actuator channels, and `inverse_filter_switch_recovery` connects the
  positive excitation margin to causal MAP recovery with a stale incoming belief.
  Clipping, tanh-output bounds and the variance floor are proved in real arithmetic.
  All heads must satisfy the prediction bounds on the causal rollout; average
  training MSE, actual excitation and variance calibration are not certified by
  these pathwise theorems. No algorithm or experiment is changed.
- Recovery probability: `inverse_filter_recovery_probability` bounds failure of
  strict true-mode MAP dominance by excitation-failure probability plus
  `sum(M_tji / e_ti^2)`, under the actual causal rollout law. The finite union
  bound requires no independence across steps, heads or coordinates.
  `inverse_error_moment_of_conditional` uses the tower identity for history-
  conditional moment bounds; boxed outputs imply squared-error integrability.
  Evidence, recursive beliefs and recovery/excitation events are measurable.
  Population moment and excitation bounds remain hypotheses; the result can
  be vacuous and does not establish posterior calibration or empirical gains.
- Recovery to return: `map_mismatch_probability` charges strict-dominance failure
  and a possible mode change after the observed window. `compensation_error_moments`
  proves same-state nominal and executed MSE <= `4*d*P(mode error)` with common
  noise and clipping. `closed_loop_mode_error_return_bound` uses the actual
  forward occupancy to derive `2*sqrt(d)*sum(L_t*error_probability_bound_t)`.
  It requires no belief calibration or conditional-mode kernel, but retains
  Bellman/value integrability, a uniform sensitivity envelope and per-decision
  error bounds on that same law. It compares with the frozen actor's oracle,
  not other independently trained algorithms.
- Random windows: `observedWindowLabel_persistent` identifies the last observed
  constant-mode suffix. `temperedFilterPath_restart` retains its actual incoming
  belief. `conditional_inverse_filter_restart_probability` applies recovery
  under the positive-mass conditional path law; `random_dwell_map_error_bound`
  averages those bounds by exact finite-fiber conditioning and separately
  charges an unobserved next-mode change. `occupancy_mode_error_bound_of_path`
  transfers the result to the same decision-state pushforward for the return
  theorem. Zero-mass cells require no certificate; uncertified cells can use
  bound one. Conditional moments and excitation remain population assumptions.
- Cumulative recovery: `inverse_cumulative_filter_recovery_probability` replaces
  the simultaneous error tube by a discounted score-deficit mean bound derived
  from second moments. `temperedFilter_cumulative_margin` retains the mixer's
  nonlinear truncation; `cumulativeMargin_deficit` propagates variable deficits.
  Intermediate weak excitation is allowed, but a quiet final step may erase this
  sufficient certificate. Population moments and excitation remain conditions;
  the bound is not uniformly tighter and does not change the frozen algorithm.
- Quiet retention: `temperedFilter_retained_margin` retains a positive global
  log-odds certificate through common sticky mixing and adverse score budgets.
  `temperedFilter_quiet_window` continues the actual incoming filter; exactly
  flat scores retain its strict MAP over every finite quiet prefix in real
  arithmetic. `inverse_quiet_retains_mode` derives flat scores from equal
  candidates, independently of prediction noise. Small excitation need not be
  flat, and an unobserved switch can leave a stale mode dominant.
- Continuous return: `measure_return_difference`, `measure_polarity_return_bound`,
  and `measureValue_project` cover probability-measure expectations and memory
  augmentation. `measureValue_eq_path_integral` links values to cumulative
  rollout rewards under independent product-path draws. The initial-return
  lemmas also allow continuous reset distributions.
- Gaussian specialization: `actuatorGaussian` is a product of scalar Gaussians
  with variance 1/2500. `actuatorGaussian_marginal` verifies each marginal;
  `gaussian_return_difference` specializes the return identity, while
  `polarity_return_distribution` transfers exact pathwise equivalence to laws.
- Information: `compProd_density_factor` and `compProd_llr` derive the joint
  likelihood factorization. `extended_controlled_trajectory_kl` proves the full
  continuous trajectory chain rule in extended nonnegative reals, including
  infinite KL and shared-policy cancellation, without log-integrability assumptions.
  Countably generated action/observation spaces and conditional absolute continuity
  at every history-action pair remain. `controlled_trajectory_kl` is the finite case.
  It permits deterministic policies and clipped noise without Lebesgue densities.
- Sufficient conditions: `bellmanIntegrable_of_bounded_reward` and
  `noisePathReturn_integrable_of_bounded_reward` derive integrability from uniform
  reward bounds and measurable integrands. `measureQ_sensitivity_of_executed_sensitivity`
  propagates executed-action sensitivity through clipping and noise averaging.
- HalfCheetah: `halfCheetah_return_telescopes` proves
  `return = 20 * displacement - 0.1 * executed_action_energy` with energy in `[0, 6H]`.
  `halfCheetah_return_integrable_iff` reduces return integrability to the first
  absolute moment of displacement for uninterrupted rollouts. The actual wrapper
  has no health termination; the CPU smoke traversed four modes over 32 steps and
  passed. This is a reward-contract check, not a reachability/moment certificate.
- Contact branch: `approachImpulse_not_lipschitz` proves the normalized strict
  approach branch is not globally Lipschitz. It does not assert that averaged Q is
  nonsmooth. The audited source is `brax/spring/collisions.py` in local `bapr-jax`
  (0.10.5) and `resac-jax` (0.14.2);
  the reward implementation is `_build_core_fns` in `jax_experiments/envs/brax_env.py`.
- Gaussian smoothing: `actuatorGaussian_Q_sensitivity` gives `L = 50B` for a
  measurable continuation integrand bounded by B on clipped actions. The proof
  derives the multivariate Gaussian affinity and L1 shift bound, proves the
  independent-Gaussian density representation, and connects to `measureQ`.
  No continuity of the raw dynamics or integrand is assumed.
- Reachability: `halfCheetah_return_integrable_of_reachable_envelope` derives
  integrability from invariant preservation and a one-step polynomial size
  envelope, uniformly over clipped actions. `spring_rotation_denominator` and
  `normalImpulse_bound` discharge two component-level denominator/growth facts.
  The expanded CPU test passed for positive mass/inertia, only ground-capsule
  contact pairs, finite states and unit rotations over 32 steps / four modes.
- Source components: coordinate bounds cover rotation, ground-capsule contact
  geometry, revolute and planar-root forces/torques, inertia application, wrench
  shifts, point velocities, finite sums, and semi-implicit integration.
  `sourceContact_denominator` derives positivity for full-matrix rotated diagonal
  inertia, not for the installed flattened scalar gather. Further lemmas
  bound the six clipped motor torques by 120, cap friction and contact averaging,
  and preserve unit quaternions after normalization. The focused CPU test
  `test_v32_spring_source_components` passed 16 source/formula cases covering
  all six revolute joints and the planar root. It checks translation against
  installed Brax/MJX formulas, not a global reachability certificate.
- Seven-link composition: `sevenJointCaches_bound` derives frame/motion bounds
  and unit rotations through COM/world and parent/child/joint reconstruction.
  `sevenWrenchAssembly_bound` uses the actual parent map `[-1,0,1,2,0,4,5]`,
  including the dropped world-parent reaction and at most two incoming children.
  `sevenJointAcceleration_bound` combines the explicit root/revolute formulas,
  six motor bounds, wrench assembly and rotated inertia, without assuming a
  bound on the total joint force. `springSignedAngle_scaling` justifies using
  an unnormalized line of nodes in the real-arithmetic angle formula.
  `sevenReconstructedStep_bound` connects integration and all frame/motion
  caches, conditional on acceleration and collision increment envelopes.
  Eight source/formula cases passed through the actual `pipeline.step` outputs.
  The test caught and corrected the parent-motion shift sign in the Lean
  translation; Brax and the frozen experimental implementation were unchanged.
- Complete contact increments: `sevenScalarGroundResolver_bound` models the
  installed flattened root-inertia gather, contact masks, the `safe_norm` branch,
  friction threshold, active-contact averaging and final full-matrix conversion.
  It bounds linear and angular delta-velocity under explicit normal-denominator
  margins and positive tangential denominators. `sevenGroundResolver_bound` is
  the separate full-matrix model, whose margins follow from positive mass and PSD
  inertia. `springCounterFlatEntry_negative` and
  `springScalar_tangent_singularity` prove the scalar-gather obstruction.
  `springScalarContactDenominator_lt_mass` proves that a negative scalar and a
  nonzero lever-direction cross product strictly violate the inverse-mass lower bound.
- Reduced coordinates and observation: `halfCheetahReducedQ_bound`,
  `halfCheetahReducedQD_bound`, and `halfCheetahObservation_bound` cover the nine
  positions, nine velocities and actual 17-coordinate observation. The root's
  angular inverse rotation and the absence of observation velocity clipping are
  explicit. `test_v32_spring_contact_observation` passed 12 contact cases and 12
  inverse/observation cases against the installed source.
- Composed matrix-contact model: `sevenMatrixStep_bound` reconstructs COM/world
  frames, joint caches, rotated inertia, capsule endpoints and forces, applies
  acceleration before contact resolution, then performs damping and integration.
  It derives a complete one-step radius without assuming acceleration, collision
  increment or effective-denominator margins. `matrix_reachable_bound` iterates
  this radius and preserves unit rotations. `matrix_halfCheetah_return_bound`
  gives `20*(radius(frames*T)+R+28*K)+0.6*T`, charging control cost only once per
  environment action. `matrix_halfCheetah_return_integrable` supplies the
  displacement moment under explicit measurability assumptions. Eight paired
  source/formula cases passed for the private matrix pipeline, including actual
  capsule geometry and pre-contact ordering; the eight original scalar-pipeline
  cases also passed. This is a separate real-arithmetic model, not a V32 rerun.
- Matrix-model measurability: `measurable_sevenMatrixStep` covers the complete
  composed update, retaining strict contact, near-zero norm and angle branches.
  `active_contact_sum_eq_masked` and `active_contact_count_eq_masked` prove the
  exact filtered-sum/indicator-mask identities used for contact averaging.
  `measurable_halfCheetahObservation` covers the reconstructed 17-coordinate
  observation. `matrix_trajectory_measurable` and
  `matrix_halfCheetah_return_measurable` derive finite-trajectory and return
  measurability from measurable initial states and action processes.
  `matrix_halfCheetah_return_integrable_of_measurable_inputs` combines these
  results with the derived radius, removing separate displacement/energy
  measurability premises. No contact-continuity assumption is introduced.
- Scalar geometric margins: `scalar_matrixContacts_normal_margin` derives
  `den_normal >= 1/4` for every unit-rotation COM state from explicit fixed
  inertia, inverse-mass and capsule geometry conditions, independent of depth.
  `scalar_matrixContacts_normal_impulse_bound` removes the normal-margin premise
  from the impulse bound. `scalar_matrixContacts_tangent_margin` derives the
  same margin on penetrating contacts when the affected COM height is at least
  `-0.5`. The friction-direction squared-norm bound includes the near-zero branch.
  Source checks passed 120 normal cases and 80 height-conditioned tangent cases;
  minimum denominators were 0.282963 / 0.277251. Fixed parameters were checked
  on the current local runtime, not an immutable historical snapshot.
- Composed scalar step: `sevenScalarStep_bound` connects the flattened gather,
  active-contact-only tangential margins, geometry, joints, collision increments,
  and integration without an unspecified full-step envelope. `scalar_reachable_bound`
  and `scalar_halfCheetah_return_bound` cover finite safe prefixes. Vertical
  collision increments are nonnegative; `scalarStep_height_forall_actions` gives
  a sufficient pre-contact position-velocity criterion for every current command,
  using the proved action independence of instantaneous translational acceleration.
  Twelve source steps / 84 body cases passed the vertical balance comparison
  (maximum error 2.73e-8); all twelve opposite-command force comparisons matched.
- Reset and first substep: `halfCheetahReset_stateBound` supplies initial COM
  radius 3 and unit rotations from bounded uniform position/velocity noise and
  COM offsets of norm at most 1/5. `halfCheetahReset_height` derives initial
  heights >= -1/3. `halfCheetahReset_first_step_safe` proves every body's height
  remains >= -1/2 for the first physics substep, for all motor commands, using
  the decimal seven-link geometry, zero joint offsets, joint linear damping
  bounded by 10, inverse masses in [0,4], gravity bounded by 10, dt=1/320 and
  linear integration multiplier 1. Initial joint position errors are exactly
  zero; the source's parent-motion shift yields twice the parent cross product,
  rather than zero relative linear velocity. The proved coordinate bounds are
  1 for COM velocity, 42/5 for leaf cache velocity, 84 for local linear force,
  and 14200 for linear acceleration. The focused reset test passed 12 wrapper
  draws plus four boundary states / 48 first-substep source updates, with maximum
  formula error 1.35e-7. No training or evaluation physics was changed.
- Multi-contact recovery: `scalar_matrixContacts_normal_upper` derives the
  upper denominator bound `inverse_mass+9/100` for non-diagonal contact bodies.
  `scalarActiveContacts_eq_approaching` identifies the actual active mask under
  nonnegative elasticity/stabilization. `scalarCollision_recovery_lower` sums
  the penetration/point-velocity contributions, retaining the real active count
  and averaging epsilon 1e-8. `scalar_barrier_reachable_bound` and
  `scalar_barrier_halfCheetah_return_bound` derive finite-horizon safety, radius
  and return bounds from the explicit current-state recovery condition instead
  of assumed safe heights at every later state. The height region now covers
  actual capsule-contact bodies only. Current source candidates occur twice
  each on bodies 0/1/3/4/6 and not on 2/5, so the non-diagonal height obligations
  are on 1/3/6. A focused source check passed 256 substeps from four normal resets,
  including 116 approaching and 11 separating leaf contacts; maximum vertical
  balance error was 2.98e-8. This is finite source correspondence, not a proof
  that every experimental trajectory satisfies the recovery condition.
- Persistent planar structure: `halfCheetahReset_planar` and
  `scalar_trajectory_planar` derive x/z positions and velocities, pitch rotations,
  and y-axis angular motion from the reset and the complete scalar step, without
  a height or contact-denominator premise. `PlanarParameters` permits inertia
  principal axes in the x/z plane or along y, including the root's mixed rotation.
  `planar_flat_entry_zero` derives gathered entries 1/3/5 = 0.
  `sevenScalarStep_bound_of_planar`, `scalar_planar_barrier_reachable_bound`, and
  `scalar_planar_halfCheetah_return_bound` therefore require a height/recovery
  condition only on contact body 6, not 1/3. The extended 256-substep source test
  had maximum planar residual 3.49e-7 and zero-entry residual 2.56e-7.
- Paired foot endpoints: `footPair_normalVelocity_sum` cancels angular velocity
  exactly between the two opposite capsule endpoints. For COM height < -1/5
  and negative pre-contact vertical velocity v, `footPair_recovery_lower` gives
  recovery >= 0.87*(-v + stabilization*(-height-1/5)), including epsilon 1e-8.
  With dt=1/320, linear multiplier 1, stabilization >= 32 and height >= -1/2,
  `scalarFootBarrier_of_pair_and_speed` needs only v >= -64, for any angular
  velocity. `halfCheetahReset_pre_speed` derives initial v >= -363/8 (-45.375).
  The paired-speed radius and return theorems derive all later safe heights
  conditional on that speed bound at each physics substep. The added 108 source
  formula cases include 36 deep/downward cases, active counts 0/1/2, both signs
  of the root scalar entry, and angular velocities -2000/0/2000. Pair-velocity
  sum error was at most 5.96e-5; all recovery and height inequalities passed.
  These constructed formula cases are not a reachable-trajectory certificate.
- Exact foot joint: `foot_joint_force_world` and `foot_pre_velocity_identity`
  derive the leaf force and pre-contact speed directly from the source model.
  Damping mixes foot/parent COM speeds with weight in [0,1/8]; the remaining
  signed defect contains anchor error, angular lever motion and gravity.
  `foot_pre_speed_iff_joint_slack` gives an exact state-only speed certificate;
  reset satisfies it, and the joint-slack radius/return theorems consume it.
  `foot_post_velocity_deep_safe` derives post-contact speed >= -8.32 when a deep
  downward contact starts with speed >= -64. Four bounded-uniform resets and
  zero/positive/negative/random commands, held for 16 substeps each, passed
  64,000 source substeps. Minimum pre-speed was -8.09563; minimum signed slack
  was 55.90437; maximum speed-formula error was 4.29e-6. The local sufficient
  certificate with signed anchor error <= 1 cm covered 62,673 of these substeps,
  not all: maximum positive error was 4.34 cm. It is not a persistent invariant.
- Forced foot/parent recurrence: `foot_relative_speed_recurrence`,
  `foot_error_second_order`, and `foot_pair_momentum_recurrence` derive the exact
  relative-speed, second-order anchor-error, and common-momentum updates.
  `footRotationRemainder_trajectory_bound` bounds the geometric input by 22/15
  at every substep from unit initial rotations and fixed offset bounds, without
  later height or speed premises. `footPair_source_stable` proves that both
  homogeneous roots lie strictly inside the unit disk for dt=1/320, inverse-mass
  sum in (0,8], joint stiffness in (0,20000], and damping in (0,10]. The extended
  64,000-substep source comparison had maximum second-order residual 1.89e-7;
  the homogeneous root modulus for the checked parameters was 0.968739.
- Whole-tree momentum: `seven_joint_vertical_sum_zero` cancels every internal
  translational joint force on the actual tree. `seven_momentum_trajectory`
  derives total vertical momentum from initial momentum, gravity, and accumulated
  contact support, with no height or joint-force-bound premise.
  `halfCheetahReset_momentum_lower` derives initial mean vertical speed >= -1
  from the already-proved reset velocity and COM-offset bounds.
  `seven_foot_momentum_decomposition` recovers foot speed from total momentum
  and six explicit edge-speed coefficients. `seven_foot_contact_supported_lower`
  bounds foot speed using their absolute sum; `seven_joint_slack_of_contact_budget`
  connects the resulting contact-aware budget to the existing speed slack.
  The extended 256-substep source check had maximum momentum-update error
  6.80e-7 and foot-decomposition error 3.55e-15.
- Coupled six-edge translation: `seven_edge_forced_speed_step` and
  `seven_edge_second_order` derive the complete forced update, including the
  shared-root coupling of both legs. `sevenEdgeMassMatrix_posDef` and
  `sevenEdge_modal_root_inside` prove positive definiteness, eigenvalues in
  (0,8], and stability of all six homogeneous modes under the stated parameter
  bounds. `sevenEdgeGeometry_trajectory_bound` gives an all-substep geometry
  bound of 9/10. `sevenEdgeBias_angular` identifies the angular lever input;
  `sevenEdgeBias_bound` bounds it by (9/10) times the maximum pitch speed.
  `seven_contact_edge_budget` controls the relative contact input by eight
  times the same total support used in whole-tree momentum. The 256-substep
  source regression had second-order error 9.32e-8 and angular-formula error
  9.54e-7; all geometry, angular-input and contact-input inequalities passed.
- Forced anchor response: `pitch_increment_vertical_bound` derives the rotation
  increment from the normalized source quaternion update. Its geometry and
  angular-bias bounds use actual pitch speeds, not an assumed uniform bound.
  `seven_transient_source_step` and `seven_transient_trajectory` derive the
  complete 12-coordinate anchor-error/scaled-speed update and finite convolution.
  `seven_transient_input_bound` bounds input by actual current/next pitch budgets
  and contact support, eliminating the constant stiffness-times-geometry term.
  `seven_pre_speed_transient_bound` gives the source's pre-contact edge-speed
  bound with the actual projected matrix powers. The extended 256-substep check
  had response error 2.43e-7, rotation-identity error 9.80e-8, and pre-speed
  observation error 5.41e-6; all new input and response inequalities passed.
- Signed foot response: `seven_signed_foot_response` projects the forced state
  onto the actual pre-contact foot velocity, combining common momentum and
  contact-induced relative motion in one signed kernel.
  `seven_contact_support_lower` derives its one-sided bound from nonnegative
  contact increments, allowing negative kernel coefficients.
  `seven_angular_response_bound` retains edge-specific angular envelopes;
  `seven_directional_foot_lower` and `seven_joint_slack_of_signed_response`
  connect the response to the existing speed certificate. The 256-substep
  source check had maximum signed-response error 2.42e-5. Bounds using absolute
  angular envelopes reached -118.08 with exact contact inputs and -127.28 with
  total support alone; both covered 202/256 of the -64 speed conditions.
  Keeping the actual signed angular response and bounding only contact support
  gave minimum -15.29 and covered 256/256 conditions on these finite prefixes.
- Source work and energy: `seven_cache_velocity_shift` derives the difference
  between cached and physical anchor velocities. `seven_joint_work_balance`
  combines all body wrenches with linear elastic work and actual angular
  motor/damping/limit work, retaining the positive frame-shift term.
  `matrixInertia_pitch_constant` derives state-independent pitch inertia on unit
  planar frames; `sevenPitchInverseInertia_ge` derives its lower bound from
  positive principal inverse inertias. `seven_scalar_energy_balance` and
  `seven_mechanical_energy_balance` connect the scalar source step to kinetic
  and elastic energy, including Euler, damping, contact and anchor-increment
  corrections. The four-reset, 256-substep source regression had maximum
  mechanical-balance error 2.73e-5 and angular-work error 1.59e-3. It writes no
  rollout or checkpoint artifacts.
- Contact and anchor energy bounds: `scalar_contact_energy_identity` retains
  the scalar resolver's actual impulses and active-count epsilon. Its quadratic
  averaging defect is nonnegative, giving `seven_contact_energy_upper` without
  assuming contact passivity. `pitch_rotation_remainder_square` and
  `scalar_lever_taylor_bound` derive the exact normalized-rotation remainder
  and its bound dt^2 * pitch_speed^2 * lever_radius / 2.
  `seven_anchor_taylor_defect` connects it to the actual velocity kick;
  `seven_elastic_correction_upper` and `seven_mechanical_energy_upper` combine
  the source-derived bounds. The source test covers 256 normal-reset substeps
  plus three constructed double-contact resolver cases. Maximum contact-energy
  identity error is 2.48e-5 and anchor Taylor identity error is 1.57e-7; all
  upper inequalities pass. It writes no rollout data or checkpoints.
- Current-state recursion: `seven_current_energy_step` removes next-state
  pitch speeds and anchor increments using current acceleration/contact budgets;
  `seven_current_energy_prefix` sums this bound, deriving planar and unit-frame
  preservation from initial structure and fixed unit link rotations.
  `seven_pre_kinetic_identity` retains acceleration work and the Euler correction.
  `foot_pre_speed_of_current_energy_bound` converts an energy envelope into
  pre-contact foot speed >= -64 for every current motor command; the sufficient
  budget is 2 * inverse_foot_mass * pre_contact_kinetic_upper <= 4096.
  `scalarFootBarrier_of_kinetic_budget` connects it to paired-contact recovery.
  The 256-substep source check passes all current-state and prefix inequalities.
  The cumulative energy envelope covers 256/256 speed conditions, with minimum
  squared-speed budget slack 295.72; its maximum energy overestimate is 1708.98.
- Contracting relative energy: `seven_modal_source_step` projects the actual
  six-edge update, with dt=1/320, stiffness=20000 and linear joint damping=10.
  `spring_modal_homogeneous_identity` gives exact contraction by 1-gamma for
  L=c*error^2+(gamma-c)*error*scaled_speed+(1-gamma)*scaled_speed^2.
  Coercivity bounds L between one quarter and twice its diagonal energy.
  `seven_modal_lyapunov_prefix` derives a geometric input-history envelope;
  `spring_modal_uniform_bound` gives an invariant bound if input energy is bounded.
  `seven_transient_current_input_bound` removes successor pitch speeds from the
  input budget using current source acceleration/contact increments.
  All 1536 modal updates over 256 source substeps pass; maximum projected-update
  error is 7.53e-8 and homogeneous-energy identity error is 1.63e-19.
- Directional output: `spring_modal_norm_prefix` derives the square-root energy
  response without the energy envelope's 1+2/gamma input multiplier.
  `spring_modal_output_bound` gives the modal velocity-direction gain.
  `seven_modal_foot_identity` reconstructs the actual pre-contact foot output;
  `seven_modal_center_trajectory` retains accumulated contact support and signed
  current angular bias. `seven_modal_history_foot_lower` connects its norm-history
  budget to all current commands, and `scalarFootBarrier_of_modal_budget` supplies
  the paired-contact recovery certificate. The four-reset source check passes
  all 256 output intervals: maximum identity error 2.16e-6, center-history error
  2.57e-7. The norm-history lower bound covers 256/256 speed conditions (minimum
  -57.24), versus 189/256 for the energy envelope (minimum -120.70).
- Restarted response: `spring_modal_norm_envelope_rebase` and
  `seven_modal_restart_budget_le` prove that restarting at an actual source state
  cannot increase the budget. `seven_modal_restart_foot_lower` supplies the
  all-current-command interval, with a current-state certificate at zero length.
  `scalar_restarted_modal_reachable_bound` and `scalar_restarted_modal_return_bound`
  derive height, radius and return bounds from restart margins, without future
  height premises. Four 1,000-action source rollouts (reset keys 90310--90313;
  zero/+1/-1/uniform random commands, NumPy seed 90310, each held 16 substeps)
  cover 64,000 substeps. Current-state and action-block certificates cover all,
  with minimum lower endpoints -11.83/-44.95. The original history certificate
  covers 47,580/64,000 (minimum -171.52); actual speed stays >= -7.25. Maximum
  output identity error is 3.98e-6. The regression is
  `jax_experiments.tests.test_v32_spring_modal_horizon` and writes stdout only.
- Signed local drift: `seven_modal_signed_gain_identity` retains the input/state
  polar term; `seven_modal_norm_gain_identity` derives its exact square-root
  increment, including the zero-energy case. `seven_modal_local_drift_identity`
  and `seven_modal_local_margin_trajectory` combine common momentum, signed bias
  changes and modal norm gains without a future safety premise.
  `seven_modal_local_lower_reset` proves local lower >= -30 for every normal
  reset under the explicit source geometry, mass and velocity bounds. No initial
  modal-margin hypothesis is needed. The four long source rollouts check 383,976
  modal updates; maximum energy/norm/drift identity errors are 1.05e-9/9.14e-8/
  1.16e-5, within the retained tolerances. The same test module's
  `test_modal_corner_commands` checks all 64 constant commands in {-1,+1}^6 at
  reset key 90310 (1,024,000 substeps). Current/block/actual floor failures are
  all zero; minimum current/block lower endpoints are -10.79/-38.93.
- Angular primitive: `seven_bias_primitive_step` proves the exact source-cache
  bias equals a bounded geometric difference minus its signed right-endpoint
  correction. `seven_weighted_bias_primitive_identity` and
  `seven_angular_input_prefix_primitive_bound` integrate the actual angular input
  with response rows, without a future height or uniform angular-speed premise.
  The remaining correction budget uses squared post-step pitch speeds.
  `jax_experiments.tests.test_v32_spring_angular_primitive` checks 64,000 source
  substeps plus 16 weighted-bias and 16 full-input windows; maximum identity
  errors are 2.20e-7, 8.69e-7 and 9.22e-7. It writes JSON stdout only.
- Pitch response: `seven_pitch_torque_split` retains the root angular damping,
  motor/limit torques, and the actual linear-force lever torques.
  `seven_pitch_mix_bound` proves non-expansion under the source step-size condition;
  `seven_pitch_source_cfl` derives this condition from fixed parameter ranges.
  `seven_pitch_source_trajectory` gives the exact signed forced response.
  `seven_pitch_trajectory_envelope` and `seven_bias_correction_source_envelope`
  replace successor pitch speeds by the initial state and explicit source inputs.
  `seven_pitch_reset_max` initializes the pitch envelope at <= 0.4.
  `jax_experiments.tests.test_v32_spring_pitch_damping` checks 64,000
  double-precision current-source substeps; maximum torque/step/trajectory
  errors are 4.47e-12/3.29e-14/1.59e-13. Maximum damping step is 0.0884.
  It retains all source forces and writes only JSON stdout.

Current inspections of the frozen interpreter paths found Brax 0.10.5 / JAX
0.4.38 on the three GPU servers and Brax 0.12.1 / JAX 0.5.3 in the shared HPC
environment, inspected through its login node. Every inspected resolver retains
the flattened gather. The compact source diagnostic in
`source_audit/contact_rollout_88003_197201.json` executes 1,000 actions per branch
on local Brax 0.14.2 / JAX 0.9.2. Original true-mode and uncompensated branches
have 1,046 and 341 penetrating-contact normal denominators below inverse mass,
respectively, but no nonpositive denominators or nonfinite substeps.
A private matrix-gather resolver removes these lower-bound violations; returns
change from 4213.7 to 4512.0 and from 1051.9 to 729.4. Same-state scalar and matrix
formula comparisons have zero maximum error. Installed modules and frozen
result artifacts are unchanged.

Compact-audit recomputation reproduces all seven switching/stationary means,
paired confidence intervals and win counts, 98.55% accuracy, and median delay 3.
All ten reference manifests record 8.4M interactions and 525,000 updates.
Lean builds; no proof placeholders are present. The revised manuscript compiles
without unresolved references or overfull boxes.

## Limitations

The belief-risk theorem treats the true conditional mass and the filter belief
as separate distributions. A learned emission or tempered update does not prove
their total-variation discrepancy is small; aggregate mode accuracy and Brier
score are not a uniform conditional calibration certificate. The action-value
corollary and closed-loop extension require a common sensitivity coefficient
across the compared modes at the same conditioning state. The latter additionally
uses an actual conditional-mode kernel, a measurable causal forward flow,
occupancy-integrable value differences, and integrable accumulated budgets.
Bellman integrability at all states and counterfactual signals is still inherited
from the measure-level return theorem. The conditional kernel is supplied as
part of the model; no automatic disintegration or learned-posterior recovery is
claimed. These are mechanism bounds, not numerical superiority theorems over
SAC, ESCP, or RE-SAC.

The continuous KL result still assumes conditional absolute continuity, but no
longer requires finite divergence. The general return theorems assume
Bellman-integrability at all states and remaining horizons. The remaining
reachability obligation is preserving a sufficient tangential safe region on
reachable states. The complete scalar envelope and radius induction are now
derived conditional on that region.
The normal margin is derived under the explicit fixed-parameter conditions;
the tangential margin is derived on penetrating contacts with COM height >= -0.5.
Reset initialization and the first physics substep satisfy the height floor.
Under the proved planar structure and actual contact layout, only body 6 retains
that height obligation; bodies 1/3 no longer require a height-floor invariant.
Its paired endpoints now reduce the coupled barrier condition to a pre-contact
COM vertical-velocity bound >= -64, without an angular-velocity bound on contact
recovery. The exact joint update now expresses this bound as a signed state-only
slack condition and proves it at reset. Preserving that slack for subsequent
reachable states remains open; joint error, parent speed and angular lever motion
must be controlled together. The geometric rotation input is now bounded for all
substeps, and whole-tree common momentum is derived without an upstream-force
envelope. A sufficient relative edge-speed budget is connected to accumulated
contact support. Establishing that budget for all reachable states remains open:
the coupled six-edge update is now derived, its geometry input is bounded,
and every homogeneous mode is stable. The remaining angular input is bounded
in terms of pitch speed, not by a proved trajectory-wide constant. Relative
contact input is controlled by the same nonnegative support as common momentum,
and its filtered effect is now represented by exact matrix powers. A finite
forced-response bound is derived around joint anchors, with local rotation
increments controlled by the actual normalized integration step. It still
depends on the actual angular-motion/contact history, not a trajectory-wide
input envelope. The conservative pre-contact edge-speed L1 bound reaches
7040.9 in the checked prefix, too loose for the -64 foot-speed certificate.
The fixed-parameter translational contact-to-prefoot impulse kernel has a
negative coefficient (-0.47285 within 64 lags), so common support cannot be
treated as monotonically improving later foot speed. This is a linear-channel
diagnostic, not a reachable-trajectory counterexample. The signed foot-response
identity and one-sided contact bound are now proved. Absolute angular envelopes
still miss 54 of the checked 256 speed conditions, whereas retaining their actual
signed response covers all 256. The remaining closure is a reachable-trajectory
bound on this signed angular contribution together with the contact support,
not a proof that every angular velocity is uniformly small. Its finite-prefix
coverage does not prove preservation for all resets/actions.
The installed joint cache uses a parent velocity at the opposite lever sign
from physical anchor velocity. Its corrected linear work has a positive
frame-shift term; the source check finds positive linear work plus elastic
first variation in 76/256 substeps, and positive contact kinetic work in two.
These observations rule out unconditional passivity for those channels, not
stability of the complete simulator. The contact averaging and second-order
anchor bounds now give a source-correct single-step mechanical-energy upper
bound. The current-state recursion now eliminates next-state pitch speeds and
anchor increments, and its prefix sum connects to an energy-based foot-speed
certificate. Its cumulative envelope covers the checked 256-substep prefix.
Proving a sufficiently tight cumulative gain bound for every supported reset
and action sequence remains necessary; the actual acceleration/contact budgets
and Euler correction still depend on the reachable state. Neither contact nor
anchor corrections can simply be discarded as damping.
The cross-term modal energy now retains spring--velocity cancellation and
contracts the coupled relative-translation response. Its geometric convolution
uses the actual rotation/contact history. A uniform bound on those input
energies, and their coupled effect on common momentum and foot speed, remains
to be derived on supported reachable trajectories.
The directional norm certificate now connects that response to foot speed,
preserving signed common momentum and current angular bias instead of bounding
all motion by total kinetic energy. Its lower endpoint covers the checked prefix.
The longer supported-command regression shows that the full-history certificate
does not preserve center minus radius >= -64 in the frozen scalar source execution,
even while actual speeds satisfy the floor. It is too conservative as an invariant.
Restarting at each 16-substep action block removes that accumulated looseness in
all 64,000 checked substeps and is formally no worse than the history envelope.
The current-state certificate now has a proved initial lower bound -30 and an
exact signed drift identity. Uniform preservation still requires a cumulative
drift lower bound of -34 (or a sharper sufficient bound) at every prefix for all
supported resets/action histories. The 64 constant-corner profiles cover one
reset, not arbitrary policies; their finite coverage does not discharge that
remaining invariant premise or universal source correspondence.
The bounded angular primitive and its endpoint correction are now source-derived.
The independent geometry/row-variation bound is not a tighter safety certificate:
on the random rollout its 16,000-step weighted-bias bound is 449.95 versus 17.67
for the actual pitch envelope. Its 15,999-step full-input bound is 9030.31 versus
observed magnitude 9.21. Further closure requires correlated rotation/contact
control or preserved cross-channel cancellation, not only bounded lever geometry.
The source angular damping now has a proved substochastic update, and its full
forced response includes motor/limit, shifted linear-force, and contact torques.
The pitch/correction envelopes need no independent future-speed premise, but
their source inputs remain coupled to current motion and contact. Closing the
signed cumulative input budget remains necessary for the -64 invariant.
The random-command absolute pitch envelope reaches 21607.85 versus actual
maximum pitch speed 32.11 in the float64 diagnostic. It is not a tight invariant.
The earlier float32 batch/individual-joint factorization differed by about
1.20e-3 in torque. The higher-precision check validates the real-arithmetic
formulas, not frozen floating-point execution or a historical replay.
The observed 4.34 cm positive anchor error rules out
a uniform 1 cm error invariant for the tested supported commands. The 64,000
finite source comparisons do not prove slack preservation for all resets/actions.
One environment action contains sixteen physics substeps.
The installed `brax/spring/collisions.py` calls `i_inv.take(link_idx)` without
`axis=0`: its seven-link inertia array is flattened before scalar gathering.
A constructed unit-rotation maximal state, with affected COM height -3.8873,
reproduces a negative tangential denominator and friction that accelerates along
the velocity outside the certified height region. This state has
not been proved reachable from the experimental reset. The full-matrix PSD
theorem cannot certify this source branch. No installed library or frozen
experimental code was changed. Unit rotations in COM, world,
anchor and joint frames are now preserved by the composed real-arithmetic
definitions. Universal equivalence with the Python/JAX pipeline remains open;
the finite comparisons do not prove it. The generic reachability theorem still
takes a complete one-step envelope as a premise. The matrix-model development
derives its own envelope from bounded fixed parameters, positive inverse masses,
nonnegative inertia weights and initially bounded COM states with unit rotations.
It does not establish the frozen scalar implementation's tangential invariant
or certify floating-point execution. Initial states and action processes must
be measurable; trajectory, observation, displacement, energy and return
measurability are now derived for the matrix-contact model.
The runtime inspections are not immutable historical dependency snapshots.
The source diagnostic is not an exact replay: the original first-episode returns
for these identifiers were 4409.4 and 1086.8. The fused, current-runtime paired
comparison is a single-case sensitivity test, not a corrected-solver confirmation
against the learned baselines.
The Gaussian sensitivity theorem still needs a measurable bounded continuation
integrand; an envelope would supply boundedness, not measurability by itself.
No additional smoothness premise is needed. No numerical value of B
or experimental return guarantee is claimed. Lean does not verify the complete
Brax implementation, floating-point code, or the statistical superiority claim.
The earlier focused Python run had 13 passes and two historical registration failures:
V12's recorded source versions of `train.py` and `stochastic_mode_env.py` differ
from the current worktree. Historical registrations were not rewritten; compact
result reproduction is distinct from replaying that frozen training source.
The new reward-contract test was invoked directly with the `bapr-jax` Python
because that environment has no pytest installation; its assertions all passed.
