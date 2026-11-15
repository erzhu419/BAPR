# Prospective HalfCheetah audit package

This package contains the frozen V32 evidence used in `paper/paper.tex`:

- `analysis.json`: ten-policy-seed aggregate and preregistered decision.
- `audits/seed_*/audit.json`: per-seed stationary, switching, and causal diagnostics.
- `estimator/`: the frozen five-head inverse model and its original manifest.
- `reference_policies/seed_*/mode_0/`: final canonical actors and provenance.
- `training_metadata.json`: final budgets, configurations, interpreter paths and hosts for the
  thirty SAC/recurrent-ESCP/RE-SAC-0 comparator runs.

The last comparator is **RE-SAC-0**, a five-critic RE-SAC ablation with both
`weight_reg=0` and `beta_ood=0`. It is not evidence against regularized RE-SAC.
Its statistical record and original preregistered gate remain unchanged.
`python scripts/check_manuscript_results.py` recomputes the released episode
means, paired intervals and win counts for HalfCheetah and Reacher, and checks
their recorded training budgets without JAX or model checkpoints. A separate
[`../halfcheetah_runtime_comparison/`](../halfcheetah_runtime_comparison/)
package contains the later five-seed runtime-matched comparison; it is not
pooled with this ten-seed confirmation.

From the repository root, `python paper/plot_v32_figures.py` regenerates the figure source CSVs and manuscript figures directly from this package. The compact audits suffice to recalculate the reported summaries and plots. Rerunning baseline environment rollouts requires the separately trained SAC, ESCP, and RE-SAC-0 policy bundles; reproducing training additionally requires the original interaction budgets. Neither replay buffers nor full training checkpoints are included here.

The confirmation uses Brax HalfCheetah with the spring backend, not Gym MuJoCo-v2. The policy seeds and event schedules are fixed in `jax_experiments/analysis/regime_polarity_action_compensation_power_confirmation_v32.py`; estimator fitting and its additional supervision cost are specified in `paper/appendix_submission.tex`.

The frozen signatures retain the actual interpreter paths. A 2026-09-28 inspection
found Brax 0.10.5 / JAX 0.4.38 in the GPU virtual environment on jtl311linux,
jtl110gpu and jtl110gpu2; the shared HPC interpreter used by node007 and the CPU
node profiles contains Brax 0.12.1 / JAX 0.5.3. Local `resac-jax` contains Brax
0.14.2 / JAX 0.9.2, so its version cannot describe the entire confirmation.
These are current inspections of the recorded paths, not historical dependency
snapshots. All inspected resolver copies use the flattened inertia gather.
The primary confirmation also spans both profiles: GPU/shared-HPC counts are
9/1 for canonical policies, 4/6 for SAC, 6/4 for recurrent ESCP and 5/5 for
RE-SAC-0. Historical simulator equivalence cannot be established from these
interpreter paths, so a runtime confound remains in the HalfCheetah comparisons.
`source_audit/contact_rollout_88003_197201.json` contains a separate one-actor
contact diagnostic and isolated matrix-gather sensitivity comparison; it does
not replace any frozen audit. The runner is
`jax_experiments/analysis/audit_v32_spring_contact_rollout.py`.

The theorem-to-code correspondence, Lean build target, and formalization limits
are recorded in [PROOF_SCOPE.md](PROOF_SCOPE.md).

## Mechanism Theory

The submission manuscript retains three core results: pathwise compensation
equivalence, action-weighted mode error, and a conditional finite-horizon return
bound. `paper/appendix_submission.tex` contains their proofs and the unchanged
experimental protocol. `paper/appendix_v32.tex` preserves the extended theory
and simulator-source notes but is not included in `paper/paper.tex`. The fourteen
general Lean modules are shipped verbatim in the independent
[`../lean/`](../lean/) package with pinned dependencies and a default build
covering every module. The package README maps manuscript statements to theorem
names and states the sufficient assumptions used in the formalization.

The fourteen general mechanism targets are independent of HalfCheetah and spring
certificates. `ActionCompensationBelief` links the finite conditional mode law
to squared nominal and executed-action error, with an explicit total-variation
calibration term and a mode-uniform action-value sensitivity bound. It preserves
the frozen MAP selector and tempered filter; no algorithm or experiment changes
are introduced. The simulator-source certificates remain outside the manuscript.
`ActionCompensationClosedLoop` accumulates the calibrated risk along the causal
occupancy sequence. It derives transport and the performance-difference sum
from the forward update law, and keeps the true conditional mode kernel separate
from the frozen filter. Integrability, sensitivity and calibration remain
explicit conditions; no new training or result claims are introduced.
`ActionCompensationFilter` proves geometric propagation of emission-score errors
through the frozen tempered sticky update, and a conditional finite-dwell MAP
recovery certificate that allows a stale incoming belief. Its TV/action-risk
connection keeps reference-filter bias separate. Persistent score separation
and learned-score accuracy are hypotheses, not consequences of Gaussian noise
or the reported estimator metrics.
`ActionCompensationEvidence` derives the emission-score error budget from
inverse-prediction errors under the implemented log-mixture score. It also
derives an error-dependent actuator-excitation margin and connects it to causal
mode recovery. Prediction accuracy along causal rollouts remains a hypothesis;
training loss is not a uniform-error or probabilistic recovery certificate.
`ActionCompensationRecovery` derives a finite-window MAP failure probability
from causal-rollout prediction-error moments and excitation failure. Correlated
times, heads and coordinates are allowed. It proves filter/event measurability,
bounded-output integrability and the conditional-moment tower implication.
The moment and excitation bounds are population assumptions, not training-MSE
certificates; the bound may be vacuous and does not prove posterior calibration.
`ActionCompensationReturn` converts actual mode-error probabilities into nominal
and executed action MSE, and derives an oracle-relative return bound on the
causal occupancy flow. It charges possible post-window mode changes separately.
No posterior calibration is required; integrability, uniform sensitivity and
per-decision error bounds on the same law remain conditions.
`ActionCompensationWindows` connects random observed dwell windows to that
same decision law. It conditions the actual path distribution, retains the
incoming filter belief, averages positive-mass window certificates, and charges
unobserved next-mode changes. Conditional moment/excitation bounds remain inputs;
neither fixed-window training moments nor zero-delay oracle metrics supply them.
`ActionCompensationCumulative` supplies a cumulative-excitation alternative to
the simultaneous error tube. It retains the sticky nonlinear margin recursion
and derives a discounted mean deficit from inverse-error second moments.
It permits intermediate weak excitation, but may lose its sufficient certificate
after a quiet final step; moments and excitation remain same-law conditions.
`ActionCompensationRetention` supplements that certificate with a positive
sticky log-odds recursion. Flat scores retain an already strict MAP on every
finite quiet prefix; adverse contrasts consume the explicitly retained margin.
Equal inverse candidates imply flat scores even with noisy predictions.
Retention concerns the actual incoming belief and may preserve a stale mode
after an unobserved switch; the proof is in real arithmetic.

## Auxiliary Source Certificates

The separate `SpringMatrixStep` Lean target derives the full matrix-contact
model's finite-horizon growth and return bounds. `SpringMatrixMeasurable`
derives trajectory/observation/return measurability and return integrability
from measurable initial states and clipped action processes. These targets
do not change the frozen training or evaluation physics.
`SpringScalarMargins` derives a normal-denominator margin for the scalar model
and a tangential margin on an explicit COM-height region; preservation of that
region by the experimental dynamics remains unproved.
`SpringScalarStep` connects those margins to the original scalar-contact
step and finite-prefix return bound, and proves a position-velocity height
criterion that can be checked once to cover every current motor command.
`SpringReset` derives initial radius 3, initial COM heights >= -1/3, and
height-floor preservation for the first physics substep from the wrapper's
uniform reset and explicit source geometry/parameter bounds. It does not
establish preservation across all sixteen physics substeps of an action.
`SpringContactBarrier` supplies a multi-contact vertical recovery lower bound
and a finite-horizon safety/radius/return certificate under an explicit
current-state barrier condition. It retains both endpoint candidates and
active-count averaging. The region is required only on non-diagonal bodies
owning actual capsule contacts, not on bodies with no contact candidates.
`SpringPlanarInvariant` proves reset and complete-step preservation of planar
motion in real arithmetic. It derives gathered entries 1/3/5 = 0 and connects
the radius and return bounds to a recovery condition on contact body 6 only.
`SpringPairedFoot` cancels angular point velocities across the opposite foot
endpoints and derives height propagation from pre-contact COM vertical velocity
>= -64. Reset gives >= -45.375; preserving the speed bound at later substeps
remains open. Radius and return bounds are conditional on this speed invariant.
`SpringFootJoint` derives the exact leaf-joint speed update, damping convexity,
the equivalent signed state-only speed slack, and deep-contact recovery to
>= -8.32. Reset satisfies the slack; finite-horizon bounds require its continued
preservation. The source regression covers four 1,000-action bounded-command
rollouts. Its observed anchor errors rule out a uniform 1 cm error invariant.
`SpringFootRecurrence` derives the forced relative-motion and common-momentum
updates, bounds the geometric rotation remainder at every substep, and proves
homogeneous joint stability under explicit source-parameter ranges. Closing the
speed slack still requires bounds on the remaining forcing and common motion.
`SpringGlobalMomentum` cancels the internal joint forces from total vertical
momentum and derives its complete contact-supported trajectory. It expresses
foot speed using six relative edge speeds and connects their budget to the
speed slack. Proving that budget on all reachable states remains open.
`SpringCoupledEdges` derives the complete forced six-edge translation update
and proves stability of all its homogeneous modes. Geometry is bounded at every
substep; angular and contact inputs reduce to pitch speed and total contact
support. Their combined forced effect on the reachable speed budget remains open.
`SpringForcedResponse` removes the fixed geometry offset by using joint anchor
error, derives local rotation increments and the finite convolution response,
and connects it to pre-contact speed. Its conservative input-history bound is
not tight enough to prove the -64 invariant; angular/contact feedback remains open.
`SpringSignedResponse` derives the signed foot-speed convolution and a one-sided
contact-support bound. Edge-specific absolute angular bounds still miss 54/256
source substeps; retaining the actual signed angular response covers this prefix.
A trajectory-wide signed angular/contact bound remains necessary for preservation.
`SpringJointWork` derives the cached/physical anchor-velocity shift, seven-link
joint work, constant planar pitch inertia, and exact scalar-step kinetic and
elastic energy balances. The balances retain Euler, damping, contact and anchor
corrections; the 256-substep source check reproduces them.
`SpringEnergyBounds` proves the nonnegative contact-averaging defect, the
second-order normalized-rotation remainder, the actual anchor-update bound,
and their combined single-step mechanical-energy upper bound. Source checks
also cover three constructed double-contact cases.
`SpringCurrentEnergy` removes next-state quantities, sums the resulting energy
recursion, and converts the cumulative envelope to a sufficient foot-speed
condition for all current motor commands. All 256 checked source substeps meet
this condition, with minimum squared-speed budget slack 295.72. A uniform bound
over every supported reset/action sequence remains open; source budgets still
depend on the reachable state.
`SpringTransientLyapunov` projects the actual six-edge source update and proves
coercive cross-term energy, exact homogeneous contraction, geometric forced
response, and a bounded-input invariant estimate. Its current-state input
budget uses source acceleration/contact increments. The 1536 modal source
checks pass; global reachable angular/contact input bounds remain open.
`SpringModalFoot` proves the square-root energy response, directional output gain,
the actual foot-speed interval, and its all-command recovery implication. The
norm-history certificate covers 256/256 checked substeps (minimum lower bound
-57.24), versus 189/256 for the energy envelope. Preservation across every
supported reset/action history is not implied: a longer source diagnostic finds
full-history bounds as low as -171.52 while actual speed remains >= -7.25.
The same module proves restarted budgets are no worse, connects their intervals
to scalar height/radius/return closure, and includes the current-state certificate.
`jax_experiments.tests.test_v32_spring_modal_horizon` checks four 1,000-action
rollouts without checkpoints or output files. Restarting at each 16-substep
action block covers all 64,000 substeps (minimum -44.95); uniform preservation
of the local margin for every supported reset/action history remains open.
`SpringModalDrift` retains the signed input polar term, proves exact local
certificate drift, and derives local lower >= -30 for every normal bounded
reset under the stated source conditions. The same test module checks these
identities over 383,976 modal updates; its `test_modal_corner_commands` covers
all 64 constant corner commands from reset key 90310 (1,024,000 substeps), with
no current/block/actual speed-floor violations. The remaining invariant requires
a cumulative drift lower bound across arbitrary supported action histories;
constant-command coverage does not prove it.
`SpringAngularPrimitive` proves the bounded primitive of the actual cache bias,
the exact normalized-rotation endpoint correction, and their weighted response
and complete angular-input bounds. The corresponding source regression checks
64,000 substeps (maximum endpoint identity error 2.20e-7). Its independent
absolute bound remains too loose to close the -64 invariant; rotation/contact
correlation and the quadratic correction budget still need control.
`SpringPitchDamping` derives the complete pitch response with a non-expanding
angular-damping matrix, retaining motor/limit, linear-force shift and contact
torques. It initializes the pitch envelope from reset and bounds the quadratic
rotation correction through explicit source inputs, without an assumed future
speed bound. Their coupled cumulative effect remains open within this auxiliary
simulator certificate, rather than a premise of the core compensation mechanism.
Its float64 current-source diagnostic checks 64,000 substeps (maximum pitch-step
error 3.29e-14). The absolute pitch envelope remains too loose; this higher-
precision formula check does not certify frozen float32 execution.
