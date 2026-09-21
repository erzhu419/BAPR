# V30 Ant finite-horizon branch-risk preregistration

V29 proved exact action-coordinate compensation and substantial Ant return
headroom, but failed its absolute zero-termination gate. Following
`markdown/GPT_diagnosis.md`, V30 does not tune another risk penalty. It first
tests whether a complete robust-policy continuation can actually rescue the
same physical states in which the compensated candidate terminates.
The synchronized V25/V26 compact bundles contain final policies but not their
risk-critic parameters, so V30 tests this necessary fallback-headroom premise
without pulling obsolete checkpoints or claiming to recalibrate those critics.

The three frozen V22 policy seeds and each seed's V29 calibration-selected
reference mode are reused without reselection. Three new source event streams
generate canonical reference-policy states. Snapshots are selected at fixed
steps and at preregistered offsets before a physical termination. Each snapshot
is branched under all four actuator modes with eight paired future-noise draws.
One arm executes true-mode action compensation for the full branch; the other
executes robust SAC for the full branch. Physical termination stops a branch;
the artificial branch horizon is a truncation.

The audit reports termination within 1, 2, 4, 8, 16, 32, 64, 128, and 250
steps, candidate/fallback returns, rescue and harm counts, initial Ant torso
height/health margin, and exact candidate invariance across actuator modes. No
simulator state, trajectory array, checkpoint, estimator, or risk model is
saved or synchronized.

At 250 steps, later finite-horizon risk-model training is authorized only when
at least two seeds contain at least four unique candidate failures, pooled and
every informative-seed candidate risk exceed fallback risk by at least 0.02,
at least three of four actuator modes improve, at least 50% of candidate
failures are rescued, and no more than 10% of candidate-surviving pairs are
harmed by fallback. Failure closes this Ant shielding branch; it does not
reopen or reinterpret V29's failed estimator gate.
