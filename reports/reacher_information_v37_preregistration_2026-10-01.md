# Reacher Same-Information Comparison

Freeze before launch. V32 and the five confirmation seeds remain untouched.
This is a three-seed development experiment, not new confirmatory evidence.

- New policy seeds: 97007/97027/97047. Switching events: 208601/208617/208633,
  five paired episodes each; stationary event base: 208701, five per mode.
- Four separately trained arms per seed: native-reference SAC, memoryless
  domain-randomized SAC, filtered-belief conditioned SAC, instantaneous-belief
  conditioned SAC. Twelve GPU trainings, one smoke, three CPU audits, one aggregate.
- Every controller: 400 iterations, 1.6M interactions, 100,000 updates;
  hidden width 256, native reward, 200-step horizon, 50-step dwell, noise std 0.02.
  Native generalized Reacher, admitted CPU sampling plus GPU updates;
  JAX 0.4.38/Brax 0.10.5/Flax 0.10.4/MuJoCo 3.1.6.
- Freeze inverse estimator 95103 and controls (0.002, 1.0, 0.98). No refitting.
  The actor and critic of conditioned SAC receive (state, pre-action belief).
  Belief updates only after the completed transition and resets at episode boundaries.
  Conditioned SAC outputs the command directly; it receives no external compensation.
- Primary: filtered direct compensation minus filtered conditioned SAC.
  Secondary predefined control: instantaneous direct compensation minus instantaneous
  conditioned SAC. Keep oracle/native, robust, fixed 0-3, and head-0 controls.
  Switching and stationary results are reported separately. Paired confidence
  intervals use three policy-seed means, not individual episodes. Do not choose
  the better filter retrospectively or call this an ESCP reproduction.
- Both same-information methods use the same historical provider (1.6M interactions,
  100,000 updates), calibration (260,000 transitions), and 4500 estimator updates.
  These costs are shared and disclosed; memoryless SAC has no estimator cost.
  Report conditioned actors' extra input parameters instead of claiming equal capacity.
- Smoke checks both causal learner paths and measures full 4000-step inference
  rollouts. Stop before full training if steady sampling alone exceeds 120 seconds
  per iteration. Resume restores full controller/replay/optimizer/RNG checkpoints.
  Train on matching GPU servers, never local; CPU nodes evaluate compact actors.
  Only small actor/estimator exports and JSON audits synchronize, not full checkpoints.

Next: analyze the matched comparisons without retuning on these outcomes. A positive
primary paired interval motivates a new independent confirmation cohort; otherwise
retain an inconclusive/negative comparison and the existing robust-SAC evidence.
