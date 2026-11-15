# Causal evidence calibration on actuator polarity

## Motivation

The frozen v1 conditional forward model fails as a literal Gaussian
likelihood: mode 0 is systematically ranked below incorrect heads, yielding
`0.537` switching accuracy and control below robust. A diagnostic
leave-one-policy-seed-out affine map over its centered four-head likelihood
vector nevertheless reaches `0.957` accuracy with ten causal transitions.
This v2 screen tests that specific calibration hypothesis without changing the
environment, controller, forward model, test seeds, or deployment gates.

## Split and estimator

- Forward model: frozen v1 parameters and empirical variance.
- Calibrator-fit controller seeds: `8,16`.
- Temporal/configuration selection controller seed: `24`.
- Untouched test controller seeds: `101,211,307,419,523`.
- Fit events: `92001,92002`.
- Selection events: `93001,93002`.
- Sealed test events: `94001,94002,94003`.
- Online input: centered four-head v1 log-likelihood from the latest causal
  transition.
- Temporal state: exponential moving evidence initialized to zero at episode
  start and never reset from an evaluator switch clock.
- Output: affine softmax posterior over the four modes.

The affine map is fitted with class-balanced ridge regression on the fit split.
EMA coefficient, ridge coefficient, and posterior temperature are selected
only on seed 24. True mode labels are available only during exploratory
calibrator fitting and validation; no mode ID, gain vector, executed action,
future transition, or switch clock is available online.

This is a supervised inference diagnostic, not yet the final self-supervised
BAPR estimator. Its purpose is to determine whether the v1 likelihood
representation can support causal control once systematic model
misspecification is calibrated.

## Frozen-control audit

The five untouched seeds rerun the identical four arms and strict horizons:
`robust`, `oracle`, `learned_soft`, and `learned_map`. Physics uses persistent
250-action actuator-polarity modes and per-step Gaussian actuator noise
`std=0.02`. The action at time `t` uses only transitions before `t`.

The preregistered v1 gates remain unchanged:

- mode accuracy at least `0.85`;
- median and P90 switch delay at most `25` and `50` actions;
- Brier score at most `0.25`;
- at least 50% oracle-headroom recovery;
- at least 4/5 policy-seed wins over robust;
- termination gap no more than five percentage points.

Only simultaneous inference and frozen-control passes authorize direct
posterior-conditioned SAC training. Failure blocks policy training and
requires changing the transition representation rather than calibrator tuning
on the sealed test seeds.

## Final result

All five untouched controller seeds and all three sealed event streams
completed. The selected calibrator uses `ema_alpha=0.9`, `ridge=0.001`, and
`temperature=0.25`; it already failed its seed-24 validation gate and also
fails the sealed test:

| Arm | Switching return | Delta vs robust | Seed wins | Oracle headroom recovery |
|---|---:|---:|---:|---:|
| robust | 1023.4 | - | - | - |
| true oracle | 2327.7 | +1304.3 | 5/5 | 100% |
| calibrated soft | 746.3 | -277.2 | 1/5 | -25.8% |
| calibrated MAP | 602.8 | -420.7 | 0/5 | -38.6% |

The soft estimator reaches only `0.537` mode accuracy, `0.530` Brier score,
23-action median delay, and 250-action P90 delay. Only the median-delay gate
passes. Its stationary mean is `800.5`, below robust `1047.4`. MAP has a
higher stationary mean (`1515.5`) but a negative worst-mode return
(`-213.1`) and fails every switching policy seed.

The central failure is mode 0, not an evenly distributed classification
error. On switching traces, the soft estimator predicts mode 0 for only about
9% of true-mode-0 actions and assigns roughly half of them to mode 3. Mode 1
is recognized at about 95%. Stationary posterior accuracy shows the same
asymmetry: approximately `0.081/0.960/0.808/0.693` for modes 0-3.

The earlier leave-one-test-seed-out diagnostic was therefore optimistic. Its
affine map was evaluated on traces generated before changing the controller
with that map. When deployed causally, a wrong posterior changes actions and
the visited state distribution, and the fixed affine correction does not
transfer across policy seeds and closed-loop trajectories. This is
compounding covariate shift, not another EMA or temperature problem.

Direct posterior-conditioned policy training remains blocked. The next
positive-control screen changes the observation representation rather than
tuning this likelihood: infer the action actually executed by the plant from
`(s_t, s_{t+1})`, then compare that inferred action with each persistent
mode's transformed commanded action. Simulator `executed_action` labels may
be used only to train this inverse system-identification diagnostic; they are
never available to the online estimator or frozen-control audit. Passing
would show that policy-invariant actuator evidence is sufficient. Failure
would reject this benchmark's learned-estimator route despite its confirmed
oracle headroom.
