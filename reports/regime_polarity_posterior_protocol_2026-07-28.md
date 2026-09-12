# Causal posterior screen on actuator polarity

## Motivation

The fresh five-seed HalfCheetah confirmation establishes a large and
reproducible control upper bound: true-mode oracle switching return exceeds the
equal-budget robust controller by 123.3%, with 5/5 seed wins and a strictly
positive paired interval. The earlier context-delay audit also retains 87.9%
and 64.0% of oracle headroom at 10- and 25-action delays. This authorizes a
learned estimator, but not an end-to-end BAPR claim.

This screen isolates inference from policy optimization. The robust and
true-context controllers remain frozen. A causal probabilistic model is fitted
on old exploratory controllers, selected on a separate exploratory controller,
and tested on the five untouched confirmation controllers.

## Data split

- Environment: `HalfCheetah-v2`
- Persistent family: `actuator_polarity`
- Dwell: fixed 250 actions
- Estimator-training controller seeds: `8,16`
- Filter-selection controller seed: `24`
- Untouched test controller seeds: `101,211,307,419,523`
- Training event seeds: `92001,92002`
- Validation event seeds: `93001,93002`
- Sealed test event seeds: `94001,94002,94003`
- Behavior data: frozen robust and true-context oracle controllers
- Estimator inputs: `(state, commanded action, reward, next state)`
- Forbidden inputs: mode ID, action-gain vector, executed action, future
  transition, or evaluator switch clock

The mode label is used only to fit the four conditional transition models. It
is never available to the online posterior. The policy context at action
`t` uses transitions strictly before `t`.

## Estimator

The estimator uses an ensemble of four mode-conditioned forward dynamics and
reward models. Conditional means are trained only by the true-mode generative
prediction loss. Classification loss is diagnostic and does not update the
model, preventing labels from manufacturing likelihood separation. Mode-wise
aleatoric variance is calibrated from empirical residual second moments;
ensemble mean disagreement is reported separately as epistemic uncertainty.

A sticky HMM accumulates conditional transition likelihood across time.
Hazard, evidence scale, and posterior decay are selected only on seed 24.
There is no BOCD, critic Q-variance, CUSUM threshold, LCB, confidence gate,
residual bank, or hard expert switch.

## Frozen-control arms

Every test event evaluates four paired arms:

1. `robust`: robust checkpoint with its sealed zero context.
2. `oracle`: conditioned checkpoint with the physical one-hot mode.
3. `learned_soft`: conditioned checkpoint with the causal four-way posterior.
4. `learned_map`: diagnostic one-hot MAP context from the same posterior.

`learned_soft` is the intended interface. `learned_map` distinguishes a good
estimator plus an out-of-distribution soft policy input from an inference
failure.

## Fixed gates

Inference passes only if the unseen-seed aggregate satisfies all conditions:

- mode accuracy after a 25-action segment burn-in is at least 0.85;
- median switch-detection delay is at most 25 actions;
- 90th-percentile delay is at most 50 actions;
- multiclass Brier score is at most 0.25.

Frozen-control feasibility passes for either learned arm only if:

- it recovers at least 50% of true-oracle switching headroom over robust;
- it beats robust on at least four of five unseen policy seeds;
- its switching termination rate is no more than five percentage points above
  robust.

Posterior-conditioned SAC training is authorized only when both inference and
frozen-control feasibility pass. A soft-arm failure with a MAP-arm pass
specifically authorizes training the policy under delayed, wrong, uniform, and
soft beliefs. An inference failure blocks policy training and sends the work
back to transition-model calibration.

## Result

All five untouched policy seeds and all three sealed event streams completed.
The frozen true-mode oracle continues to confirm large control headroom, but
the learned posterior fails both the inference and frozen-control gates:

| Arm | Switching return | Delta vs robust | Seed wins | Oracle headroom recovered |
|---|---:|---:|---:|---:|
| robust | 1023.4 | +0.0 | - | - |
| oracle | 2327.7 | +1304.3 | - | - |
| learned soft | 799.4 | -224.0 | 1/5 | -22.6% |
| learned MAP | 727.8 | -295.6 | 0/5 | -27.4% |

The soft posterior has mode accuracy `0.537`, Brier score `0.839`, median
switch delay `10`, and 90th-percentile delay `369.4` actions. Only the median
delay gate passes. Both learned contexts are termination-neutral, so the loss
is caused by wrong conditioning rather than survival bias.

The error is asymmetric rather than a general absence of transition signal.
On the maximally conflicting switching pair, mode 1 is classified correctly
on 98.4% of transitions, while mode 0 is classified correctly on only 7.9%;
its mean one-step likelihood is incorrectly highest under mode 2. Across the
five test policy seeds, however, a leave-one-seed-out linear calibration of
the centered four-head likelihood vector reaches `0.848` one-step accuracy,
`0.957` with a causal 10-action average, and `0.984` after burn-in with a
causal 25-action average. This diagnostic is not a reportable test result
because it uses cross-validation over the sealed seeds, but it establishes
that the frozen forward model retains mode information and that the failure is
likelihood ranking/calibration, not observability.

Direct posterior-conditioned SAC remains blocked. The next development stage
freezes this forward model and fits a compact causal evidence calibrator using
only controller seeds `8,16`; temporal hyperparameters are selected only on
seed `24`. The five sealed seeds remain excluded from fitting and are rerun
only after the calibrator is frozen.
