# V29 Ant oracle action-compensation preregistration

V28 showed that a frozen HalfCheetah reference policy plus causal actuator-sign
compensation outperforms robust SAC and the V21 policy bank. V29 asks only
whether the corresponding **oracle** mechanism is useful and survival-safe on
Ant before any Ant estimator is trained.

## Frozen inputs

- Environment: spring `Ant-v2`, persistent four-mode `actuator_polarity`,
  250-step dwell and 1,000-step audit horizon.
- Policies: the three existing V22 robust sources and four full-state
  specialists for seeds 85003, 85021 and 85039. No parameters are updated.
- The V28 structural audit is reused as the algebra/environment check for the
  exact polarity transform.
- Calibration events: 195001, 195017, 195033.
- Stationary holdout events: 195101, 195117, 195133.
- Switching events: 195201, 195217, 195233, with schedules frozen in code.

For each policy seed, calibration evaluates each specialist only in its native
mode. The reference is the highest-return mode after a dominating penalty for
any calibration termination. Holdout data cannot change this selection.

## Arms

1. Robust SAC.
2. The selected reference specialist without action compensation.
3. The same reference specialist with true-mode action-sign compensation.
4. The V22 matching-specialist bank with true-mode routing.

All arms share event seeds and mode schedules. Audits record return, every
termination, time to first termination, survival fraction, and executed-signal
error. The compensated switching trajectory is also compared with the same
reference policy running in its native stationary mode.

## Decision

Training an Ant causal estimator is authorized only if all of the following
hold:

- action error is at most `1e-6` and paired native-return error at most `1e-3`;
- compensated oracle beats robust SAC and the uncompensated reference for all
  three policy seeds and all nine switching seed-event cells;
- compensated oracle gains at least 10% over robust SAC for every policy seed;
- compensated oracle beats robust SAC in mean stationary return for every
  policy seed;
- compensated oracle has zero stationary and switching termination.

The dynamic specialist bank comparison is diagnostic and is not allowed to
relax this gate. Failure stops Ant estimator work; success authorizes only a
new development estimator experiment, not a paper-level confirmation claim.

## Accounting

V29 adds zero training interactions. Scheduler tasks are Linux CPU-only and
sync compact bundles, manifests and JSON; checkpoints, replay buffers and full
trajectory arrays remain remote.
