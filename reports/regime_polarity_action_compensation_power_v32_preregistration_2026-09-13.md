# V32 prospective action-compensation power confirmation

## Purpose

V31 showed a large positive mean for causal compensation but failed its
five-seed confidence-interval gates against equal-budget SAC and RE-SAC. V31 is
therefore a failed pilot, used only for prospective sample-size planning. Its
outcomes will not be pooled with V32.

The V31 paired standardized effects were 1.197 against SAC and 1.222 against
RE-SAC. Ten new policy seeds give estimated two-sided one-sample t-test powers
of 0.919 and 0.929, respectively.

## Frozen experiment

- Environment: HalfCheetah actuator-polarity regimes.
- Policy seeds: 88003, 88021, 88039, 88057, 88079, 88103, 88121, 88139,
  88157, 88179.
- Canonical controller: fixed mode 0.
- Compensation: the V31 action transform and frozen v5 causal estimator.
- Training: 5.6M robust-source interactions plus 2.8M fixed-mode fine-tuning.
- Comparators: SAC, recurrent ESCP, and RE-SAC b0, each trained for 8.4M
  interactions.
- Evaluation: three new stationary streams and three new switching streams.
- No estimator retraining, reference-mode calibration, checkpoint selection, or
  reuse of V31 policy or event seeds.

## Primary decision

Causal compensation must beat each equal-budget comparator with all of:

- a positive paired ten-seed mean;
- a two-sided 95% paired t confidence interval whose lower bound is positive;
- at least 8/10 policy-seed wins;
- at least 24/30 policy-seed by switching-event wins.

The overall result additionally requires exact oracle action/trajectory
equivalence, oracle headroom in at least 8/10 seeds, recovery of the frozen
oracle headroom in at least 8/10 seeds, and stationary retention in at least
8/10 seeds. Any failed primary comparator gate is a failed confirmation.

## Claim boundary

A pass supports only causal action compensation for invertible HalfCheetah
actuator-polarity regimes. It does not establish general adaptation to arbitrary
dynamics changes or termination safety. A failure remains part of the record
and will not be repaired by pooling V31 or by adding seeds post hoc.
