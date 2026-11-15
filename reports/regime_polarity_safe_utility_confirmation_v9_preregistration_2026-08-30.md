# V9 independent safe-utility confirmation preregistration

## Frozen candidate

The candidate is the v8 robust-inclusive policy bank with the frozen v5
expected-action estimator. At each step, the estimator posterior MAP selects a
mode. A bank-specific stationary calibration map either enables that mode's
matching specialist or substitutes the robust SAC actor. No confirm-3 delay,
BOCD, Q-variance gate, or post-holdout threshold change is permitted.

## Fresh split

- Policy-bank and paired-baseline seeds: `61003, 61021, 61039, 61057, 61079`.
- Stationary calibration streams: `156001, 156017, 156033`.
- Untouched switching holdouts: `156101, 156117, 156133`.
- Per seed: robust SAC, four fixed-mode SAC specialists, recurrent ESCP, and
  released-B0 RE-SAC are trained from scratch.
- All controllers use 1,400 iterations, 5.6 million environment steps, 350,000
  updates, a 1,000-step strict horizon, the same stochastic-mode family, and the
  same fixed dwell schedule.

## Frozen selection rule

For mode `m`, enable specialist `m` only if it beats the robust actor by at least
5% on the mean stationary calibration return, wins all three calibration
streams, and has zero termination. Otherwise use the robust actor for that
mode. Holdout switching returns cannot change this map.

## Bank-specific gate

When true-mode safe-utility headroom over robust is at least 10%, posterior-MAP
safe utility must gain at least 10%, recover at least 70% of headroom, win all
three switching holdouts, and have zero termination. Without 10% headroom, it
must remain within 5% of robust on every holdout and have zero termination. All
five policy banks must pass.

## Baseline comparison

The paired comparison is confirmed for a comparator only if the two-sided 95%
paired interval for MAP-minus-comparator return is above zero, MAP wins at least
four of five training seeds, and MAP has no larger termination rate. The three
comparators are the policy bank's robust SAC actor, recurrent ESCP, and
released-B0 RE-SAC.

## Cost boundary

BAPR trains five policies per seed: 28 million interactions and 1.75 million
updates. Each comparator trains one policy: 5.6 million interactions and 350,000
updates. The comparison therefore measures deployment performance under equal
per-controller budgets, not equal total construction cost. Frozen estimator
pretraining is additional and excluded from the 5x ratio.
