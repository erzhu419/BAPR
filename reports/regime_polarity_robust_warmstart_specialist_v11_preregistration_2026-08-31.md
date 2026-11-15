# Robust-warm-started specialist development protocol

## Question

V10 showed deployable four-step causal margin whenever a policy bank had
headroom, but only three of five independently trained banks had that headroom.
This development screen tests one controller-side explanation: fixed-mode SAC
from scratch is unstable, while a matched robust SAC controller may provide a
reliable initialization.

No estimator, posterior, gate, residual policy, regularization term, or
environment parameter is changed. Seeds `61039` and `61057` are reused only as
development seeds because their v9 maps were `[R,R,R,R]`; they cannot later be
counted as confirmation evidence.

## Frozen arms

Each of four fixed-mode controllers starts from the same matched robust SAC
actor at iteration 1400 and 5.6M transitions, clears replay, and trains for 700
additional iterations (2.8M transitions) in one persistent mode.

- `full_state`: preserve actor, critic, target critic, and entropy temperature;
  reset all Adam states and replay.
- `actor_only`: preserve only the actor; initialize critic, target critic,
  entropy temperature, Adam states, and replay freshly.

The actor must be functionally identical to the matched robust actor at the
fork boundary. Training tasks produce policy-only inference bundles; critic,
optimizer, replay, and full checkpoints remain remote and are not synchronized
as results.

## Data split

- stationary calibration: `157001, 157017, 157033`;
- stationary holdout: `157101, 157117, 157133`;
- switching holdout: `157201, 157217, 157233`.

Calibration enables a specialist only when it beats robust by at least 5%,
wins all three calibration streams, and has zero termination. Holdout streams
are not used to select the map or either initialization variant.

## Gate

A `(variant, seed)` cell passes only when:

1. matching specialists beat robust by at least 5%, on all three stationary
   holdout streams, with zero termination, in at least `3/4` modes; and
2. the calibrated true-mode safe oracle beats robust by at least 10% on all
   three switching holdout streams with zero termination.

A variant advances only if both development seeds pass. If neither variant
passes, robust warm-starting is not sufficient and independent specialist-bank
optimization is closed as the main BAPR path. If one passes, the simpler
passing initialization is frozen and evaluated on entirely new policy and
event seeds before any estimator is reintroduced.

This is a controller-capacity development experiment, not a final BAPR versus
baseline comparison. Its additional robust pretraining and four specialist
fine-tuning budgets must remain explicit in any later report.
