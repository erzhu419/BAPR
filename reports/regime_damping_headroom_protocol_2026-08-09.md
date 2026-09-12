# Persistent joint-damping oracle-headroom protocol

## Question

Before training another BAPR estimator, test whether a true persistent mode can
improve control over an equal-budget robust policy. This benchmark is an
untouched positive-control candidate, not an environment tuned against BAPR
returns.

## Environment

Each mode lasts 250 simulator actions. Robot morphology and physics remain
fixed throughout the dwell. A mode increases damping on one registered subset
of actuated joints: low half, high half, even, or odd. The L2 norm of the log
damping displacement is equalized across modes, including odd action
dimensions. Every mode has the same small per-step action noise of 0.02.

No mode resamples body shape, damping, gravity, or friction per step. The only
per-step randomness is execution noise. The environment family, severity,
seeds, budgets, and gate are frozen before training by a source registration.

## Arms and budget

For each of HalfCheetah, Ant, Hopper, and Walker2d, train:

- robust `regime_sac`, which receives an all-zero four-mode context;
- oracle `regime_sac`, which receives the true one-hot persistent mode.

Both use the same actor, critic, optimizer, 5.6 million environment steps, and
350,000 gradient updates. Training seeds are `32011, 32117, 32233`. Strict CPU
evaluation uses untouched event seeds `132011, 132117, 132233`, five stationary
episodes per mode, and five 1000-step switching streams.

## Gate

An environment passes only if oracle exceeds robust by at least 10% for both
switching return and worst stationary mode, both training-seed-paired 95%
intervals are above zero, at least three of four stationary modes improve, and
the switching termination gap is no more than five percentage points.

The benchmark family passes only if at least three of four environments pass.
Only then may a causal learned estimator be trained. A failure is retained as
a negative headroom result; the environment or threshold is not adjusted using
these returns.
