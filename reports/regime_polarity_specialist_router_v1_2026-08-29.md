# Causal independent-specialist router development protocol

## Motivation

The fresh ten-seed deployment confirmation places BAPR at 1981.5 switching
return. It is stable and exceeds SAC, but its frozen shared teacher and
true-context student remain near a 2.0k ceiling. In contrast, the earlier
independent-specialist source audit reaches 2762.6 and 3103.9 with privileged
true-mode routing. This protocol tests whether the already frozen causal
estimator can recover that controller headroom without changing the benchmark.

## Frozen inputs

- Independent specialist banks and matched robust controllers from source
  seeds `4021` and `4049`.
- Frozen expected-action inverse-system-ID estimator and frozen causal fallback.
- New development event streams `153101`, `153113`, and `153127`.
- Strict deterministic 1000-step evaluation with 250-step mode dwell.

## Arms

- `robust_sac`: matched robust controller.
- `dynamic_oracle`: privileged true-mode specialist selector.
- `posterior_map_fallback`: robust fallback while uncertain, then the MAP
  specialist selected from the causal posterior.
- `posterior_soft_fallback`: the same fallback with posterior-weighted
  specialist actions after release.

The learned arms observe only observation, commanded action, reward, and next
observation. They do not receive true mode, actuator gain, executed action, or
the switch clock.

## Development gate

A learned arm advances only if it passes on both source seeds: at least 10%
switching gain over matched robust, at least 2200 switching return, at least
70% recovery of both stationary and switching oracle headroom, wins on all
three event streams, and has no switching termination penalty. Passing permits
fresh independent-specialist training and a separate confirmation; failure
stops this construction without spending GPU budget.
