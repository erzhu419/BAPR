# Conservative frozen-residual development protocol

## Motivation

The frozen-anchor v2 result localized the remaining failure to controller
optimization. The expected-action estimator was already accurate
(`99.58-99.96%`, median switch delay `2-3` actions), while true-mode oracle
control improved equal-budget robust control by only `8.3%` for
`shared_small` and regressed sharply for wider or independent residuals.
Therefore another estimator or gate change is not justified.

This v3 protocol keeps the seed-matched robust actor byte-for-byte immutable,
trains only a bounded residual, and constrains the residual against the exact
base action under the same state and mode context. The target-critic ensemble
computes

`LCB[Q(s,a_res,z)-Q(s,a_base,z)] / max(mean|Q_base|,1)`.

A smooth actor penalty pushes this normalized LCB above zero. After each
candidate actor step, the update is rejected together with its optimizer state
if any represented mode violates its registered relative-advantage floor or
regresses beyond its tolerance.

## Fixed benchmark and budget

- Environment: `HalfCheetah-v2`.
- Family: persistent `actuator_polarity`.
- Dwell: fixed 250 actions.
- Per-transition actuator noise: Gaussian `std=0.02`.
- Development policy seeds: `1103,1213,1301`.
- Calibration event seeds: `96401,96402`.
- Strict audit event seeds: `96501,96502,96503`.
- Frozen expected-action estimator: unchanged.

Adaptive branches fork from the same iteration-2099 robust checkpoints used by
frozen-anchor v2: 8.4M transitions and 525k updates. Replay and optimizer states
are reset, then each branch receives 700 iterations and finishes at iteration
2799, 11.2M transitions, and 700k updates.

The completed v2 `robust_long` bundles at the identical 11.2M budget are reused
read-only. They are not retrained or copied into the v3 output tree.

## Registered variants

All variants use a frozen robust actor, zero-initialized shared residual,
oracle-mode development rollouts, residual SAC minimum targets, and a
target-critic LCB scale of `1.0`.

1. `strict_small`: residual cap `0.15`, action-deviation weight `0.05`, zero
   update-regression tolerance, and nonnegative mode LCB floor.
2. `trust_small`: residual cap `0.15`, action-deviation weight `0.05`, update
   tolerance `0.005`, and mode LCB floor `-0.01`.
3. `trust_tight`: residual cap `0.075`, action-deviation weight `0.05`, update
   tolerance `0.005`, and mode LCB floor `-0.01`.

The relaxed variants distinguish genuine residual failure from critic noise
that causes the strict filter to reject every useful update. The tight variant
tests whether v2's 0.15 residual range is itself too destructive.

## Audit and decision rule

Held-out calibration and strict causal audit retain the v2 arms:
equal-budget robust, immutable adaptive-checkpoint base, true-mode residual,
oracle-safe, learned-raw, and learned-safe. The new event streams are disjoint
from v2 development.

The existing promotion gates are unchanged: immutable-base preservation at
least 95% on every seed, true-mode gain at least 5% on average with 2/3 seed
wins, learned-safe gain at least 3% with 2/3 wins, bounded termination gaps,
and all frozen-estimator accuracy/delay checks.

Additional interpretation is fixed before launch:

- near-zero actor update acceptance means the critic constraint is too noisy or
  the residual has no certified local ascent direction;
- high acceptance with negative strict return means the critic is
  miscalibrated for policy improvement;
- positive oracle but negative learned-safe performance would return the
  bottleneck to posterior-conditioned deployment;
- failure of all three variants ends this actuator-polarity controller line
  rather than triggering another threshold sweep.

## Scheduler graph

The file-gated matrix contains 30 tasks:

- 9 GPU adaptive continuations;
- 9 CPU held-out calibrations;
- 9 CPU strict audits;
- 3 CPU aggregates.

GPU training may use only `local`, `jtl110gpu`, and `node007`.
`jtl110gpu2` is excluded while thermally unstable, and `jtl311linux` remains
excluded. CPU evaluation may use only `node001-node006`. Submission is through
scheduleurm with checkpoint-managed resume; no Slurm or auto-adopt path is
used.

The registered graph was submitted at high priority as `t64314-t64343`:

- GPU branches `t64314-t64322`;
- calibrations `t64323-t64331`;
- strict audits `t64332-t64340`;
- aggregates `t64341-t64343`.

All nine branches resumed from `iter=2100`, 8.4M transitions. The launch
command and runtime config both confirm the registered advantage constraint,
per-mode update filter, tolerance, floor, residual cap, and action-deviation
weight. The first placement used `node007` only; downstream CPU work remained
file-gated.

## Completed result

All 30 tasks `t64314-t64343` completed and published their expected local
artifacts. Every adaptive branch reached iteration `2799`, corresponding to
11.2M transitions and 700k controller updates. The three registered variants
produced exactly the same strict-audit result:

| seed | robust long | frozen base | oracle residual | learned safe |
|---:|---:|---:|---:|---:|
| 1103 | 1468.2 | 1410.7 | 1410.7 | 1410.7 |
| 1213 | 600.5 | 772.3 | 772.3 | 772.3 |
| 1301 | 1902.8 | 1752.3 | 1752.3 | 1752.3 |

The mean paired delta is `-12.1`; the arithmetic mean of the three relative
gains is `+5.6%` because the low-return seed 1213 contributes `+28.6%`, but
only `1/3` seeds wins and the paired interval spans substantial loss. Every
calibration mask is `0000`. Estimator accuracy remains `0.9958-0.9996`, median
switch delay is `2-3` actions, and no evaluated arm terminates.

This is not evidence that residual cap `0.075` or `0.15` is inadequate. All
nine adaptive-policy hashes are unchanged from their fork checkpoints.
Training update acceptance is exactly zero for every seed and variant; the
logged current-policy normalized LCB is exactly zero and the smooth shortfall
is `0.00693147` (`0.01 * log(2)`). Consequently `oracle_residual`,
`learned_raw`, `learned_safe`, and `anchored_base` are the same policy.

## Failure localization

The all-mode candidate filter vetoed every shared-residual update. A single
minibatch candidate had to preserve the target-critic LCB for every represented
mode simultaneously. This all-or-nothing rule can freeze a shared policy when
mode gradients conflict, when one mode's target critic is pessimistic, or when
the useful actuator-polarity solution requires crossing a locally negative
path. The completed batch therefore did not compare residual capacities and
did not retest the already validated mode estimator.

Instrumentation now records candidate minimum/mean LCB, minimum regression and
floor margins, and separate non-finite, regression, and floor rejection rates.
A non-finite sample is isolated to its represented mode rather than
contaminating all mode aggregates. Twelve related regression tests pass.

Before another full controller run, a short scheduler diagnostic must resume
one source checkpoint for only a few updates and identify which rejection
condition is active. The next structural candidate, if finite candidates are
systematically vetoed across modes, is an immutable robust actor with bounded
mode-specific residual heads and per-head acceptance/rollback. A global
all-mode veto on one shared parameter tree is retired. No untouched
confirmation or learned-estimator retraining is justified by this result.

## Exact post-hoc diagnosis

The candidate instrumentation changed the interpretation above. Fresh
one-iteration diagnostic tasks `t64456-t64458` found `nonfinite_rate=1.0` for
all three variants and all four modes, while finite regression and floor
rejection rates were zero. The global veto was behaving correctly; every
candidate policy was non-finite.

At the zero-residual anchor, every critic head has identical
`Q_adaptive-Q_base=0`. The direct ensemble `std` has a forward value of zero
but an undefined reverse-mode derivative at zero variance. The smooth LCB
penalty therefore injected NaNs into the first residual gradient. This is the
direct cause of all nine unchanged v3 actors.

The LCB now uses `sqrt(variance+1e-6)-sqrt(1e-6)`, preserving an exact
zero-valued anchor with a finite derivative. Clean diagnostic tasks
`t64460-t64462` then show:

| variant | accept rate | candidate min LCB | non-finite | actor changed |
|---|---:|---:|---:|---|
| strict_small | 0.000 | -0.00040 | 0.000 | no |
| trust_small | 0.976 | -0.00524 | 0.000 | yes |
| trust_tight | 1.000 | -0.00345 | 0.000 | yes |

Thus the strict arm is locally frozen by its registered zero floor, but both
relaxed arms now perform real finite optimization. The original v3 return
comparison is invalid rather than a negative controller result. A clean v4
full rerun retains only `trust_small` and `trust_tight`; its separate
registration is in
`reports/regime_polarity_conservative_residual_stable_protocol_2026-07-30.md`.
