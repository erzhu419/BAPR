# Fresh five-seed polarity confirmation

## Decision basis

The checkpoint-only context-causality audit advances only HalfCheetah.
Across the exploratory training seeds `8,16,24`, dynamic true context scored
`2633.9` versus `738.8` for the equal-budget robust controller and `1394.5`
for the per-seed best fixed-context envelope. All four stationary rows were
diagonal-optimal. A 10-action delayed oracle retained `87.9%` of instantaneous
oracle headroom and a 25-action delay retained `64.0%`; both beat robust on
every exploratory seed.

Ant does not advance. Its 10- and 25-action delayed oracles retained only
`59.8%` and `15.2%` of headroom, and delay 25 did not beat robust on every
seed. Hopper and Walker2d were already rejected by the preceding headroom
screen.

## Sealed confirmation

This protocol is fixed before confirmation jobs are submitted.

- Environment: `HalfCheetah-v2`
- Persistent family: `actuator_polarity`
- Dwell: fixed 250 actions
- Arms: all-zero-context robust controller and true-mode oracle controller
- New training seeds: `101,211,307,419,523`
- Sealed evaluation event seeds: `91001,91002,91003`
- Budget per arm and seed: 1,400 iterations, 5.6M transitions, 350k updates
- Architecture and optimization: identical to the exploratory headroom screen
- Evaluation: deterministic policy, four held-out stationary modes plus five
  strict 1,000-action switching episodes per event stream
- Inferential unit: independent training seed (`n=5`); event streams are
  averaged within seed

No exploratory checkpoint initializes a confirmation run. Checkpoint resume is
allowed only within the same confirmation arm and seed.

## Confirmation gate

All conditions must hold:

1. Oracle switching return exceeds robust by at least 10%, and the paired 95%
   training-seed confidence interval for the absolute difference is above zero.
2. Oracle worst-mode stationary return exceeds robust by at least 10%, and its
   paired 95% confidence interval is above zero.
3. Oracle mean return improves in at least three of four stationary modes.
4. Oracle switching termination rate is no more than five percentage points
   above robust.

Passing this gate authorizes implementation of a learned causal posterior.
Failure blocks estimator training on this benchmark; it cannot be repaired by
selecting favorable seeds or adding two seeds to the exploratory sample.

## Scheduler graph

The graph contains 10 fresh GPU training producers, 10 file-gated CPU audits,
and one file-gated CPU aggregate. It uses scheduleurm only, excludes
`jtl311linux`, and assigns the measured 2.3 GB VRAM estimate rather than an
unmeasured 4/8 GB default.

- GPU training: `t54602-t54611`
- File-gated CPU audits: `t54612-t54621`
- File-gated aggregate: `t54622`

At submission, `t54602` launched on local and `t54603` on `jtl110gpu`.
The remaining producers are unpinned and retain normal checkpoint-safe
rerouting. Their initial queue state is resource pressure, not a protocol
dependency; `jtl311linux` remains outside the allowed-node set.

## Audit recovery (2026-07-28)

The original robust CPU audits reached the final 5.6M-step checkpoint and
completed their first evaluation stream, but a shared legacy validator then
incorrectly required the robust arm's action-context ID to equal the physics
mode. In this protocol the robust policy is intentionally supplied with the
all-zero context, which `RegimeSAC` records as `-1`. The validator now checks
that `-1` is preserved for the robust arm while retaining the exact
physics-context alignment check for oracle traces. Five replacement CPU audits
`t56196-t56200` run from the existing sealed bundles; no GPU controller is
retrained and no checkpoint is modified.

## Sealed confirmation result (2026-07-28)

All ten controller bundles reached the sealed 5.6M-transition/350k-update
budget. The five replacement robust audits, the five original oracle audits,
and aggregate task `t54622` completed with 10/10 manifests. All action-level
context, event-seed, checkpoint-hash, horizon, and termination checks passed.

HalfCheetah passes every preregistered confirmation condition:

| Metric | Robust | True-mode oracle | Paired delta | 95% CI | Seed wins |
|---|---:|---:|---:|---:|---:|
| Switching return | 1033.8 +/- 318.8 | 2308.0 +/- 475.4 | +1274.2 (+123.3%) | [750.2, 1798.1] | 5/5 |
| Worst-mode stationary return | 682.2 +/- 289.9 | 2012.3 +/- 486.4 | +1330.1 (+195.0%) | [932.7, 1727.5] | 5/5 |
| Mean stationary return | 1071.3 +/- 472.5 | 2320.8 +/- 450.8 | +1249.5 | [715.2, 1783.8] | 5/5 |

The oracle improves all four stationary modes, with 5/5 seed wins in every
mode. The mode-wise paired deltas are `+1451.0`, `+1059.0`, `+1122.4`, and
`+1365.6`; every corresponding 95% interval is strictly positive. Robust and
oracle termination rates are both zero.

This is a strong positive-control result rather than a marginal threshold
pass. The actuator-polarity HalfCheetah benchmark has reproducible adaptation
headroom under independent policy seeds, and the preceding delay audit shows
that useful headroom remains under a causal 10-25 action detection delay.
Learned causal posterior work is therefore authorized. Ant, Hopper, and
Walker2d remain blocked under their existing protocol.

The next stage is deliberately estimator-first. A heteroscedastic transition
model and sticky semi-Markov posterior must be trained and evaluated on
disjoint event streams while the validated controllers remain frozen. It must
report physical-mode accuracy, calibration, switch-detection delay, and the
return obtained when its causal posterior drives the frozen conditioned
controller. Full posterior-conditioned SAC training is allowed only if that
screen recovers a material fraction of the delayed-oracle headroom.
