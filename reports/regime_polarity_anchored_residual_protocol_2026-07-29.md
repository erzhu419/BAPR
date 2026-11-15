# Robust-anchored residual development protocol

## Motivation

The independent frozen-method confirmation established two facts:

1. the frozen expected-action v4 estimator transfers, with `0.995` mode
   accuracy, `0.008` Brier score, and `4/10` median/P90 switch delay;
2. the independently trained true-context controller is seed-unstable, so
   learned posterior control fails the preregistered five-seed gate despite
   staying close to its oracle.

This protocol changes the controller, not the estimator or environment. It
tests whether adaptation can retain an exact robust fallback while learning a
bounded mode-conditioned improvement.

## Frozen benchmark

- Environment: `HalfCheetah-v2`.
- Family: `actuator_polarity`.
- Four persistent actuator modes.
- Fixed dwell: 250 actions.
- Strict horizon: 1000 actions.
- Frozen estimator manifest:
  `fc4b28662ac1f38081aab02723ebcbea2fa47fbe37eea27bb4e1a564b8d571d0`.
- Frozen estimator parameters:
  `c1ff9021a3d3684116464c421c852eb16cd7ab286b2fd53bcd764d9a50393af1`.
- Online estimator inputs remain observation, commanded action, and next
  observation. Mode ID, realized action, and switch clock remain forbidden.

## Fresh development split

- Policy seeds: `1103,1213,1301`.
- Calibration event seeds: `96001,96002`.
- Audit event seeds: `96101,96102,96103`.

These policy seeds are disjoint from prior exploratory, development, and
confirmation splits. This is a three-seed development screen, not a paper
claim. A pass only authorizes an untouched five-seed confirmation.

## Paired training

Each seed first trains one robust `RegimeSAC` source from scratch for 1400
iterations, 5.6M transitions, and 350k update steps. The source then forks at
the same audited checkpoint into:

- `robust_continue`: another 700 iterations of the same robust controller;
- `anchored`: another 700 iterations of `AnchoredRegimeSAC`.

Both branches therefore finish at iteration `2099`, 8.4M transitions, and
525k update steps. Replay and optimizer states are reset at the fork for both
branches; policy, critic, target critic, alpha, and update count are copied
from the same source.

The anchored controller has:

- an observation-only robust actor that continues ordinary SAC training;
- a zero-initialized, bounded mode residual with cap `0.5`;
- separate robust and adaptive critic ensembles;
- no adaptive-to-robust actor or critic gradient path;
- a minimum critic target matching SAC;
- per-context entropy temperatures;
- paired robust/oracle replay relabels;
- a 1:1 robust/oracle rollout-source cycle;
- no BOCD, Q-std gate, LCB objective, raw IPM shift, or RE-SAC regularizer.

At the fork, robust and every one-hot adaptive context must produce the same
deterministic action and Q values within the registered numerical tolerance.
The residual output is exactly zero.

## Explicit fallback calibration

For each policy seed, the calibration streams compare:

- equal-budget `robust_continue`;
- the anchored checkpoint with gate zero, `anchored_base`;
- the same anchored checkpoint with true mode and gate one,
  `oracle_residual`.

A mode is enabled only when `oracle_residual` exceeds the better robust
reference by at least 2% and its termination rate is no more than two
percentage points above the safer robust reference. The resulting four-bit
mask is frozen before audit.

During audit:

- `learned_raw` always applies the frozen soft posterior;
- `learned_safe` applies the soft posterior only when its maximum probability
  is at least `0.85` and its MAP mode is enabled by calibration;
- otherwise `learned_safe` supplies gate zero and is exactly the anchored
  robust actor.

## Audit arms

Every audit event evaluates six deterministic arms on shared stationary and
switching streams:

1. `robust_continue`;
2. `anchored_base`;
3. `oracle_residual`;
4. `oracle_safe`;
5. `learned_raw`;
6. `learned_safe`.

The primary learned arm is `learned_safe`.

## Development gates

Promotion requires all of the following:

- `anchored_base` preserves at least 95% of `robust_continue` on every seed;
- `oracle_residual` gains at least 5% on average and wins at least 2/3 seeds;
- `learned_safe` gains at least 3% on average and wins at least 2/3 seeds;
- no relevant arm adds more than two percentage points of termination;
- frozen-estimator accuracy, Brier score, and median/P90 switch delay retain
  the previously frozen thresholds;
- calibration enables at least one mode for every seed.

Failing any gate blocks a five-seed expansion. No threshold is changed after
the audit.

## Scheduler graph

The graph contains 16 file-gated tasks:

- 3 fresh robust-source GPU tasks;
- 3 robust-continuation GPU tasks;
- 3 anchored GPU tasks;
- 3 CPU calibration tasks;
- 3 CPU audit tasks;
- 1 CPU aggregate.

GPU tasks may use `local`, `jtl110gpu`, `jtl110gpu2`, or `node007`.
`jtl311linux` is excluded. CPU-only evaluation may use `node001-node006`.
The graph uses scheduleurm only, with scheduler-managed checkpoint resume and
node-down rerouting.

The local full-shape update smoke measured 237 MB incremental device memory
for robust and 369 MB for anchored, a 1.56x ratio. Historical remote robust
tasks peaked at 1548 MB. Claims are therefore 1800 MB for robust and 2600 MB
for anchored, rather than an unmeasured 4/8 GB default. The real MuJoCo
two-iteration smoke also verified rollout sources `[robust, oracle]`, dual
context relabeling, logging, and checkpoint persistence.

The graph was submitted at high priority as `t59185-t59200`:

- source controllers: `t59185-t59187`;
- robust continuations: `t59188-t59190`;
- anchored continuations: `t59191-t59193`;
- calibrations: `t59194-t59196`;
- strict audits: `t59197-t59199`;
- aggregate: `t59200`.

At submission, the three sources entered real training on node007 GPU0/1/2.
All 13 downstream tasks remained queued with their expected missing-file
reason; none was launched before its producer artifacts existed.

## Development result

All source, branch, calibration, audit, and aggregate artifacts validate.
Every source finishes at iteration `1399`, 5.6M transitions, and 350k
updates. Every branch finishes at iteration `2099`, 8.4M transitions, and
525k updates. Bundle and audit file hashes match their manifests.

The method fails every controller promotion gate:

| Arm | Mean delta vs robust | Relative gain | Seed wins |
|---|---:|---:|---:|
| `anchored_base` | -361.3 | -38.6% | 1/3 |
| `oracle_residual` | -760.5 | -72.4% | 1/3 |
| `oracle_safe` | -347.0 | -37.8% | 1/3 |
| `learned_raw` | -755.1 | -71.8% | 1/3 |
| `learned_safe` | -349.2 | -37.9% | 1/3 |

The per-seed switching means are:

| Seed | Mode mask | Robust | Anchored base | Oracle residual | Learned safe |
|---:|---|---:|---:|---:|---:|
| 1103 | `0000` | 1402.9 | 848.2 | -76.9 | 848.2 |
| 1213 | `0000` | 774.1 | 139.0 | -137.6 | 139.0 |
| 1301 | `0111` | 1823.0 | 1929.0 | 1933.1 | 1965.4 |

Termination is zero for every arm, so it does not explain the loss. The
frozen estimator is also not the primary cause: seeds 1103 and 1301 reach
`0.9996` mode accuracy with median delay below four actions, yet only seed
1301 has usable residual modes. Calibration disables all modes for seeds
1103 and 1213 and modes 1-3 only for seed 1301.

The failed preservation gate exposes a design error in the intended anchor.
Although adaptive rows have no gradient into the robust actor, that actor is
still updated on a different closed-loop replay distribution: half of the
rollouts are generated by the adaptive policy, then relabelled as robust.
Consequently it is not the same paired robust actor after the fork. The
gate-zero path loses 39.5% and 82.0% on two seeds. The shared bounded
residual is also an unstable controller: true mode drives switching return
below zero on those same seeds.

This result does not authorize a five-seed confirmation. The next development
branch must start from the completed `robust_continue` checkpoint, freeze its
base actor byte-for-byte, and train only a zero-initialized adaptive path.
It should compare a shared bounded residual with independent mode residual
heads while retaining an equal-total-budget robust continuation and explicit
calibrated fallback.
