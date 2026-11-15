# Stable-LCB conservative residual rerun

## Reason for rerun

The completed v3 matrix was invalid as a controller comparison. Its
zero-initialized residual makes every critic head's
`Q(s,a_adaptive)-Q(s,a_base)` identical. The original LCB used a direct
ensemble `std`; its forward value was zero but its gradient at zero variance
was undefined. Every first actor candidate therefore contained non-finite
values and was rolled back. This explains the zero acceptance and unchanged
actor hashes across all nine v3 branches.

The LCB standard deviation is now
`sqrt(variance + 1e-6) - sqrt(1e-6)`. It remains exactly zero at the robust
anchor and has a finite gradient. Unit tests cover the zero-residual gradient
and the full candidate-update/rollback path.

## Short diagnostic

Fresh one-iteration tasks `t64460-t64462` restarted from the clean 8.4M source
checkpoint. All reached iteration 2100, 8.404M transitions, and 525250
updates. Non-finite candidate rate is zero in every mode.

| variant | accept rate | candidate min LCB | regression reject | floor reject | actor changed |
|---|---:|---:|---:|---:|---|
| strict_small | 0.000 | -0.00040 | 1.000 | 1.000 | no |
| trust_small | 0.976 | -0.00524 | 0.000 | 0.024 | yes |
| trust_tight | 1.000 | -0.00345 | 0.000 | 0.000 | yes |

The strict zero-floor arm still cannot leave the zero residual and is retired.
The registered relaxed arms now exercise actual policy optimization and merit
the full three-seed rerun.

## Fixed full protocol

- Environment, task family, dwell, noise, source checkpoints, event seeds,
  training budget, estimator, audit arms, and promotion gates are unchanged
  from v3.
- Variants: `trust_small` (cap 0.15) and `trust_tight` (cap 0.075).
- Each branch starts from the clean iteration-2099 robust source, trains for
  700 iterations, and finishes at iteration 2799 and 11.2M transitions.
- The old v3 checkpoints are never resumed.
- GPU nodes: `local`, `jtl110gpu`, `node007`.
- CPU nodes: `node001-node006`.
- `jtl110gpu2` and `jtl311linux` are excluded.
- Scheduler only; no Slurm or auto-adopt.

The graph has 20 tasks: six GPU branches, six calibrations, six strict audits,
and two aggregates. No learned-estimator retraining or untouched confirmation
is part of this rerun.

## Scheduler launch

The complete graph is `t64464-t64483`:

- GPU branches: `t64464-t64469`;
- file-gated calibrations: `t64470-t64475`;
- file-gated strict audits: `t64476-t64481`;
- file-gated aggregates: `t64482-t64483`.

All six producers entered running state on `node007`; the scheduler placed two
measured 2.8-GB claims on GPU0 and GPU1 and one on GPU2 and GPU3. Every CPU
task remained queued on its declared prerequisite files.

## Completed result (2026-08-01)

All 20 tasks `t64464-t64483` completed and their result directories were
synced. Both registered variants fail promotion. The numerical-stability fix
worked: all six adaptive-policy hashes changed, no candidate update was
non-finite, and mean actor-update acceptance was `0.959-1.000`. The result is
therefore a valid test of the intended conservative residual rather than a
repeat of the frozen v3 run.

| variant | oracle residual delta vs robust | oracle wins | learned raw delta | learned safe delta | masks by seed 1103/1213/1301 |
|---|---:|---:|---:|---:|---|
| `trust_small` | `+29.8`, CI `[-92.2, 137.4]` | `2/3` | `+38.3` | `+9.2` | `0000 / 0000 / 1010` |
| `trust_tight` | `-5.6`, CI `[-140.8, 187.6]` | `1/3` | `-4.6` | `+25.9` | `0000 / 0100 / 1000` |

The paired robust switching returns for seeds `1103/1213/1301` are
`1468.2/600.5/1902.8`. `trust_small` oracle residual obtains
`1376.0/737.9/1947.2`; `trust_tight` obtains
`1404.6/788.1/1762.1`. Neither variant has stable seed-level gain, and the
held-out calibration has no mode that can be enabled for every seed.

Mode inference is not the limiting component. Frozen-estimator accuracy is
`0.9952-0.9996`, median switch delay is `2-3` actions, and every estimator
gate passes. The critic certificate is also not calibrated to realized
control return: for example, `trust_tight` seed `1213` reports mean training
LCB `+0.2824`, but its true-context switching gain over its own frozen base is
only about `+2%`. Conversely, accepted updates can reduce return materially in
other modes and seeds.

This closes the bounded local-residual/critic-LCB branch. A local residual
around an immutable robust action does not reliably reach the nonlocal
controller changes required by actuator sign reversal, and additional LCB,
floor, cap, or calibration sweeps are not justified. The previously trained
true-context policy still establishes substantial adaptation headroom, so the
next diagnostic is checkpoint-only controller-variance analysis: evaluate
multiple independently trained robust and oracle policies on identical fresh
event streams, then test deterministic mean-action and median-action policy
ensembles. Only if that stabilizes the oracle advantage should a single
posterior-conditioned student be distilled from the ensemble.
