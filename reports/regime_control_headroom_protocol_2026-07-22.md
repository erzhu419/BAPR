# Regime-control headroom reset protocol

Date frozen: 2026-07-22

## Purpose

More than one hundred historical BAPR variants mixed estimator, router,
residual-policy, expert-bank, regularization, budget, and benchmark changes.
The next experiment therefore asks one narrower falsifiable question before
another learned estimator is built:

> Under an identical controller architecture and training budget, does exact
> knowledge of the current persistent actuator regime improve control over a
> policy that receives no regime information?

The earlier independent-specialist oracle is not sufficient for this test
because it changed the controller bank as well as the information available to
the controller. This reset changes only one input vector.

## Frozen environment

The family is `structured_channel` in `StochasticModeEnv`.

- Four modes impair, respectively, the low half, high half, even, or odd
  actuator channels with gain `0.45`.
- Gravity and robot morphology remain fixed. No mass, arm length, damping,
  friction, or gravity parameter is resampled each step.
- A mode persists for exactly 250 transitions during training and evaluation.
- Every mode has the same per-step Gaussian actuator noise standard deviation
  `0.04`. The stochastic draw changes each step; the affected channel subset
  does not change within a dwell.
- Mode transitions are generated independently of policy actions. Robust and
  oracle arms with the same seed receive the same training mode schedule.
- The frozen environments are `HalfCheetah-v2`, `Ant-v2`, and `Walker2d-v2`.
  Hopper is excluded because the structured-channel definition requires an
  even action dimension.

No environment magnitude, dwell, task set, horizon, reward, or termination
rule may be changed after results are observed. A future benchmark change is a
new named protocol, not a retry of this one.

## Controllers and fairness

Both arms use `RegimeSAC`: the same two-layer conditioned Gaussian actor, the
same ten-head conditioned critic, target critic, entropy temperature,
initialization, optimizer, replay format, and update code.

- `robust`: receives an all-zero four-dimensional context.
- `oracle`: receives the true one-hot mode used by the current physics step.

There is no encoder, residual branch, gate, BOCD, CUSUM, specialist bank,
LCB, Q-ensemble switch detector, or RE-SAC regularization term in either arm.
For a paired environment/training seed, parameters are bit-identical at
initialization; only `regime_context_source` differs.

Each run receives `1400 x 4000 = 5,600,000` environment transitions and
`1400 x 250 = 350,000` gradient updates, with hidden width 256, ten critic
heads, learning rate `3e-4`, horizon 1000, and no context warm-up. Training
seeds are `8,16,24,32,40`.

The GPU matrix contains 30 unpinned scheduler tasks:

`3 environments x 2 context arms x 5 independent training seeds`.

## Strict audit

Each completed checkpoint triggers one CPU-only audit task. The task evaluates
five sealed event seeds (`73100` through `73500` in steps of 100), each with:

- five deterministic strict-horizon episodes in every stationary mode;
- five deterministic 1000-step switching streams at dwell 250;
- simulator resets after termination without resetting the mode clock;
- exactly paired physical tasks, switching sequences, reset keys, and
  actuator-noise streams across robust and oracle arms.

The independent statistical unit is the training seed (`n=5`). Event seeds
are averaged within a training seed and are not treated as 25 independent
training replications.

## Preregistered gate

An environment passes only if all conditions hold:

1. Oracle switching return improves by at least 10% over robust.
2. The paired 95% interval for switching improvement is above zero.
3. Oracle worst-mode stationary return improves by at least 10%.
4. The paired 95% interval for worst-mode improvement is above zero.
5. Oracle improves at least three of four stationary modes.
6. Oracle switching termination rate is no more than 5 percentage points
   worse than robust.

A learned causal estimator is permitted only if at least two of three
environments pass. If the gate fails, another estimator cannot solve the
measured problem because even exact current-mode information lacks adequate
control value under the matched architecture and budget.

## Scope protection

This diagnostic does not modify the bus implementation or its benchmark. The
positive sign of the inherited RE-SAC regularization coefficient remains
unchanged in bus code. Bus retention becomes a separate hard gate only after
this oracle test permits a learned estimator.

## Final result

All 30 controller bundles and all 30 audit manifests completed. Every bundle
passed hash and budget validation at iteration 1399, next iteration 1400,
5,600,000 environment steps, and 350,000 updates. The independent local
aggregation reproduced the remote decision; the rendered Markdown was
byte-identical and JSON differences were limited to last-bit floating-point
rounding.

| Environment | Robust switching | Oracle switching | Relative gain | Robust worst mode | Oracle worst mode | Relative gain | Modes improved | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| HalfCheetah | 1945.0 +/- 326.3 | 1317.7 +/- 324.4 | -32.3% | 1553.9 +/- 451.5 | 832.2 +/- 402.8 | -46.4% | 0/4 | no |
| Ant | 2930.8 +/- 173.7 | 3291.4 +/- 238.3 | +12.3% | 2716.3 +/- 242.4 | 2906.6 +/- 176.2 | +7.0% | 4/4 | no |
| Walker2d | 2177.4 +/- 73.0 | 2200.6 +/- 18.3 | +1.1% | 192.7 +/- 54.2 | 172.2 +/- 38.7 | -10.6% | 1/4 | no |

Ant is the only positive information-value signal: its paired switching delta
is +360.6 with 95% interval [+22.4,+698.8]. It nevertheless fails because the
worst-mode gain is only 7.0% and its interval includes zero. HalfCheetah loses
significantly in switching and in every aggregate stationary comparison.
Walker2d provides essentially no oracle gain, and both arms terminate in every
stationary and switching audit stream, so it is not a useful adaptation
positive case under this frozen protocol.

Passing environments: **0/3**. Learned-estimator gate: **FAIL**. This result
forbids another estimator on this benchmark version; it does not invalidate
the separate bus result or prove that persistent-mode information has no value
under a newly specified environment/control formulation.

## Execution incident

Scheduler tasks `t50115-t50175` were submitted on 2026-07-22. The first bulk
dispatch cold-started 16 JAX processes on node007 at once. Five processes
(`t50118,t50119,t50130,t50134,t50139`) failed before iteration 0 while launching
`ptxas` or aborting under transient compile load; checkpoint-safe retries were
routed to the jtl GPU nodes.

A separate scheduler bug then classified every successful terminal marker
containing `HEADROOM` as OOM because the crash scanner searched for the bare
substring `OOM`. This produced false failed states, duplicate retries, and
withheld result synchronization despite `exit_code=0` and `DONE`. The matcher
now requires `OOM` to be a standalone token and has regression tests for both
`HEADROOM ... DONE` and real standalone `OOM`. Historical successful attempts
were reclassified, their scheduler-managed result sync completed, and no
controller was retrained during recovery. One Walker2d audit had a real
one-off CPU thread-creation abort and completed on its scheduler retry.
