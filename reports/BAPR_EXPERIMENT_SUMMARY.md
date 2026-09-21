# BAPR experiment summary

Last updated: 2026-09-12

This is the single retained experiment summary after deleting historical raw
checkpoints, result arrays, plots, and intermediate reports. The final v82/v83
artifacts were also removed after the corrected sweep was synchronized and
audited; source code and scheduler metadata remain.

## Bottom line

- The strongest completed legacy BAPR reference is **v45**, not any later
  controller variant. Its 3-seed tail-20 returns are HalfCheetah
  `18124.2 +- 1189.6`, Ant `1776.5 +- 623.5`, Hopper `1909.0 +- 858.6`, and
  Walker2d `1607.5 +- 1161.4`.
- v73-v80 did not produce a stable cross-environment replacement for v45.
  Several Ant-only improvements disappeared under additional seeds or hurt
  HalfCheetah/Hopper/Walker2d.
- The old benchmark and evaluation path contained material protocol defects:
  some physics parameters did not affect MuJoCo, train/test task splitting was
  wrong, and the first switching sweep barely changed gravity. Results made
  before the environment audit are useful only as development history.
- After fixing the protocol, the original single-observation context was found
  to be collapsed. BAPR-v2 therefore replaced it with a causal transition
  history encoder and a robust-base plus residual-policy architecture.
- Corrected evidence does not support a universal four-environment claim. The
  frozen V21 recipe does support a strong five-seed HalfCheetah result under
  the persistent actuator-polarity benchmark, including the equal-policy-budget
  control. Ant retains large oracle headroom but fails controller-reliability
  gates through V26; Hopper and Walker2d lack survival-valid headroom at the
  tested polarity severity.
- V21 HalfCheetah is sealed positive evidence. V26 Ant is a sealed negative
  transfer result and must not be followed by another posterior, gate,
  risk-weight, or controller-budget sweep on the same development seeds.
- V86's learned continuous blend does not resolve the final Ant/HalfCheetah
  conflict. HalfCheetah benefits from a high-capacity conditioned policy, while
  Ant needs a much more conservative correction and still terminates on most
  low-gravity tasks. No V86 setting passes the predeclared mechanism gate.
- The legacy bus path is a separate hard constraint. Its RE-SAC-derived
  regularization sign must remain unchanged; the apparently counterintuitive
  sign is intentional and was previously required for convergence. MuJoCo-only
  changes cannot replace the bus implementation without bus smoke and
  multi-seed regression tests.

## Current four-environment evidence

The latest valid rows use different evidence tiers and must not be read as one
homogeneous BAPR benchmark. The complete interpretation is in
`reports/BAPR_FOUR_ENVIRONMENT_EVIDENCE_2026-09-12.md`.

| Environment | Evidence tier | Adaptive / privileged switch | Robust switch | Gain | Termination | Decision |
|---|---|---:|---:|---:|---:|---|
| HalfCheetah-v2 | V21 learned BAPR v5, frozen 5-seed confirmation | 3197.3 | 1448.8 | +120.7% | 0.0% / 0.0% | PASS |
| Ant-v2 | V26 safe true-mode joint controller, 3-seed development | 3117.2 | 2598.1 | +19.5% | 6.7% / 13.3% | FAIL (1/3 seed gates) |
| Hopper-v2 | V27 true-mode oracle headroom, 3-seed screen | 2463.9 | 2424.0 | +1.6% | 100% / 100% | FAIL |
| Walker2d-v2 | V27 true-mode oracle headroom, 3-seed screen | 2141.0 | 2186.7 | -2.1% | 100% / 100% | FAIL |

Only the HalfCheetah row is direct BAPR performance. Ant measures privileged
controller capacity; Hopper and Walker2d measure whether a survival-valid
adaptation opportunity exists. Current evidence therefore supports a strong
HalfCheetah claim, not universal four-environment superiority.

## Legacy version history

| Version | Main change | Best valid evidence | Decision |
|---|---|---|---|
| v45 | Mean ensemble objective, `weight_reg=0.003`, `beta_ood=0.003`, recent replay floor `0.03`, recovery latch | Best complete 4-env, 3-seed legacy reference | Retain as historical anchor only |
| v73c | Ant latch with no controller reg and stronger recent replay | Ant seed0 tail20 `1467.2` | Seed0 probe only |
| v75r_c | `weight_reg=0.01`, recent12, actor multiplier `0.25` | Ant seed0 tail20 `3001.6` | Rejected by clean multi-seed validation |
| v76 | Clean v75r_c validation | Ant 5-seed `713.0 +- 1633.7`; other envs below v45 | Reject; recovery became permanent undertraining |
| v78c | Soft-release controller | Ant 5-seed `2010.6 +- 605.9` | Narrow Ant gain; weak transfer |
| v79b | Explicit exit/cooldown | Ant 5-seed `2220.5 +- 887.5`; HC canary `14994.3` | Best v79, still below strong baselines |
| v80 | Conservative residual gate | Best Ant about `1760`; gate nearly constant | Reject; not genuinely adaptive |
| v81 | Repair RMDM task count and probe context health | Embedding variance remained near zero; 40-task centroid accuracy about `0.024-0.028` | Stop tuning legacy context/gates |

The old node007 runs `v74c2`, `v74d2`, `v75r_a`, and `v75r_b` used stale staged
code and are not valid latest-controller evidence.

## Protocol corrections

The corrected experimental path enforces all of the following:

1. MuJoCo physics audits verify that gravity, body mass/inertia, and damping
   really change the simulator fields used by rollout.
2. Saved training and test tasks are disjoint and evaluated separately.
3. Stationary ID, stationary OOD, and nonstationary switching are reported as
   separate protocols.
4. Strict-horizon switching continues the mode clock and causal adaptation
   state across physics terminations while resetting only simulator state.
5. The switching pair is the maximally separated saved task pair. Its normalized
   latent span must exceed `0.5`; the corrected sweeps use span `1.851`.
6. Completion is established from the final recorded iteration/output, not only
   scheduler terminal status.

The earlier switching stream used task indices `0 -> 1 -> 2` over 1000 steps
with a 500-step period. Tasks 0 and 1 happened to have almost identical gravity,
and task 2 entered only at the end. Those switching numbers are invalid; the
stationary 40-train/40-test sweeps remain usable.

## Stochasticity audit and revised benchmark hypothesis

The current corrected MuJoCo experiments are not bus-like stochastic regimes.
V87 samples a finite set of gravity systems, holds one Brax `System` fixed for
500 environment steps, and then switches to another fixed system. Conditional
on simulator state and action, rollout is deterministic. The only incidental
randomness is stochastic training-policy sampling, small reset-state noise, and
the initial task draw; deterministic evaluation has no per-step process,
action, observation, or reward noise. The older `discrete_mode` environment has
random dwell times and next modes, but each selected mode is still a fixed,
deterministic gravity/mass/damping system.

The bus environment has a materially different uncertainty structure. Route
speed is resampled during the episode (every 300 simulated seconds in the
current config), passenger arrivals are Poisson samples every 20 seconds, and a
hidden regime changes speed mean, speed variance, speed cap, and passenger
demand rates. Thus a bus mode defines a transition distribution, not one fixed
dynamics function. Reward randomness is induced by those stochastic transitions
and coupled bus/passenger trajectories rather than by arbitrary reward noise.

The retained seed0 `mildfix` final sweeps provide the only direct, same-protocol
legacy comparison between the exact v45-style flag profile and ESCP. These are
development diagnostics, not final estimates:

| Environment | v45-profile OOD / switch | ESCP OOD / switch | BAPR relative to ESCP |
|---|---:|---:|---:|
| Ant | `298.7 / 99.7` | `1582.5 / 271.5` | `-81.1% / -63.3%` |
| HalfCheetah | `2273.2 / 2297.9` | `2647.8 / 2377.3` | `-14.1% / -3.3%` |
| Hopper | `1083.0 / 174.0` | `755.1 / 980.2` | `+43.4% / -82.2%` |
| Walker2d | `672.8 / 414.6` | `282.3 / 530.6` | `+138.4% / -21.9%` |

This table does not measure V87 against ESCP. V87 uses a different task salt,
switch-matched 500-step training, a third reserved stream, and a new
teacher/student controller. No ESCP checkpoint has been trained on that exact
protocol, so quoting an exact V87-versus-ESCP percentage would be invalid.

The current V2 transition model predicts one point estimate of normalized
`delta_state` and reward and trains it with squared error. Its online mismatch
score is raw prediction MSE, and a single large residual can reset or suppress
context when fallback is enabled. It predicts no conditional variance and
therefore cannot distinguish an unlikely transition under a low-variance mode
from an ordinary transition under a high-variance mode. V87 itself contains
almost no process noise, so this ambiguity does not explain V87's controlled
failures; it does explain why simply adding bus-like noise to the existing
point-residual detector would be unsound.

The next algorithm hypothesis is therefore not another V87 scalar adjustment.
It is a probabilistic regime model that separates within-regime aleatoric
variance from uncertainty about the regime:

1. Keep physical morphology and mean dynamics fixed during a persistent regime;
   do not resample link lengths, body masses, or gravity independently every
   step.
2. Add mode-conditioned transition noise through actuator gain/noise or external
   disturbances. Start without observation noise so the benchmark remains a
   stochastic MDP rather than adding a separate POMDP confound.
3. Predict both conditional mean and diagonal log-variance of
   `(delta_state, reward)` with an ensemble. Mean predicted variance is the
   aleatoric component; disagreement between predicted means is the epistemic
   component.
4. Accumulate standardized log-likelihood evidence in a sticky semi-Markov mode
   filter. A noisy sample must not by itself become a changepoint.
5. Condition the residual policy on the mode posterior, but retain the frozen
   robust actor whenever posterior confidence or paired adaptive advantage is
   insufficient.

The first controlled screen must cross environment mechanism and algorithm,
not tune only on a favorable stochastic task. For Ant and HalfCheetah seed0,
use three frozen regimes: deterministic mean shift, variance-only shift, and
mean-plus-variance shift. Compare robust SAC, RE-SAC, ESCP, point-residual BAPR,
variance-aware BAPR, and an oracle-mode upper bound (`2 x 3 x 6 = 36` training
runs). Promotion requires oracle headroom, low false-switch rate in the
stationary-noise control, and learned recovery of the oracle gain. Standard
deterministic MuJoCo remains a reported negative/control benchmark rather than
being removed after observing unfavorable results.

### BAPR-v3 implementation and submitted screen

The stochastic-regime hypothesis is implemented as a new `bapr_v3` path; the
legacy BAPR/V2 and bus paths remain available for exact historical comparison.
Each hidden mode holds gravity and actuator gain fixed for 500 steps and samples
only actuator/process noise at each transition. The context model predicts an
ensemble of mode-conditional means and diagonal variances for normalized
`(delta_state, reward)`, reports aleatoric and epistemic components separately,
and updates a sticky mode posterior from accumulated likelihood evidence. Low
posterior confidence closes the context gate. The learned deployment path also
uses the explicitly enabled conservative Q-advantage gate, so an adaptive
action that does not beat the frozen robust action falls back at execution.
The oracle teacher remains ungated while establishing headroom.

The environment audit passes for all three families. In `variance_only`, every
mode keeps gravity exactly at `-9.81` and actuator gain at `1.0`, while empirical
per-action standard deviations are `0.0000`, `0.0500`, `0.1199`, and `0.2199`
for targets `0.00`, `0.05`, `0.12`, and `0.22`. `deterministic_mean` has zero
per-step noise, and `mean_variance` changes both mean physics and transition
variance. All algorithms use the same 500-step chunk clock and preserve the
simulator state across switches. Focused regression passes are `27/27` for V2
and `10/10` for V3, including the positive bus regularization-shift sign,
checkpoint-compatible rollout, probabilistic losses, and deployment fallback.

The complete seed0 screen was bulk-submitted through scheduler only on
2026-07-11. No task has `require_node`, `preferred_node`, or `allowed_nodes`:

| Family | SAC / RE-SAC / ESCP | point / probabilistic / oracle BAPR-v3 |
|---|---|---|
| `deterministic_mean` | `t29063-t29068` | `t29069-t29074` |
| `variance_only` | `t29075-t29080` | `t29081-t29086` |
| `mean_variance` | `t29087-t29092` | `t29093-t29098` |

The first dispatch reached every available GPU node and started 27 tasks with
zero launch failures; nine remained safely queued for the next capacity wave.
Submission used one `submit-jsonl` transaction and targeted bulk dispatch, and
excluded historical eval bundles from cwd staging. Every task saves resumable
checkpoints every 50 iterations and syncs compact logs on completion.

This is a mechanism-identification screen, not a final OOD benchmark. The four
named modes are known during training; therefore the existing log key
`eval_stationary_ood` means same-family static test here and must not be called
unseen OOD in the paper. Held-out noise levels, gains, dwell lengths, and mode
combinations are required only after a seed0 mechanism passes these fixed gates:

1. Oracle mode must improve switching return over SAC and the frozen robust
   branch by at least 10%; otherwise that family has no useful adaptation
   headroom and BAPR is not tuned on it.
2. On `variance_only`, probabilistic BAPR must beat point-likelihood BAPR and
   recover at least 70% of positive oracle gain. This is the direct test of the
   aleatoric/epistemic hypothesis.
3. Probabilistic BAPR must retain at least 95% of SAC/robust static return and
   must not increase termination risk; an always-open adaptive policy fails.
4. On `deterministic_mean`, probabilistic modeling may match point inference but
   must not create extra false switches or reduce return. This is the negative
   control for needless variance modeling.
5. Only a method passing both Ant and HalfCheetah may proceed to untouched mode
   parameters and seeds 0-4. Bus multi-seed smoke remains a separate mandatory
   non-regression before any paper claim.

### BAPR-v3 stochastic screen result (2026-07-12)

All `36/36` tasks reached iteration `1399` (`5.6M` environment steps), and
every run contains 70 finite strict-horizon static and switching evaluations.
No run resumed an older checkpoint, and protocol signatures confirm the
requested family, fixed 500-step dwell, algorithm, environment, and seed. The
table reports the mean of the last five evaluations as `static / switching`.
The static ID and the legacy `stationary_ood` arrays are identical by design in
this first known-mode mechanism screen; these values are not unseen OOD.

| Family | Env | SAC | ESCP | BAPR point | BAPR probabilistic | BAPR oracle |
|---|---|---:|---:|---:|---:|---:|
| deterministic mean | Ant | `3505 / 3905` | `2511 / 3157` | `1603 / 2436` | `1038 / 1140` | `1939 / 2352` |
| deterministic mean | HalfCheetah | `2516 / 2593` | `2978 / 3079` | `2370 / 2164` | `1535 / 1609` | **`3152 / 3312`** |
| variance only | Ant | `3543 / 3394` | `1850 / 2806` | `1756 / 2780` | `1106 / 2744` | `1138 / 2225` |
| variance only | HalfCheetah | `2802 / 2752` | `3146 / 3103` | **`3814 / 3762`** | `2255 / 2185` | `2983 / 2855` |
| mean plus variance | Ant | `2333 / 2607` | `1497 / 2724` | **`2634 / 2972`** | `1637 / 2614` | `1750 / 2537` |
| mean plus variance | HalfCheetah | `2235 / 2140` | **`2529 / 2570`** | `2195 / 2124` | `1603 / 1379` | `2309 / 2141` |

The bold point-BAPR scores are not evidence of adaptation. Its final evaluation
context gate is only `0.0012` on variance-only HalfCheetah and `0.0260` on
mean-plus-variance Ant; policy adaptation strength is `0.00023` and `0.00297`.
Variance-only HalfCheetah was already `3809 / 3742` at iteration 580, before
the oracle teacher began, versus `3814 / 3762` at the end. These runs are
strong frozen robust-base trajectories with an effectively closed adaptive
branch.

The probabilistic hypothesis fails its direct test. On variance-only Ant it is
`37.0%` worse than point BAPR statically and `1.3%` worse in switching; on
variance-only HalfCheetah it is `40.9% / 41.9%` worse. Learned aleatoric variance
converges to `0.898-0.995` even in the zero-process-noise deterministic control,
instead of tracking the injected `0.00-0.22` actuator-noise scale. This inflated
variance makes transition likelihoods uninformative (`posterior_max` only
`0.45-0.84`) while allowing the confidence gate to open (`0.20-0.85`). The
model therefore explains mismatch as high noise, then executes poorly
identified adaptive actions. Point likelihood has the opposite calibration
failure: fixed variance is too small, predictive NLL remains `5.3-12.6`, and
the surprise term closes adaptation almost everywhere.

There is one genuine oracle-headroom case. Deterministic-mean HalfCheetah oracle
is `+25.3% / +27.7%` over SAC and `+5.9% / +7.6%` over ESCP, with paired
normalized teacher gain `+0.150` and no paired termination-risk increase. The
learned point and probabilistic branches are nevertheless below SAC, so they
recover none of this usable headroom. The other five environment/family pairs
do not show the required 10% oracle advantage over SAC in both protocols.

RE-SAC with `weight_reg=beta_ood=0.01` is numerically unstable in all six runs:
final Q means reach `13.7k-27.6k`, versus roughly `114-315` for stable
baselines, and returns collapse. This does not reverse the proven positive
regularization sign: the retained PyTorch implementation also uses
`+ weight_reg * reg_norm`. It instead exposes an unresolved scale mismatch in
the current JAX port, which uses a `3e-4` optimizer rate versus `1e-5` in the
original bus implementation and lacks an equivalent normalization/controller
for the growing L1 norm. These RE-SAC rows are failed implementation-scale
controls, not valid evidence that RE-SAC is intrinsically worse than SAC.

**Decision:** every learned BAPR-v3 candidate fails the preregistered promotion
gate. No additional checkpoint PKL was pulled. Four checkpoint directories
already present locally were produced directly by tasks that ran on `local`;
they were not remote downloads. The only justified next algorithm change is to
calibrate variance rather than tune another gate: predict residual variance
relative to a frozen mean model, constrain/log-prior the variance to empirical
mode residuals, and train the posterior on multi-step likelihood ratios. The
deterministic-mean HalfCheetah oracle case is the fixed positive control for
that repair; variance-only HalfCheetah point is a robust-base control, not a
candidate to promote.

### Calibrated-variance repair and validation plan

The variance repair is a separate, explicitly selected path; the failed
`legacy_state` log-variance decoder remains the default for exact reproduction.
The new `mode_calibrated` path makes four structural changes:

1. It learns one bounded diagonal residual variance per hidden mode and output
   dimension instead of predicting an unconstrained variance from each state.
2. The transition mean is trained with explicit MSE. Variance is trained by a
   Gaussian calibration loss on stop-gradient squared residuals, so increasing
   variance cannot weaken the mean-model gradient.
3. Posterior CE, temporal loss, and mode-classification loss see a
   stop-gradient copy of variance. They can improve mode means but cannot make
   classification easier by inflating variance. Only the calibration objective
   updates variance.
4. The online sticky posterior uses evidence scale `1` with per-transition
   likelihood-ratio clipping at `4`, then accumulates evidence over time. The
   previous evidence scale `4` remains available only in the legacy path.

Every calibrated run also logs same-checkpoint `robust`, unrestricted `oracle`,
and deployed `learned` static/switching returns. This removes cross-run policy
initialization and hardware nondeterminism from the headroom/recovery test and
allows screening before any remote PKL transfer. Per-mode learned variance,
variance range/spread, posterior entropy, context gate, advantage acceptance,
and adaptation strength are saved as lightweight arrays.

Focused validation passes `14/14` BAPR-v3 tests and the unchanged `27/27` V2
suite. A synthetic equal-input experiment learns low/high residual variances of
approximately `0.003/0.051` without reaching the configured `0.20` ceiling. A
full CPU train/eval smoke writes all three context ladders and per-mode variance
metrics, then restores the calibrated parameter and optimizer pytrees exactly
from next iteration `2` and continues at iteration `2`.

The fixed validation matrix is two variance ceilings (`0.05`, `0.20`) by three
families by Ant/HalfCheetah: 12 seed0 scheduler tasks. Existing SAC/ESCP results
are not retrained, because promotion is decided primarily within each new
checkpoint:

1. oracle must exceed robust by at least 10% in both static and switching;
2. learned must retain at least 95% of robust static return and recover at least
   70% of each positive oracle gain;
3. learned termination safety must not regress when final selected checkpoints
   are evaluated; this expensive check occurs only after lightweight screening;
4. variance-only learned mode variances must separate, while deterministic-mean
   variance must not inflate toward its ceiling;
5. both Ant and HalfCheetah must pass before any PKL pull, untouched-mode test,
   or multi-seed expansion.

### Calibrated result and empirical-variance repair (2026-07-14)

All 12 calibrated tasks (`t29831-t29842`) completed iteration `1399` and synced
their lightweight logs. The bounded NLL repair prevented the old variance
explosion, but it did not identify the stochastic modes: final per-mode means
are only `0.022-0.023` for the `0.05` ceiling and `0.024-0.027` for the `0.20`
ceiling, with negligible within-run mode spread. Variance-only modes therefore
remain almost indistinguishable despite having different process noise.

The environment and fused rollout are not the source of that failure. A
fixed-state repeated one-step audit measured mean target variances of
`0/.00747/.03620/.09878` across Ant modes and
`0/.03042/.08512/.16005` across HalfCheetah modes. The action disturbance is
executed inside the fused adaptive rollout, and task IDs remain fixed for each
500-step dwell. The missing separation is a variance-estimation problem.

The calibrated screen also exposed teacher forgetting. HalfCheetah oracle
static return is strong near the end of teacher training, then often collapses
after the deployment branch resumes policy/critic updates. Representative
`iter 980 / 1180 / 1380` oracle trajectories are:

| Run | 980 | 1180 | 1380 |
|---|---:|---:|---:|
| `cal05 deterministic_mean HC` | `3048` | `3195` | `1571` |
| `cal05 mean_variance HC` | `2646` | `3032` | `-251` |
| `cal05 variance_only HC` | `2702` | `2524` | `745` |
| `cal20 mean_variance HC` | `2176` | `2321` | `803` |

Thus the final oracle ladder was no longer a valid upper bound: the same
conditioned policy and critic used as the teacher had been overwritten during
learned-context deployment. This confounded inference quality with controller
forgetting and explains why several final learned policies appeared to match a
robust base while their final oracle branch was unusable.

The next repair is deliberately structural rather than another gate sweep:

1. `mode_empirical` trains only conditional means with the optimizer. It
   computes residual second moments around the ensemble mean, aggregates them
   by true mode/output over whole supervised chunks, and updates bounded
   aleatoric variances by a closed-form EMA.
2. Per-transition mode-classifier CE can be set to zero. The sticky multi-step
   posterior CE remains, preventing individual stochastic samples from being
   treated as mode labels.
3. `freeze_teacher_after_teacher` freezes the conditioned policy, critic, and
   policy gate in deployment while the causal context model and variance EMA
   continue learning. Final oracle evaluation now refers to the actual frozen
   teacher.
4. Lightweight logs now include empirical target variance/count per mode,
   log-space calibration error, posterior accuracy/true-mode probability, and
   online true-mode probability/correctness.

Focused validation passes all `16/16` V3 tests. A full Brax training smoke with
optimizer updates shows empirical-update count `0 -> 4`; deployment logs show
`train_residual=train_critic=train_gate=0` while context variance continues to
update. A separate checkpoint smoke restores policy, critic, context, all
optimizer pytrees, replay size, `iter=5`, and `steps=80`, then continues from
that exact iteration. One legacy V2 direct-warmstart test differs by
`2.3e-4` on GPU because zero-padded and unpadded GEMMs round differently; the
gated-direct warmstart used here and all V3 tests pass.

The new fixed seed0 matrix is `2 variants x 3 families x 2 environments = 12`
scheduler tasks:

| Variant | Variance update | Instant classifier | Post-teacher controller |
|---|---|---:|---|
| `nll_freeze` | bounded NLL control | `0` | frozen |
| `emp05_freeze` | residual-moment EMA `0.05` | `0` | frozen |

Both use the same `0.5` variance ceiling, stage schedule, seed, stochastic
physics, and same-checkpoint robust/oracle/learned ladder. This isolates the
empirical estimator from teacher preservation. Lightweight screening comes
first; remote checkpoint PKLs remain untouched until a run demonstrates
preserved oracle headroom, mode-ordered variance on variance-only tasks,
above-chance posterior inference, and learned recovery of useful oracle gain.

The matrix was submitted through scheduler only as `t33157-t33168`. All tasks
are unpinned (`require_node=null`, `allowed_nodes=null`) and reached `running`
across local, jtl311linux, both jtl110 GPU servers, and node007. The final task
uses the scheduler's task-local over-one-third permission to share one 11 GB
GPU; no scheduler policy or global packing rule was changed. Startup logs for
both variants confirm the requested variance model, zero instantaneous
classifier weight, frozen teacher, disabled advantage/safety gates, and full
same-checkpoint context ladder.

### Empirical/frozen-teacher result (2026-07-15)

All 12 tasks completed iteration `1399` and `5.6M` steps with complete finite
lightweight logs. No remote checkpoint PKL was downloaded. The teacher freeze
is correct: all deployment controller update flags are zero, and every static
oracle curve is exactly constant after iteration 1000.

The empirical estimator tracks its residual target, but the target is total
model residual rather than aleatoric variance. Learned mode variances are
approximately `.20-.29` in every family. On variance-only tasks they are nearly
identical across the four true noise levels, and posterior accuracy is
`.305` on Ant and `.254` on HalfCheetah, effectively chance. Bounded NLL remains
under-dispersed at `.025-.029`, incurs calibration error `1.65-1.93`, and
almost closes the HalfCheetah context gate.

No new BAPR row beats both SAC and ESCP in both static and switching. The
closest case is empirical deterministic-mean HalfCheetah (`3175 / 2931`)
versus ESCP (`2978 / 3079`): static improves, switching remains `4.8%` lower.
NLL mean-plus-variance Ant reaches switching `3009` but its static `2142`
remains below SAC `2333`. Multi-seed promotion and PKL transfer are therefore
blocked.

The next experiment is oracle-first environment validation, not another
variance scalar sweep. A bus-like stochastic MuJoCo family should keep
morphology fixed but draw per-step exogenous packet-loss or burst-load events
from a persistent regime. Learned inference is justified only after a
privileged oracle shows at least 10% robust-relative headroom in both protocols
on Ant and HalfCheetah. Full tables and diagnostics are in
`reports/bapr_v3_empirical_freeze_2026-07-15.md`.

## BAPR-v2 design

BAPR-v2 is implemented independently from the frozen legacy `bapr.py` path:

- robust Gaussian base actor that works with zero context;
- bounded adaptive residual, `pi = pi_base + gate * clip(delta_pi)`;
- causal GRU-style encoder over recent `(s, a, r, delta_s, done)` transitions;
- single point transition/reward predictor for mismatch and confidence (the
  critic is ensembled, but this decoder is not probabilistic);
- identical `reset_adaptation()` and `observe_transition()` semantics in train
  and evaluation;
- separate replay storage for current and next causal contexts;
- no raw MuJoCo IPM shift by default;
- oracle, supervised, hybrid, and robust context modes forming an explicit
  mechanism ladder.

The v82 seed0 screen contained 36 runs: nine variants across Ant,
HalfCheetah, Hopper, and Walker2d. The v83 capacity screen adds corrected oracle
normalization, a direct conditioned actor, a conditioned actor with separately
trained fallback, five routed experts, and a larger residual radius.

## Corrected final-sweep evidence

The following numbers combine the valid stationary 40-task sweeps with the
contrastive fixed 1000-step switching stream. They are seed0 diagnostic results,
not final paper estimates.

| Environment | Robust OOD / switch | Best complete oracle OOD / switch | Best complete learned OOD / switch | Current diagnosis |
|---|---:|---:|---:|---|
| Ant | `1115.3 / 1374.4` | `1946.9 / 1874.0` (`+74.6% / +36.3%`) | `1165.8 / 1843.5` (`+4.5% / +34.1%`) | Real headroom; learned policy recovers switching but not broad OOD gain |
| HalfCheetah | `1941.7 / 1568.6` | `3117.6 / 2609.0` (`+60.6% / +66.3%`) | `2019.1 / 2078.3` (`+4.0% / +32.5%`) | Real headroom; learned policy captures only a small part of it |
| Hopper | `1072.1 / 3454.1` | `458.8 / 3006.6` (`-57.2% / -13.0%`) | `516.5 / 3262.3` (`-51.8% / -5.6%`) | Robust policy dominates every tested adaptation family |
| Walker2d | `375.7 / 2540.4` | `448.7 / 1845.8` (`+19.4% / -27.3%`) | `314.5 / 2263.3` (`-16.3% / -10.9%`) | Oracle improves stationary OOD but damages switching |

Learned latent/true-gravity correlations on the corrected switching stream are
nontrivial: about `0.58-0.91` for the strongest available runs. However,
detector AUC values remain roughly `0.42-0.68`, far below the required `0.8`.
The selected learned gates are effectively always on (`~0.97` mean for the
no-fallback runs), and prediction error does not spike at a true switch. This
separates two failures: the encoder often identifies gravity, but policy
conditioning and the mismatch-trigger signal do not convert it into reliable
control improvement.

Matched-task diagnostics make the policy failure explicit. The best oracle
beats the robust base on all `40/40` OOD tasks in Ant and HalfCheetah. In
contrast, it wins only `5/40` Hopper tasks. Walker2d's direct oracle wins
`25/40` stationary tasks but loses `27.3%` on the switching stream. The best
learned policy wins only `23/40`, `18/40`, `7/40`, and `12/40` tasks in Ant,
HalfCheetah, Hopper, and Walker2d, respectively.

Predeclared promotion gates:

- oracle improves both OOD mean and switching by at least 10% in 3/4 envs;
- learned context recovers at least 70% of each positive oracle gain;
- stationary performance retains at least 95% of robust base;
- switch AUC at least 0.8 and median delay below 50 steps;
- bus effectiveness preserved.

At the latest audit, `51/55` eligible runs have paired stationary and switching
results. The four unevaluated checkpoints are all additional HalfCheetah v83
capacity variants, so they cannot increase the number of environments passing
the oracle gate. The oracle gate passes in `2/4` environments, learned recovery
passes in `0/4`, and detector AUC passes in `0/4`. The broad-algorithm verdict
is therefore negative even before those four redundant capacity evaluations.

## Technical conclusion and next direction

The failure is not simply "BAPR cannot infer the environment." The causal
encoder can infer gravity. The larger problem is that the same task-conditioned
policy must learn robust control and exploit context without destroying the
fallback behavior. V82 also stores encoder outputs in replay while the encoder
continues to change, so policy updates see stale and drifting context
coordinates. On Hopper and Walker2d even true oracle context fails the joint
gate, which additionally exposes policy interference and limited benchmark
headroom.

The next defensible design is an oracle-teacher/residual-student path, restricted
first to the positive Ant and HalfCheetah mechanism cases:

1. Train a robust base to convergence and freeze or slowly update it.
2. Train actor and critic with the stable true-task coordinate reconstructed
   from replay `task_id`, rather than stale encoder outputs. The
   `bapr_v2_policy_context_source=oracle_task` plumbing for this v84 experiment
   is implemented and unit-tested; the first seed0 validation is complete.
3. Train the causal encoder against that fixed teacher coordinate, then replace
   oracle context with the online estimate at evaluation. Freeze or slowly
   update the robust base and constrain the residual to zero whenever its
   estimated per-task advantage is non-positive.
4. Trigger adaptation from a dynamics-residual detector trained directly on
   switch labels/prediction error, not critic Q-standard-deviation.
5. Validate Ant and HalfCheetah first. Treat Hopper and Walker2d as
   no-headroom/robust-policy cases unless a separately trained oracle family
   clears the joint OOD/switching gate.

Do not claim a unified four-environment improvement from v82/v83. If v84 cannot
recover at least `70%` of the oracle gain on both Ant and HalfCheetah without
hurting the robust fallback, the scientifically supportable route is an
analysis paper about when online adaptation has exploitable headroom, with Ant
and bus as positive cases and robust-policy-dominated tasks as explicit
negative cases.

## V84 teacher-student validation

V84 implements the full staged mechanism rather than only the earlier replay
context fix:

- iterations `0-599`: context-free robust pretraining; only base actor and
  critic update;
- iterations `600-999`: oracle teacher; the base actor is frozen, actor/critic
  updates reconstruct stable context from replay `task_id`, and only the
  residual actor is trainable;
- iterations `1000-1199`: student distillation; actor, critic, target critic,
  entropy temperature, and robust base are frozen, while only the causal
  transition encoder updates;
- teacher rollout uses true task context, student rollout uses only causal
  history, and both stationary and switching evaluation use the same learned
  context path;
- deployment selects the residual action only when the ensemble lower
  confidence bound of `Q(s,a_adapt,z)-Q(s,a_base,z)` is positive; otherwise it
  executes the frozen robust action.

Local validation passed `19/19` focused tests, an actual three-stage optimizer
smoke (base-only, residual-only, then controller-frozen), and compiled Brax
training/evaluation rollouts with the advantage gate. A source-level regression
test also locks the legacy bus expressions to the proven positive
`+ weight_reg * reg_norm` shift.

Scheduler-only seed0 screen submitted on 2026-07-10:

| Task | Environment | Residual radius | Initial node/GPU |
|---|---|---:|---|
| `t24881` | Ant | `0.25` | `jtl311linux:0` |
| `t24882` | HalfCheetah | `0.25` | `jtl311linux:1` |
| `t24883` | Ant | `1.0` | `node007:0` |
| `t24884` | HalfCheetah | `1.0` | `node007:1` |
| `t24885` | legacy bus sign smoke (complete) | `n/a` | `node007:2` |

No task is node-bound; scheduler checkpoint migration and resume remain active.
The MuJoCo promotion criterion is unchanged: learned v84 must recover at least
`70%` of the positive oracle gain on both Ant and HalfCheetah, retain at least
`95%` of robust stationary performance, and avoid a switching regression. The
bus path must complete without changing its regularization sign before any
multi-seed expansion.

The bus smoke completed both requested episodes and wrote all four final model
artifacts. Its regularization monitor remained finite and nonzero (two critic
branches approximately `9.41` and `9.47`). Scheduler initially classified the
normal exit as failed because the legacy script did not emit a success marker;
the final artifacts and episode log were verified, the record was corrected to
done, and the submit command now appends `DONE` for future runs. All four v84
training tasks reached `iter 1199` and saved resumable final checkpoints. Their
wall times were approximately `2.0 h`, `3.1 h`, `1.8 h`, and `2.8 h` for
`t24881-t24884`, respectively.

### V84 final-sweep result

The corrected final sweep evaluated all 40 train tasks, all 40 held-out tasks
with three episodes per task, and five fixed-horizon switching streams. CPU
tasks `t27159-t27162` all completed and produced the full task and per-step
trace files.

| Variant | Environment | ID | OOD | Switching | OOD oracle-gain recovery | Switching oracle-gain recovery |
|---|---|---:|---:|---:|---:|---:|
| V84a, residual radius `0.25` | Ant | `1550.1` | `1466.9` | `1877.0` | `42.3%` | `100.6%` |
| V84a, residual radius `0.25` | HalfCheetah | `2397.2` | `2377.8` | `2219.9` | `37.1%` | `62.6%` |
| V84b, residual radius `1.0` | Ant | `1414.4` | `1337.0` | `1800.4` | `26.7%` | `85.3%` |
| V84b, residual radius `1.0` | HalfCheetah | `1744.2` | `1846.5` | `1832.2` | `-8.1%` | `25.3%` |

Recovery is measured as `(V84 - robust)/(oracle - robust)` using the retained
corrected baselines. V84a is the strongest learned-context result so far. It
improves over the previous best learned result by `25.8% / 1.8%` on Ant OOD /
switching and by `17.8% / 6.8%` on HalfCheetah. Relative to the robust base, its
gains are `31.5% / 36.6%` on Ant and `22.5% / 41.5%` on HalfCheetah. This is a
real mechanism improvement, but it does not pass the predeclared promotion
gate because OOD oracle-gain recovery is below `70%` in both environments and
HalfCheetah switching recovery is `62.6%`.

The radius ablation is decisive on HalfCheetah: radius `0.25` wins `37/40`
matched held-out tasks over radius `1.0`, with a mean return increase of
`531.2`. The larger residual is therefore not a useful capacity increase; it
overrides the robust controller too aggressively. Ant is less uniform: radius
`0.25` wins only `19/40` matched tasks, although its mean is `129.9` higher.

The student representation itself is no longer the primary bottleneck. On the
switching streams, learned/true latent correlation is `0.967` for Ant and
`0.945` for HalfCheetah. The decision gate remains weak: switch AUC is only
`0.550` and `0.476`, and residual activation is almost unchanged from the 50
steps before a switch to the 50 steps after it (`0.780 -> 0.753` for Ant and
`0.560 -> 0.553` for HalfCheetah). Thus the critic-LCB gate is selecting actions
but is not detecting mismatch.

Ant's remaining OOD loss is highly structured. In the lowest-gravity held-out
quartile its mean return is only `527.9` with `93%` termination, versus `2366.8`
and `3%` termination in the highest-gravity quartile. The encoder can identify
gravity over a long stream, but many low-gravity Ant episodes terminate before
the online estimate can become useful. This must be separated from teacher
quality before another training variant is justified.

The same-checkpoint policy ladder completed as CPU tasks `t27284-t27289`; the
deployed learned-plus-gate rows reuse `t27159-t27160`.

| Environment | Frozen base OOD / switch | Oracle teacher OOD / switch | Learned, no hard gate OOD / switch | Learned, hard gate OOD / switch |
|---|---:|---:|---:|---:|
| Ant | `1354.7 / 1681.8` | `1492.0 / 1976.5` | `1454.1 / 2048.2` | `1466.9 / 1877.0` |
| HalfCheetah | `2215.5 / 1893.9` | `2358.8 / 2230.8` | `2304.8 / 2267.9` | `2377.8 / 2219.9` |

This controlled comparison revises the cross-checkpoint diagnosis. The causal
student is already effective: without the hard gate it recovers `72.4%` of the
Ant teacher's held-out gain and `124.3%` of its switching gain; HalfCheetah
recovery is `62.3%` and `111.0%`. The hard critic-LCB gate raises held-out mean
slightly but removes `171.2` Ant switching return and `48.1` HalfCheetah
switching return. It is therefore disabled in the next design. The remaining
limitation is teacher capacity: the frozen radius-0.25 oracle residual improves
the same base by only `10.1% / 17.5%` on Ant and `6.5% / 17.8%` on HalfCheetah.

The column historically named OOD is an independently sampled held-out task
set from the same `pow1p5`, `[-3,3]` log-scale support as training. It tests
unseen task values but is not a strict out-of-support distribution shift; paper
claims must call it held-out/interpolation unless a separate support-shift
protocol is run.

V85 targets teacher capacity without returning to joint/stale-context training:

- `v85a_residual_r050`: intermediate bounded radius `0.5`;
- `v85b_direct_warm`: full conditioned actor, copied exactly from the converged
  robust actor at the robust-to-teacher boundary;
- `v85c_direct_warm_fast`: the same actor with burn-in `4` and confidence time
  constant `16` to address short low-gravity Ant episodes.

All variants retain the frozen robust fallback, stable oracle replay coordinate,
student-only final phase, zero MuJoCo raw regularizer, and no hard Q gate. The
warm-start flag is checkpointed so resume cannot overwrite a trained teacher.
Focused tests pass `22/22`. Scheduler tasks `t27332-t27337` launched on
`jtl311linux` and `node007`; no node binding, Slurm, or auto-adopt is used.

### V85 staged-capacity result

All six training tasks completed at `iter 1199` and the corrected full sweeps
completed as `t28273-t28278`.

| Variant | Environment | Held-out | Switching | Change from V84 no-gate held-out / switch |
|---|---|---:|---:|---:|
| Residual radius `0.5` | Ant | `1371.5` | `1632.4` | `-5.7% / -20.3%` |
| Residual radius `0.5` | HalfCheetah | `2155.6` | `2090.0` | `-6.5% / -7.8%` |
| Warm-start direct | Ant | `1000.1` | `1893.5` | `-31.2% / -7.6%` |
| Warm-start direct | HalfCheetah | `2811.2` | `2354.6` | `+22.0% / +3.8%` |
| Warm-start direct, fast encoder | Ant | `1272.0` | `1807.7` | `-12.5% / -11.7%` |
| Warm-start direct, fast encoder | HalfCheetah | `2627.6` | `2367.0` | `+14.0% / +4.4%` |

Radius `0.5` is not the missing compromise: it loses in both environments.
The full conditioned actor is a genuine HalfCheetah improvement, winning
`35/40` matched held-out tasks for the standard encoder, but it is unsafe for
Ant and increases held-out termination to `82%`. Faster inference does not
resolve that policy-capacity conflict.

Oracle-context sweeps `t28279-t28282` isolate teacher quality:

| Checkpoint | Environment | Learned held-out / switch | Oracle held-out / switch |
|---|---|---:|---:|
| Warm-start direct | Ant | `1000.1 / 1893.5` | `1061.3 / 2025.9` |
| Warm-start direct | HalfCheetah | `2811.2 / 2354.6` | `2816.7 / 2031.7` |
| Warm-start direct, fast encoder | Ant | `1272.0 / 1807.7` | `1527.0 / 1398.2` |
| Warm-start direct, fast encoder | HalfCheetah | `2627.6 / 2367.0` | `2635.5 / 2570.4` |

For standard direct Ant, the teacher itself is poor on held-out tasks; encoder
error is not the primary cause. For HalfCheetah the student recovers essentially
all stationary teacher performance. A learned latent can exceed instantaneous
oracle task switching because gradual context movement smooths the control
transition from the previous mode's state distribution. Therefore task-oracle
performance is a stationary capacity reference, not a strict switching upper
bound.

The next architecture should learn a continuous state/task-dependent blend
between the exact robust base and the full conditioned actor. This gives one
policy the capacity used by HalfCheetah while allowing Ant to retain a small
correction. The blend must be trained end-to-end with the actor, not derived
from the failed post-hoc critic-LCB threshold.

### V86 adaptive-capacity screen

V86 adds `gated_direct`, whose policy is a differentiable blend between the
frozen robust actor and the full conditioned actor. The blend is predicted from
state and latent context and optimized directly through the SAC actor loss.
The conditioned actor is still initialized as an exact copy of the robust base,
so enabling the new branch cannot create a policy discontinuity. An optional
deterministic action-deviation penalty regularizes the blend without using the
miscalibrated deployment critic gate.

The seed0 screen compares deviation weights `0`, `1`, and `10` on Ant and
HalfCheetah. Focused tests pass `23/23`, and a compiled teacher-stage gradient
smoke verified warm-start, blend gradients, the deviation objective, and the
new adaptation-strength metric. Scheduler training tasks `t28598-t28603` all
completed at iteration `1199`. The learned-policy sweeps are `t28700-t28705`;
same-checkpoint robust/oracle ladders are `t28762-t28773`. All result files were
synchronized and passed the expected checkpoint/task/trace completeness checks.

| Variant | Environment | Robust held-out / switch | Oracle held-out / switch | Learned held-out / switch | Learned termination | Switch AUC / latent corr. |
|---|---|---:|---:|---:|---:|---:|
| deviation `0` | Ant | `1188.4 / 1776.3` | `614.1 / 1382.3` | `506.6 / 1702.8` | `89.2%` | `0.618 / 0.954` |
| deviation `0` | HalfCheetah | `1302.7 / 1286.4` | `1899.7 / 1815.3` | `1900.4 / 1688.6` | `0%` | `0.519 / 0.935` |
| deviation `1` | Ant | `1159.2 / 1525.6` | `1120.9 / 1646.6` | `970.9 / 1591.3` | `61.7%` | `0.418 / 0.948` |
| deviation `1` | HalfCheetah | `1885.7 / 1578.4` | `3582.9 / 2794.6` | `3470.6 / 2226.2` | `0%` | `0.494 / 0.937` |
| deviation `10` | Ant | `949.0 / 1673.0` | `1225.4 / 1961.6` | `992.3 / 2112.2` | `77.5%` | `0.546 / 0.958` |
| deviation `10` | HalfCheetah | `1397.4 / 1277.5` | `1489.3 / 1385.5` | `1486.3 / 1345.3` | `0%` | `0.472 / 0.952` |

The adaptive-capacity hypothesis is rejected. Deviation `1` is a real
HalfCheetah stationary improvement: learned context recovers `93%` of the
oracle held-out gain and wins `40/40` tasks over its own robust base. It
recovers only `53%` of switching gain, however, and does not transfer to Ant.
Deviation `10` is the only Ant setting whose oracle improves both metrics, but
the learned policy recovers only `16%` of its held-out gain. Its learned
held-out improvement is only `4.6%`, below the `10%` gate, with `77.5%`
termination. The unregularized setting is destructive on Ant. No one setting
provides joint positive evidence in both environments.

V86's learned gate is again almost a constant. During the student phase, mean
adaptation strength is `0.759/0.743`, `0.599/0.671`, and `0.290/0.280` for
Ant/HalfCheetah at deviation weights `0`, `1`, and `10`; the corresponding
between-task standard deviations are only `0.0018-0.0055`. The deviation
weight therefore acts as a global capacity knob rather than inducing
state/task-dependent safe adaptation. All switch AUC values remain below
`0.62` despite latent correlations above `0.93`.

The failure is especially asymmetric under termination. Across Ant's four
held-out gravity-magnitude quartiles, the first two low-gravity quartiles have
`100%` termination for deviation `0` and `10`; deviation `1` still has
`90%/80%`. HalfCheetah has no corresponding termination cliff and benefits
from the full conditioned actor. This is the concrete reason that a single
action-deviation coefficient cannot serve both environments.

## Systematic failure audit after V86

The `v1-v86` labels should not be read as 86 independent, clean algorithmic
tests. They comprise several repeated variants of the same hypotheses, and a
large early subset used protocols later shown to be defective. The useful
history is better divided into five evidence eras:

1. Early legacy BAPR explored BOCD, belief updates, ensemble regularization,
   LCB objectives, recent replay, and recovery controls. Some physics changes
   did not affect rollout, train/test task streams overlapped, and switching
   evaluation could select nearly identical gravity tasks. These runs cannot
   establish comparative performance.
2. V45 is the strongest legacy multi-environment anchor. V46-V80 largely tune
   collapse recovery, regularization, replay, release, and residual-gate
   settings around that controller. Seed0 improvements repeatedly failed
   clean multi-seed or cross-environment validation.
3. V81 audited the legacy representation and found context collapse, ending
   the case for further BOCD/controller tuning on that implementation.
4. V82-V83 rebuilt the mechanism ladder under the corrected protocol. It
   established that tested oracle adaptation has joint headroom in Ant and
   HalfCheetah, but not in Hopper and Walker2d, and that learned gating does not
   recover the available oracle gains.
5. V84-V86 separate robust pretraining, oracle teacher training, causal
   student fitting, policy capacity, and safe blending. They improve the
   diagnosis but still fail the predeclared cross-environment gate.

### Causes established by controlled evidence

1. **Adaptation headroom is not universal.** Under the tested corrected task
   family and policy classes, even true task context cannot jointly improve
   held-out and switching returns in Hopper/Walker2d. A unified four-environment
   win is therefore impossible for the current mechanism before inference is
   considered.
2. **Representation is no longer the main bottleneck.** V84-V86 switching
   latent correlations are approximately `0.93-0.97`, while switch detection
   AUC remains `0.42-0.68`. The agent often estimates gravity but does not know
   when or how strongly that estimate should change control.
3. **The control-utilization mechanism is weak.** BOCD, Q-standard-deviation,
   critic-LCB, and V86's learned blend all become nearly constant or select the
   wrong action. The V84 hard critic gate directly reduced switching return.
   V86 only turns the deviation penalty into a global adaptation-strength
   setting.
4. **Policy capacity and safety conflict by environment.** Full conditioned
   actors are useful in HalfCheetah but cause high Ant termination. Small
   residuals preserve Ant better but cap HalfCheetah. Intermediate radius
   `0.5` and learned scalar blending do not resolve the conflict.
5. **Ant's failure is a survival problem, not only a mean-return problem.** In
   low gravity, many episodes terminate before useful online correction is
   possible. An unconstrained expected-Q actor can trade these catastrophic
   failures for gains on easier tasks and still optimize its training loss.

### Causes strongly supported by code and ablations

1. **Teacher/student deployment mismatch.** The actor and blend gate update
   only during the oracle-teacher phase. They are frozen when rollout switches
   to the learned causal latent. Encoder error, latency, and transient latent
   trajectories therefore never receive a policy-level corrective update.
2. **The training curriculum does not teach the reported switching task.** A
   training rollout holds one physics system fixed for a complete `4000`-step
   scan, changes tasks every `20000` steps, and trains encoder chunks entirely
   within one task. Final evaluation changes physics every `500` steps. The
   encoder has supervised steady-task identification but no explicit
   switch/recovery target matching deployment.
3. **The objective has no reason to learn a selective gate.** V86 optimizes the
   ordinary SAC actor objective plus a global action-deviation penalty. It has
   no paired robust/adaptive return target, termination-risk constraint, or
   switch-recovery loss. A nearly constant gate is therefore a predictable
   optimum, not just an optimization accident.
4. **Critic uncertainty is not calibrated for fallback decisions.** Ensemble
   Q dispersion is affected by critic regularization and replay coverage. The
   critic has little counterfactual data for actions proposed by the adaptive
   branch, so its LCB is not a reliable safety certificate.
5. **The robust foundation is itself noisy.** V86 same-seed frozen-base
   held-out returns vary from `949-1188` on Ant and `1303-1886` on
   HalfCheetah, even before each deviation setting is applied. Chaotic RL,
   hardware/numerical nondeterminism, and a single seed make small variant
   differences untrustworthy.

### Experimental causes that amplified wasted iteration

- Early physics, task split, horizon, switching, and OOD-label defects mixed
  protocol repair with algorithm search. The current held-out set is sampled
  from the same support as training and must not be called strict OOD.
- Most late design choices were selected on seed0. V75r_c's apparent Ant win
  becoming `713.0 +- 1633.7` over five seeds is the clearest multiple-selection
  warning. Searching dozens of variants on the same tasks creates substantial
  selection bias even when each individual run is valid.
- Stale staged code, checkpoint/sync failures, false scheduler terminal states,
  grouped seeds, serialized evaluation, and disk exhaustion caused reruns and
  missing evidence. These explain cost and some invalid historical rows, but
  they do not explain V84-V86's controlled mechanism failure after those paths
  were corrected.
- Earlier SAC/ESCP/RE-SAC comparisons mixed environment families and evaluation
  definitions. They cannot prove that the papers' algorithms are intrinsically
  weaker. Final claims require one frozen paper-aligned protocol and verified
  per-run configs.

### Final scientific verdict and stop condition

BAPR is not supported as a single algorithm that reliably beats
SAC/ESCP/RE-SAC across standard nonstationary MuJoCo benchmarks. The strongest
positive mechanism result is HalfCheetah with a high-capacity conditioned
teacher/student; Ant has limited switching gains but unresolved catastrophic
termination; Hopper and Walker2d are robust-policy-dominated under the tested
families. The defensible mechanism conclusion is that **environment inference
is insufficient: safe utilization of inferred context is the bottleneck**.

Do not launch V87 as another scalar-gate, deviation-weight, regularization, or
replay sweep. A new algorithm attempt must start from a new predeclared
formulation: train the estimator on a switch-heavy curriculum matching
deployment, train the policy on learned-latent trajectories rather than only
oracle latents, and gate adaptation with an explicit paired improvement and
termination-risk objective. It should first demonstrate oracle headroom in at
least `3/4` environments on an untouched task set, then recover at least `70%`
of that gain and pass five-seed validation. Otherwise the stronger paper route
is an analysis of when online adaptation helps or fails, using bus and
HalfCheetah as positive cases and the robust-policy/termination cases as
negative evidence. The bus regularization sign remains fixed and any new
controller still requires bus smoke plus multi-seed regression.

## V87 constrained-deployment preregistration

V87 is not another scalar gate or regularization sweep. It implements the new
formulation required by the V86 failure audit:

- `0-599`: train a context-free robust base;
- `600-999`: freeze the base, warm-start a full conditioned teacher, keep its
  gate near one and frozen, and train it with privileged task context;
- at the teacher boundary, evaluate robust and teacher policies from matched
  deterministic resets on every training task and convert paired normalized
  return gain plus termination-rate gap into a fixed safe-adaptation target;
- `1000-1199`: freeze policy/critic and train the causal student on transition
  sequences containing real task-switch boundaries with per-step task targets;
- `1200-1399`: clear oracle/stale replay once, collect only learned-latent
  rollouts, and jointly update encoder, conditioned branch, critic, and gate.
  The gate is supervised by the paired safety target, while unsafe tasks also
  penalize deviation from the frozen robust action.

Training physics is now held for `500` steps at a time, matching the reported
switching dwell instead of the old `20000`-step training dwell. The 4000-step
collection batch remains continuous across the eight physics chunks. Encoder
windows deliberately include switch boundaries and no longer penalize latent
movement when the true task changes.

The task seed salt is frozen at `870000`, creating three deterministic,
disjoint streams: training, seed0 model-selection validation, and a third
reserved final-test stream. The reserved stream must not be evaluated during
the seed0 screen.

The complete seed0 matrix is fixed before results are observed:

| Variant | Paired calibration | Gate supervision | Unsafe deviation | Purpose |
|---|---:|---:|---:|---|
| `v87a_switch_deploy` | off | `0` | `0` | isolate switch-matched curriculum plus learned-latent deployment |
| `v87b_paired_safe` | 3 episodes/task | `2` | `5` | primary paired return/risk constraint |
| `v87c_paired_strict` | 3 episodes/task | `5` | `20` | test stronger catastrophic-risk fallback |

Each variant runs seed0 on Ant, HalfCheetah, Hopper, and Walker2d (`12`
training tasks). Same-checkpoint robust/oracle/learned validation sweeps are
required before promotion. The promotion gate is fixed:

1. Ant and HalfCheetah learned policies must each improve both held-out and
   switching return by at least `10%` over their own robust base.
2. Learned context must recover at least `70%` of each positive oracle gain.
3. Hopper and Walker2d must retain at least `95%` of robust performance in both
   protocols; adaptation is allowed to stay closed where oracle headroom is
   absent.
4. Switch AUC must reach `0.8`, median delay must be below `50` steps, paired
   safety targets must be nonconstant, and Ant termination must not exceed its
   same-checkpoint robust base.
5. Only one predeclared winner may expand to seeds `0-4`. Only that frozen
   multi-seed candidate may be evaluated on the reserved third task stream.

Implementation validation currently passes `27/27` focused tests, a compiled
four-stage Brax smoke, checkpoint/resume without repeated calibration or replay
reset, and separate validation/reserved evaluator smokes. The legacy bus
regularization-sign regression remains active.

The preregistered seed0 screen was submitted through scheduler only on
2026-07-11. No task is node-pinned, and no reserved-stream evaluation has been
run:

| Variant | Ant | HalfCheetah | Hopper | Walker2d |
|---|---|---|---|---|
| `v87a_switch_deploy` | `t28789` | `t28790` | `t28791` | `t28792` |
| `v87b_paired_safe` | `t28793` | `t28794` | `t28795` | `t28796` |
| `v87c_paired_strict` | `t28797` | `t28798` | `t28799` | `t28800` |

All 12 tasks completed all 1400 iterations (`5.6M` environment steps), synced
their logs, and saved final checkpoints at next iteration `1400`. The fixed
same-checkpoint validation ladder is `t28938-t28973`: 12 checkpoints times
`robust`, `oracle`, and `learned`, evaluated only on the untouched validation
stream by scheduler CPU tasks on `node001-node006`. Each mode has an isolated
read-only bundle path, and representative logs confirm that all three modes
load the final `iter=1400` checkpoint. Five-seed expansion remains blocked on
the fixed seed0 promotion gate above; the reserved stream remains unopened.

### V87 result and stop decision

The complete `36/36` same-checkpoint validation ladder finished. Five original
CPU evaluations on `node005` hit the node's pthread limit; scheduler retried
the same signatures as `t28974-t28978` on `node004`. Every accepted result
loads next iteration `1400`, reports `heldout_task_stream=validation`, and has
the requested `robust`, `oracle`, or `learned` context source. This is an
execution retry only and does not change the evaluation matrix or RNG.

| Variant | Oracle headroom in 4 envs | Ant and HC +10% | Oracle recovery | Hopper/Walker retain 95% | Detector | Safe targets | Ant risk | Promote |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v87a_switch_deploy` | `0/4` | fail | pass | fail | fail | off/fail | fail | **no** |
| `v87b_paired_safe` | `2/4` | fail | fail | fail | fail | pass | pass | **no** |
| `v87c_paired_strict` | `1/4` | fail | pass | fail | fail | pass | fail | **no** |

The most important paired comparisons are:

- `v87b` Ant is a real positive case. Against its own frozen robust base,
  oracle gains are `+71.4%` held-out and `+21.2%` switching; learned context
  recovers `+66.6%` and `+15.2%`, while stationary termination falls from
  `0.858` to `0.600`.
- `v87c` Ant also improves (`+12.8%` held-out, `+33.5%` switching), but its
  stationary termination rate increases from `0.758` to `0.792`, so it fails
  the risk gate.
- HalfCheetah has no useful oracle headroom in the paired variants: `v87b` is
  only `+0.2%/+0.5%`, while `v87c` is negative. Learned adaptation therefore
  cannot satisfy the required gain and correctly has little useful work to do.
- Hopper exposes control-transfer failure. `v87b` oracle is strong
  (`+22.3%/+59.8%`), but learned context gives `-17.7%/+33.4%`; the estimator
  finds the task, but the learned controller/gate does not recover the oracle
  stationary advantage.
- Walker2d has no switching adaptation headroom: every oracle branch loses
  `43-89%` versus its same-checkpoint robust base, and learned branches lose
  `40-54%`. This is not an inference failure.

Learned switching latent correlations are high (`0.769-0.980`, and above
`0.93` except Walker2d), while switch-error AUC is only `0.419-0.667`. The
paired targets are nonconstant and do reduce average deployment adaptation
strength (`0.161-0.621` versus `0.845-0.941` without constraints), but they do
not generalize into a reliable switch detector or universally beneficial
controller. Thus the remaining bottlenecks are absent/run-dependent oracle
control headroom, weak transfer from privileged teacher to learned rollout,
and a training-task safety target that does not generalize to untouched tasks;
task identifiability is no longer the primary problem.

The robust action audit also passes: `CONTEXT_ROBUST` has a zero history gate
and exactly equals the frozen base actor. Base scores differ across variants
because each stochastic SAC run followed a separate trajectory on different
GPU hardware; all conclusions therefore use only within-checkpoint paired
comparisons. Post-iteration-600 logs show `train_base=0` for every run, and the
focused regression suite passes `27/27`, including the proven positive bus
regularization sign.

**Decision:** no V87 variant is eligible for five-seed promotion, and the
reserved stream remains unopened. Stop creating scalar BAPR versions. The
defensible next paper route is the systematic analysis of when adaptation has
usable oracle headroom and when a robust policy is preferable, using bus and
`v87b` Ant as positive cases, historical HalfCheetah only after protocol-matched
revalidation, and Hopper/Walker/termination failures as negative cases. The
full machine-readable ladder and fixed-gate report are
`reports/bapr_v87_validation_2026-07-11.csv` and
`reports/bapr_v87_validation_2026-07-11.md`.

## Scheduler launch audit

The 2026-07-10 BAPR CPU-eval queue sometimes waited 10-45 minutes before
changing from `queued` to `launching`. GPU/CPU capacity was not the primary
cause. The launch preparation path had five compounding delays:

- resume prefetch handled only 8 tasks per pass with 2 task workers;
- each task scanned all six CPU nodes using `python3`, but those nodes expose
  only system Python 2 outside the task conda environment;
- GNU `find` produced valid results but returned nonzero while restoring an
  inaccessible inherited working directory;
- staging ran large cwd transfers serially before exact checkpoint transfers,
  and scan happened after staging, forcing two additional watcher cycles;
- node001-node006 share one workspace, but checkpoint cache keys treated them
  as six independent filesystems and could rsync the same bundle six times.

The scheduler now uses a Python-free remote `find` metadata scan from `/`,
prefetches up to 48 tasks per pass with 4 workers, stages up to 12 cwd candidates
with 6 target-node lanes, gives exact checkpoints priority, performs
`scan -> stage -> dispatch` in one pass, and shares checkpoint staging markers
across `shared_workspace_group`. BAPR's `--dispatch` path now resolves this
batch's exact queued task IDs and performs one targeted bulk dispatch. The
actual launch executor was already 16-way concurrent and did not require a
change.

Real post-fix validation: task `t24542` completed scan, shared-workspace
checkpoint preparation, placement, and remote process launch in `12.57s`.
All six logical CPU nodes hit the same persistent checkpoint marker, and the
remaining BAPR queue reached zero queued tasks. Focused scheduler tests passed
`60/60`; the legacy monolithic regression still contains unrelated/stale
assumptions about watcher phase order and persistent staging cache isolation.

## Cleanup and reproducibility note

Historical raw arrays, checkpoints, eval bundles, figures, and intermediate
reports were intentionally deleted due to disk pressure. This file preserves
the conclusions and headline metrics but cannot reconstruct per-step traces.
The compact V84-V86 diagnostic outputs and current source tree remain locally;
older conclusions rely on this summary plus scheduler metadata.

## Oracle-first stochastic-control screen (2026-07-15)

The empirical-variance round fixed teacher forgetting but showed that one-step
prediction residuals do not identify aleatoric modes. The next experiment now
tests controller headroom before another estimator redesign. Two new persistent
mode families keep morphology and nominal physics fixed: per-step actuator
packet loss and burst torque. Ant and HalfCheetah environment audits passed,
including empirical event rates and identical gravity across modes.

Scheduler tasks `t33462-t33465` train one frozen robust base followed by a
privileged one-hot direct teacher, then compare both branches from the same
checkpoint. A family advances only if oracle gain is at least 10% in both
stationary and switching evaluation on both environments. No learned inference,
remote PKL download, or multi-seed expansion occurs before that gate. The exact
protocol is recorded in
`reports/bapr_v3_oracle_headroom_protocol_2026-07-15.md`.

## Structured action-channel protocol (2026-07-17)

The next fail-closed screen is now running on Ant and HalfCheetah. The new
`structured_channel` family holds morphology and gravity fixed, adds the same
small per-step actuator noise in every mode, and persistently attenuates one of
four equal-sized channel masks: low half, high half, even, or odd. This removes
the packet-loss/burst severity ordering while keeping per-step randomness.

Environment unit tests and 4096-sample audits pass on both robots. Scheduler
tasks `t42217/t42218` are training from fresh immutable `v2` roots on the two
jtl311 GPUs; logs confirm actual CUDA rollout startup. The earlier `v1` tasks
failed before training on a missing CLI choice and are not reused. Learned
mode inference remains blocked until strict robust/oracle and independent
specialist gates establish that switching controllers has real value.

The follow-up is now a controller-budget-matched persistent-mode screen, not
another estimator variant. `t35869-t35876` compare `robust_long` (1400 updates
to one context-free actor) against `oracle_direct` (700 base updates followed
by 700 updates to its exact warm-started conditioned copy) on Ant and
HalfCheetah under both deterministic persistent mean shifts and persistent
mean-plus-per-step-noise modes. Each arm uses 5.6M transitions and 350k critic
updates. No node is pinned. A protocol advances only if dynamic oracle beats
both the equal-budget robust controller and every fixed context, with at least
3/4 matching diagonal optima. Learned inference remains out of scope until
that causal control gate passes.

All four tasks completed `1400` iterations. Burst torque provides substantial
same-checkpoint oracle headroom: Ant gains `+302.20%` stationary and `+74.63%`
switching, while HalfCheetah gains `+14.93%` and `+28.25%`. Packet loss also
technically passes, but its HalfCheetah switching gain is only `+10.18%`, just
above the fixed threshold, so it remains provisional rather than driving the
next estimator choice.

The first learned-inference continuation is therefore burst-only. Tasks
`t34535` (Ant) and `t34536` (HalfCheetah) resume the exact teacher checkpoints
at iteration `1400`, freeze all controller components, and train only a causal
`mode_shared_empirical` context model through iteration `1999`. The model uses
one conditional transition mean and mode-specific empirical residual
distributions with a sticky posterior. Launch logs confirm preservation of the
controller/replay state and a one-time context reset. No remote checkpoint PKL
was manually downloaded, and focused tests pass `20/20`.

Both tasks completed, but the estimator failed rather than the frozen
controller. Ant retained large oracle headroom (`+367.59%` stationary,
`+106.54%` switching), while learned context improved only `+12.44%/+11.34%`
and recovered `3.38%/10.65%` of oracle gain. HalfCheetah learned gains were
only `+0.23%/+0.97%`; its final switching oracle headroom was itself just
`+0.84%`. Last-100 posterior accuracy was `0.309/0.301` on Ant/HalfCheetah,
near four-mode chance, and the four empirical transition variances collapsed.

The causal diagnosis is mean-model error: a 200k-transition Ant audit found
almost no mode separation in raw forward residuals. A clean-mode nonlinear
inverse-dynamics audit, predicting executed action from `(s,s')`, raised the
mode-3 clean-q90 exceedance rate to `0.221` versus `0.100` in mode 0 and reached
`0.735` average 500-step mode classification. Modes 0/1 remain intrinsically
close, but the mode-0/mode-3 switching pair is identifiable.

The implemented follow-up is `inverse_empirical`: train the inverse mean only
on clean mode 0, stop posterior gradients through it, learn mode-specific
commanded-action residual variances by EMA, and collect student evidence with
the frozen robust actor to break posterior-policy feedback. Tests pass `24/24`.
A real-checkpoint smoke preserved iter/steps/replay/controller state, reset the
new context exactly once, completed a finite iteration, and restored that
context on the next start. Scheduler tasks `t35195` (Ant) and `t35193`
(HalfCheetah) continue the same checkpoints from iteration `2000` to `2599`,
without node binding, Slurm, auto-adopt, or manual remote PKL download.

An initial Ant launch (`t35192`) exposed and did not survive a checkpoint
freshness audit: `jtl311linux` had an older iter-1399 copy, and scheduler treated
directory presence as sufficient. It was killed before finishing an iteration.
Resume placement now compares target metadata to the newest source generation,
generation-keys staging markers by mtime/size, and passes `95` focused tests.
The inverse continuation command additionally aborts unless resume begins at
iteration `2000` or later. Replacement `t35195` then loaded the correct
iter-1999 checkpoint on an unpinned node.

Both inverse continuations subsequently completed iter `2599` (`10.4M` steps).
A corrected final-checkpoint audit then evaluated robust, privileged oracle,
and learned contexts over five paired disturbance event streams for each of
Ant and HalfCheetah. All `30/30` accepted outputs load next iter `2600`; this is
one trained policy seed with five event seeds, not a five-training-seed result.

Ant learned context beats its frozen robust branch by `+2758.9` stationary
(95% paired t CI `[+1801.1,+3716.6]`) and `+2055.2` switching
(`[+1574.7,+2535.6]`), positive on all five streams and recovering
`66.7%/62.4%` of oracle headroom. HalfCheetah gains are smaller but also 5/5:
`+224.2` stationary (`[+95.3,+353.2]`) and `+204.4` switching
(`[+79.9,+328.9]`), recovering `70.1%/60.0%` of headroom.

This does not yet prove online adaptation. Exact mode accuracy is only
`0.520/0.489`, expected-mode correlation `0.222/0.139`, and switch AUC
`0.486/0.460` on Ant/HalfCheetah. Both learned filters select mode 0 through
most true mode-3 steps, so a generally stronger static conditioned branch may
explain part or all of the gain. A leakage audit confirms learned context does
not consume privileged task metadata, and non-oracle eval now explicitly
receives a zero oracle latent; the focused suite passes `28/28`.

The next evaluation-only gate fixed context at each of modes `0-3` while the
environment still switched. Ant rejects the adaptation interpretation. Fixed
mode 1 scores `5655.7` stationary and `6600.2` switching, versus learned
`3897.7/5310.8`; learned loses on every event stream with paired differences
`-1758.1` (95% CI `[-2768.3,-747.8]`) and `-1289.4`
(`[-1708.7,-870.2]`). Even dynamic oracle is below the per-stream best fixed
policy in stationary return by `-546.0` (`[-910.6,-181.5]`).

The Ant 4x4 physics-mode by fixed-context matrix is not diagonally optimal:
context 1 is best under clean mode 0, while clean context 0 is best under the
strongest burst mode 3. HalfCheetah independently rejects the same claim.
Fixed mode 0 scores `2831.9/2818.9`, above learned `2624.3/2570.9` and dynamic
oracle `2719.7/2707.3`. Learned minus the per-stream best fixed branch is
`-261.7` stationary (95% CI `[-395.4,-128.1]`) and `-295.5` switching
(`[-381.1,-209.9]`), with 0/5 wins. Dynamic oracle is also below per-stream
best fixed in stationary evaluation by `-166.3` (`[-324.0,-8.6]`). None of the
four HalfCheetah physics modes prefers its matching context (`0/4` diagonal
optima).

Thus the apparent oracle headroom in both environments compares an undertrained
frozen base with a later-trained direct conditioned branch. The base receives
700 actor-update iterations, then freezes while its warm-started conditioned
copy receives another 700. Weak online inference merely activates part of that
stronger static controller; it does not provide demonstrated mode adaptation.
The next mechanism screen must equalize controller-update budget and require
dynamic oracle to beat every fixed context before any learned-estimator work.
The reproducible analyzer is
`scripts/analyze_bapr_v3_static_context_audit.py`; full values and scheduler
retry details are in
`reports/bapr_v3_oracle_headroom_protocol_2026-07-15.md`.

## Budget-matched controller screen result (2026-07-16)

The equal-budget shared-fork screen is complete. Every arm reached iteration
`1399` and `5.6M` environment steps, and every environment/family pair has
`30/30` strict controller outputs over five paired event streams. These are
five evaluation seeds for one training seed, not five policy seeds.

| Family | Env | Oracle - robust stationary | Oracle - robust switching | Diagonal optima | Gate |
|---|---|---:|---:|---:|---|
| deterministic mean | Ant | -266.8 | -384.3 | 1/4 | fail |
| deterministic mean | HalfCheetah | +981.7 | +1035.5 | 2/4 | fail |
| mean + variance | Ant | -154.4 | -304.6 | 1/4 | fail |
| mean + variance | HalfCheetah | +421.1 | +417.7 | 4/4 | **pass** |

Mean-plus-variance HalfCheetah is the sole clean positive mechanism cell:
dynamic oracle beats equal-budget robust and all four fixed contexts in both
stationary and switching paired means, with `4/4` matching stationary context
optima. Its oracle-minus-robust 95% intervals are `[+379.9,+462.3]` and
`[+312.4,+523.1]`. Ant fails under the same family, with oracle below robust
in both protocols and only `1/4` diagonal optima.

Therefore neither disturbance family passes the preregistered requirement on
both Ant and HalfCheetah. No learned estimator is promoted. The result is not
that adaptation never has value: it has well-controlled headroom in the
HalfCheetah mean-plus-variance protocol. The blocker is cross-environment
controller specialization, especially Ant, not another latent-estimator
hyperparameter. A future continuation must either change the Ant controller
objective/curriculum until the privileged oracle passes the same fixed-context
gate, or narrow the scientific claim to oracle-screened environments.

The HalfCheetah producer recovery used immutable archived source and preserved
exact iteration/step budgets. Scheduler protocol-integrity errors now hard-fail
instead of retrying unchanged code; focused scheduler tests pass `44/44`, and
the BAPR budget-match protocol tests pass `8/8`. Full controller tables,
confidence intervals, task IDs, and provenance details are in
`reports/bapr_v3_oracle_headroom_protocol_2026-07-15.md`.

## Ant controller specialization (launched 2026-07-16)

Per-mode inspection localizes the Ant failure. The equal-budget shared direct
oracle improves mode 1 by `+206.5` under deterministic means and `+86.8` under
mean plus variance, but loses `-984.0/-463.7` in mode 2 and also degrades modes
0 and 3. Mean-plus-variance fixed context 1 is best for all four physics rows.
The problem is therefore shared conditioned-actor interference or collapse,
not merely an estimator error.

A new `categorical_expert` actor supplies four independent mode-specific MLPs.
One-hot oracle context routes exactly to one expert; a learned soft posterior
can later mix them. All experts are exact robust-actor copies at the teacher
boundary, and uncertain inference still falls back to the frozen robust actor.
Tests cover exact warm start, categorical routing, posterior mixing, and
configuration validity. A real GPU Ant smoke (`t36492`) completed robust and
teacher stages through iter 3, saved 2048 steps, and confirmed the conditioned
warm-start transition.

Tasks `t36494-t36501` now screen four seed0 variants in both deterministic-mean
and mean-plus-variance Ant: independent experts alone, experts with LCB, and
experts anchored to robust actions at weights `0.1` and `1.0`. All eight use
the same shared-fork, equal-budget protocol, dedicated resumable checkpoint
directories, scheduler-only unpinned placement, and no automatic large-PKL
sync. Learned inference remains blocked until a privileged dynamic oracle beats
equal-budget robust and every fixed context in stationary and switching audits,
with at least `3/4` matching diagonal optima.

Two completed categorical pairs were initially mislabeled failed at finalization:
`t36627/t36628` inherited an obsolete byte-exact long-rollout check from the
direct actor protocol. Exact policy canaries and identical task schedules prove
the categorical experts were correctly warm-started; Ant then amplifies tiny
floating-point differences between the base and one-hot expert computation
graphs. The audit now accepts this narrowly defined categorical proof while
retaining byte-exact enforcement for direct actors. Finalize-only tasks
`t36638/t36639` validated the existing iteration-1399 checkpoints without
retraining. Finalize-only tasks `t37198-t37201` subsequently recovered the
other four completed node007 categorical pairs under the same proof. Scheduler
protocol errors no longer enter unchanged retries; the two `cat_mean` producer
runs were still training and were left untouched.

## Ant categorical specialization result (2026-07-16)

All eight categorical pairs and all `240/240` strict controller outputs are
complete. Each pair is one seed-0 policy evaluated on five paired event streams.

| Variant | Family | Oracle - robust stationary | Oracle - robust switching | Beats every fixed context (S / W) | Diagonal optima | Gate |
|---|---|---:|---:|---|---:|---|
| `cat_mean` | deterministic mean | +76.8 | -290.2 | fail / fail | 2/4 | fail |
| `cat_mean` | mean + variance | +38.0 | +161.3 | fail / fail | 1/4 | fail |
| `cat_lcb` | deterministic mean | +123.7 | -61.8 | fail / fail | 1/4 | fail |
| `cat_lcb` | mean + variance | -774.5 | -400.5 | fail / fail | 2/4 | fail |
| `cat_anchor_0p1` | deterministic mean | -619.3 | -80.3 | fail / fail | 1/4 | fail |
| `cat_anchor_0p1` | mean + variance | +575.2 | +226.0 | fail / fail | 2/4 | fail |
| `cat_anchor_1p0` | deterministic mean | +1641.6 | +826.0 | pass / fail | 2/4 | fail |
| `cat_anchor_1p0` | mean + variance | +284.8 | -172.9 | fail / fail | 1/4 | fail |

The categorical screen is `0/8`: independent per-mode actors, conservative
LCB, and two robust-action anchors do not produce a semantically specialized
Ant controller bank. Several privileged oracles beat a weak robust arm, but no
oracle beats every fixed branch under switching and no fixed-context matrix
has more than `2/4` diagonal optima. The strongest apparent result,
`cat_anchor_1p0/deterministic_mean`, is still dominated by one static branch
for three of four physics modes.

This is stronger than the earlier conclusion that learned inference failed.
The privileged controller itself fails the adaptation gate, even after actor
parameters are fully separated. Shared critic/temperature/replay interference
and limited mode-specific Ant headroom remain the two plausible explanations.
The environment audit confirms persistent 500-step physics modes with only
actuator disturbance sampled per step, and the strict evaluator keeps physics
streams paired while freezing only policy context.

No learned estimator or five-seed expansion is submitted. A future Ant
diagnostic must first train fully independent per-mode specialists, including
critics and entropy temperatures, and require dynamic assembly to beat robust
and every fixed specialist. Otherwise Ant should remain an explicit negative
case, while the already validated HalfCheetah mean-plus-variance and bus cases
carry the positive adaptation evidence. Full values and provenance are in
`reports/bapr_v3_oracle_headroom_protocol_2026-07-15.md`.

## Fully independent Ant specialists (2026-07-16)

The final Ant diagnostic is complete. Each of the four stationary modes has an
independent policy, critic, target critic, entropy temperature, optimizer state,
and replay sampling path; all specialists and the robust control reached
iteration `1399`, `5.6M` steps. Five paired event streams were audited for each
disturbance family.

| Family | Dynamic - robust stationary | Dynamic - robust switching | Beats every fixed specialist (S / W) | Diagonal optima | Gate |
|---|---:|---:|---|---:|---|
| deterministic mean | -173.9 | -431.1 | fail / fail | 2/4 | fail |
| mean + variance | -101.3 | -4.9 | fail / fail | 2/4 | fail |

Thus completely separating the actor, critic, target, alpha, optimizers, and
replay does not recover semantic Ant specialization. A static specialist still
matches or beats dynamic true-mode selection, so estimator, posterior, or gate
tuning cannot repair this cell. No learned-estimator or five-training-seed Ant
batch is justified; Ant remains a negative case under this task family.

The apparent node migration was a scheduler bookkeeping retry, not a second
experiment. Original tasks `t38876-t38880` and `t38882-t38886` produced the ten
valid audit groups on `node004-node006`, but their new completion string was not
yet in the scheduler success patterns. Retry records `t38908-t38917` moved to
`node003-node006`, found the already complete results in the shared remote
workspace, validated them, and exited without recomputation or overwrite. All
ten groups pass the current validator and both aggregate analyses are complete.
Detailed returns, confidence intervals, and provenance are in
`reports/bapr_v3_oracle_headroom_protocol_2026-07-15.md`.

## Stochastic-regime strict rerun (2026-07-16)

The earlier oracle-first random-environment result was correctly recovered:
all eight packet-loss/burst-torque environment/protocol cells exceeded the
frozen 700-iteration base, from `+10.18%` to `+302.20%`. Those numbers prove
potential headroom but are not final BAPR evidence because the conditioned
teacher received 700 additional actor updates while the base was frozen.

Tasks `t38918-t38921` now repeat packet loss and burst torque on Ant and
HalfCheetah with an equal-budget shared-checkpoint fork. They are scheduler-only,
unbound, running across `jtl311linux` and `node007`, and retain large checkpoints
remotely. After completion, strict five-stream audits will compare dynamic
true-mode selection with robust and all four fixed contexts. No learned
estimator or multi-seed expansion is authorized until dynamic selection wins
both stationary and switching comparisons and achieves at least `3/4`
mode-matched diagonal optima. Full protocol details are in
`reports/bapr_v3_oracle_headroom_protocol_2026-07-15.md`.

## Stochastic-regime strict result (2026-07-17)

The equal-budget rerun and all `120/120` strict outputs are complete. Five
paired event streams evaluate one training seed per pair; no checkpoint or
replay PKL was downloaded.

| Family | Env | Oracle - robust stationary | Oracle - robust switching | Diagonal optima | Gate |
|---|---|---:|---:|---:|---|
| packet loss | Ant | +519.9 | +296.3 | 1/4 | fail |
| packet loss | HalfCheetah | -2.1 | +155.7 | 1/4 | fail |
| burst torque | Ant | -879.9 | -174.0 | 0/4 | fail |
| burst torque | HalfCheetah | +304.9 | +256.8 | 1/4 | fail |

Packet-loss Ant and burst-torque HalfCheetah have statistically clear
robust-to-oracle headroom, but no dynamic oracle beats every fixed context and
no matrix has more than `1/4` matching context optima. The learned estimator is
therefore still blocked: the privileged controller bank itself is not
semantically aligned with the stochastic modes.

This changes the diagnosis from "MuJoCo has no stochastic adaptation space" to
"scalar global noise severity does not reliably induce distinct optimal
controllers." The final attribution test is four fully independent stationary
specialists for the two positive-headroom cells. If that also fails, the
benchmark must move to qualitatively distinct action-channel modes such as
joint-group dropout, actuator delay, or persistent signed bias. Full confidence
intervals, fixed-context matrices, task provenance, and compact JSON reports
are in `reports/bapr_v3_oracle_headroom_protocol_2026-07-15.md` and
`jax_experiments/results_bapr_v3_stochastic_headroom_audit_v1/analysis/`.

The final attribution tasks are now running. `t41476-t41479` train independent
packet-loss Ant specialists for modes 0-3 on jtl311; `t41480-t41483` train
independent burst-torque HalfCheetah specialists on node007. Two-iteration GPU
smokes passed first, all tasks resumed the exact shared fork, and only compact
completion manifests will sync locally. Learned inference remains blocked
until the independent true-mode assembly passes the same fixed-specialist and
`3/4` diagonal gates.

## Independent stochastic-specialist result (2026-07-17)

The attribution screen is complete. All eight specialists reached iteration
`1399` / `5.6M` steps, and node-local validation proved independent policy,
critic, target, alpha, optimizer, and replay continuations from one shared
iteration-699 snapshot. Ten strict audit groups evaluated five paired event
streams per cell.

| Family / env | Dynamic - robust stationary | Dynamic - robust switching | Best-row specialist labels | Diagonal | Gate |
|---|---:|---:|---|---:|---|
| packet loss / Ant | +240.7 | +138.0 | `0,0,0,3` | 2/4 | fail |
| burst torque / HalfCheetah | +91.3 | -77.8 | `1,1,1,1` | 1/4 | fail |

Packet-loss Ant still has robust-to-oracle headroom, but fixed mode 0 beats the
dynamic assembly in stationary return and fixed mode 1 beats it in switching.
Burst-torque HalfCheetah is dominated by fixed mode 1 across every stationary
physics row. Completely independent optimization therefore does not recover
four semantically aligned controllers.

The failure is now attributed to benchmark structure rather than BAPR's
estimator or shared optimization state. Scalar packet-loss probability and
zero-mean torque variance form severity scales on which one robust or
moderately conservative policy often dominates. Learned inference and a
five-seed expansion remain blocked. The next oracle screen must use persistent,
qualitatively distinct action-channel regimes such as actuator-group
attenuation, signed bias, and action delay, while retaining per-step noise to
separate aleatoric uncertainty from latent mode changes. Detailed means,
confidence intervals, task provenance, and matrices are in
`reports/bapr_v3_oracle_headroom_protocol_2026-07-15.md`.

## Structured-channel headroom result (2026-07-17)

The equal-severity structured actuator-mask screen is complete for Ant and
HalfCheetah. Ten strict audit tasks evaluated five paired event streams per
environment, with no checkpoint or replay PKL downloaded.

| Env | Oracle - robust stationary | Oracle - robust switching | Row winners | Diagonal | Gate |
|---|---:|---:|---|---:|---|
| Ant | +288.9 | +115.4 | `3,3,1,3` | 1/4 | fail |
| HalfCheetah | +286.3 (+12.2%) | -21.2 (-0.9%) | `0,3,2,3` | 3/4 | fail |

Ant remains a negative case because one static context dominates most modes.
HalfCheetah is materially different: it has significant stationary headroom,
all five paired wins, and three mode-aligned rows. Its small switching deficit
is entirely attributable to the mode-1 controller; it persists long after the
switch and is not an inference-delay artifact because the oracle receives the
true mode.

The remaining attribution test is therefore HalfCheetah only. Bootstrap
`t42594` validated the completed structured source pair and created four exact
iteration-699 forks. GPU tasks `t42595-t42598` resumed at iteration `700` and
train fully independent mode specialists, two per `jtl311linux` GPU. Learned
inference and five-seed expansion remain blocked until the dynamic specialist
assembly beats robust and every fixed specialist in stationary and switching
evaluation with at least `3/4` diagonal optima.

## HalfCheetah control-equivalence breakthrough (2026-07-18)

The four fully independent structured-channel HalfCheetah specialists finished
at iteration `1399` / `5.6M` steps. Their identity true-mode assembly improves
stationary return but still does not beat robust switching. The decisive
diagnostic is the stationary controller matrix: its row winners are
`[0,2,2,3]`. Physics modes 1 and 2 are different transition kernels but share
the same best controller, while specialist 1 is not optimal for its nominal
mode. Earlier BAPR variants failed by treating mode IDs as controller IDs.

A controller map `[0,2,2,3]` was frozen on calibration streams `1100-1500` and
then audited on untouched streams `2100-2500`. The mapped dynamic controller
scores `2742.7 +/- 45.3` stationary and `2523.9 +/- 56.4` switching, versus
equal-budget robust at `2316.9 +/- 53.4` and `2263.5 +/- 120.4`. Differences
are `+425.9` (95% CI `[+334.5,+517.2]`) and `+260.4` (95% CI
`[+74.2,+446.6]`), both `5/5` paired wins. It beats every fixed specialist,
the calibration/holdout row mapping agrees `4/4`, and the strict promotion gate
passes.

This is the first strong MuJoCo adaptation result under the corrected strict
protocol. The next experiment is a causal probabilistic router: infer four
persistent transition modes from `(s,a,r,s')`, model per-step actuator noise as
aleatoric variance, aggregate posterior mass through the frozen many-to-one
control map, and use robust fallback under low confidence. Five policy seeds
remain blocked until the learned router recovers a substantial fraction of the
oracle gain on new stationary and switching streams.

## Learned router: positive result, utility-map gate failure (2026-07-18)

The causal probabilistic estimator was trained once, frozen after independent
validation, and audited on five new event streams by `t43091-t43095` with
aggregate `t43096`. The learned router reaches `2601.9 +/- 90.0` stationary and
`2392.1 +/- 45.3` switching, compared with robust at `2320.5 +/- 91.1` and
`2192.3 +/- 163.7`. Paired gains are `+281.4` (95% CI
`[+157.4,+405.4]`) and `+199.8` (95% CI `[+48.1,+351.6]`), both `5/5` wins.
Routing accuracy is `99.4%/97.0%` with only `0.5%/2.8%` wrong routes.

The result passes every accuracy, delay, positive-CI, and five-win check, but
fails the conservative overall gate because it recovers `60.8%/51.0%` of the
privileged oracle gain rather than the required `70%`. The deficit is not
explained by mode ambiguity. The frozen `[0,2,2,3]` map considered only the four
specialists and omitted robust from per-mode utility selection. Calibration
already shows robust is the best available controller for physics mode 1, yet
the accurate router sends that mode to specialist 2. The old 500-step switching
audit also visits only modes 0 and 1 within its 1000-step horizon.

The next revision is protocol-level and does not retrain the estimator: freeze
a robust-inclusive utility map, encode deliberate robust selection separately
from uncertainty fallback, and evaluate both the original slow switch and a
250-step four-mode cycle on fresh streams. Legacy bus code and the positive
RE-SAC regularization sign remain unchanged.

## Utility-aware router validation (2026-07-18)

The robust-inclusive calibration map is `[0,4,2,3]`; code `4` denotes the
robust controller and the dominated specialist 1 is removed. On disjoint
validation streams, the frozen learned estimator plus utility decision beats
robust in stationary, 500-step slow-switch, and 250-step four-mode-cycle
returns. The original filter recovers `66.2%`, `87.1%`, and `65.6%` of oracle
headroom, respectively. This confirms that learned adaptation is useful under
the corrected stochastic benchmark, but the full-cycle route is not yet
reliable enough for promotion: `86.6%` action accuracy and `13.4%` wrong
routes.

Seven scalar decision/filter variants all failed. Lowering confidence or
minimum history reduced full-cycle recovery; increasing hazard/evidence
shortened some delays but raised noisy misrouting. The strongest fast variant
(`h020e50c80h4`) still has only `83.1%` full-cycle action accuracy and `16.9%`
wrong routes. Sealed streams `7100-7500` were not evaluated. The remaining
bottleneck is the sticky posterior's unbounded accumulated evidence, not the
controller bank, utility map, or absence of return headroom. The next test
keeps all learned parameters frozen and replaces scalar tuning with explicit
bounded-memory evidence accumulation.

The bounded-memory test is now complete and negative. Decays
`0.90/0.95/0.975/0.99` all fail the validation gate. Decay `0.975` improves
four-mode-cycle recovery to `71.8%`, but stationary recovery is only `61.0%`
and mode-3 stationary routing accuracy falls to about `70.5%`. Decay `0.99`
recovers `72.1%/74.4%` on stationary/slow switching but only `39.5%` on the
full cycle. Uniform continuous forgetting cannot preserve weak persistent
mode-3 evidence while also adapting quickly. Sealed holdouts remain unopened;
the next targeted test uses persistent likelihood conflict to trigger a causal
posterior reset while leaving stable-regime accumulation unchanged.

The first event-triggered implementation is also negative. A nonnegative
likelihood-gap EMA reset recovers only `7.1%/38.7%/56.0%` stationary oracle
headroom at thresholds `0.25/0.50/1.00`; all variants sharply increase wrong
routes. Because supporting evidence cannot decrease this statistic, false
resets are inevitable under sustained aleatoric noise. One final
frozen-emission diagnostic will replace it with drift-corrected CUSUM. Failure
there will end filter tuning and require a sequence-aware estimator trained on
switch-matched data.

Drift-corrected CUSUM improves the dynamic result but does not pass the full
gate. Its best predeclared configuration (`threshold=4`, `drift=0.25`) recovers
`87.1%` of full-cycle oracle headroom, with a positive validation paired CI,
but only `60.3%` stationary headroom and `83.6%` full-cycle action accuracy.
The frozen per-transition emission/filter route is therefore exhausted;
sealed holdouts remain unopened. The next estimator must learn temporal
evidence on balanced stationary and 250-step switch-matched sequences rather
than relying on another hand-tuned posterior update.

The first causal GRU sequence router completed `2500` updates after fixing an
NNX non-parameter-state merge bug. It improves stationary physical inference
to `95.6%`, but its strict action-time no-fallback ceiling is only `88.1%` on
full four-mode cycles; the normal robust-fallback decision obtains `82.0%`.
The deficit is almost entirely the first 32 steps after each switch: accuracy
rises from `2.9%` (`0-7`) to `22.9%` (`8-15`), `60.2%` (`16-31`), and `90.2%`
(`32-63`). A bounded internal threshold/temperature diagnostic cannot pass the
gate, so return validation and sealed streams `7100-7500` remain blocked. The
next test is switch-centered training and validation checkpoint selection with
the same frozen emissions and controller bank, not another posterior scalar
sweep.

That controlled curriculum test is complete and negative. Uniform training with
validation checkpoint selection remains best at `93.3%` stationary and `87.9%`
full-cycle no-fallback accuracy. Strong switch-centered variants reduce median
delay from `26.7` to `18-20` steps, but stable accuracy collapses: the best 50%
variant reaches only `85.4%/80.5%`, and stronger/shorter-context variants are
worse. The next justified mechanism is therefore a two-timescale oracle ladder,
not a milder scalar sweep: freeze the stable uniform expert, use a separately
trained fast expert only in a privileged post-switch window, and train a causal
gate only if that assembly clears the internal `90%` ceiling.

The dual-timescale ladder `t43331` exposes only narrow selector headroom. A
privileged 32-step post-switch window reaches `93.3%` stationary and `89.8%`
full-cycle action accuracy (`10.23%` wrong routes), so the fixed-window gate
fails by `0.23` percentage points. A stronger per-step correctness oracle over
the same frozen slow/fast experts reaches `96.2%/91.19%` and `8.81%` full-cycle
wrong routes. Expert complementarity therefore exists, but true elapsed
switch-time is insufficient.

The final causal posterior-based gate `t43333` also fails. Across five model
initializations, the best result is `95.13%` stationary but `87.75%`
full-cycle action accuracy, `12.25%` wrong routes, and `26.9`-step delay. It
uses the fast expert on only `2.35%` of full-cycle actions and does not improve
the frozen slow expert. This closes the dual-router line; further gate or
threshold tuning is unjustified and sealed streams `7100-7500` remain
unopened. A separate control-level observation remains positive: the frozen
CUSUM utility router previously gained `+387.1` full-cycle return over robust
with validation 95% CI `[+108.1,+666.1]`. Its exact-route score is low largely
because deliberate robust fallback is counted as an error, while committed
route accuracy is `93.8%`. The next defensible experiment, if pursued, is a
fresh five-seed return/termination confirmation with all endpoints frozen in
advance, not another estimator sweep.

That confirmation is now predeclared and running as `t43340-t43344`. It freezes
`cs4d025c80h8` on untouched seeds `10100-10500`; primary requirements are a
positive paired full-cycle return CI, at least `4/5` wins, at least `70%`
oracle-headroom recovery, stationary noninferiority within 100 return points,
termination noninferiority within five percentage points, and at least `90%`
accuracy among committed non-fallback routes. The original `7100-7500`
holdouts remain sealed.

The fresh confirmation `t43340-t43346` is complete. Its overall gate is
**FAIL only because full-cycle oracle recovery is `53.9%`, below the frozen
`70%` target**. The control claim itself is positive and reproducible: CUSUM
beats robust on all five fresh seeds in stationary (`+349.5`, CI
`[+283.4,+415.5]`), slow switching (`+345.3`, CI
`[+215.2,+475.4]`), and full-cycle switching (`+237.1`, CI
`[+19.4,+454.9]`). Termination does not increase, and committed full-cycle
route accuracy is `92.7%`; the remaining `18.3%` robust fallback explains much
of the oracle gap. Thus the structured stochastic benchmark now supplies a
confirmed positive BAPR case, but the current hard-switch controller does not
recover enough oracle headroom for the stronger mechanism claim. CUSUM is
frozen after confirmation. The next defensible controller is a
robust-anchored, posterior-conditioned residual policy trained on
switch-matched rollouts, followed by a new untouched five-seed confirmation;
it is not another CUSUM or gate threshold sweep.

The next controller screen is now prospectively frozen. It uses development
seeds `6100/6200` and four action-residual caps (`0.25/0.50/0.75/1.00`) while
freezing the CUSUM posterior and controller bank. Residual strength is derived
continuously from posterior expected specialist advantage, with its scale
fixed by the calibration utility table. Promotion requires a >=50-point mean
full-cycle gain over hard CUSUM, gains on both development seeds and over
robust, stationary/termination noninferiority, and nonzero use. Seeds
`11100-11500` are reserved and will not be opened during this screen.

The linear posterior residual screen `t43349-t43365` is complete and fails.
Baseline replay is bit-for-bit identical. The strongest cap (`1.0`) raises
stationary return by `+66.1` over hard CUSUM and remains `+291.6` above robust
on full cycles, but is `-95.5` below hard CUSUM on full cycles. Smaller caps
are substantially worse. The result closes action interpolation: independent
policy actions do not form a useful linear control path. The next and only
promoted mechanism is a nonlinear state/posterior-conditioned residual trained
on 250-step switch-matched oracle targets with an entropy-weighted robust
anchor and one validation-selected DAgger round. Training/model-selection/
return streams are `3100-3200`/`4100-4200`/`6100-6200`; confirmation seeds
`11100-11500` remain sealed.

That nonlinear residual experiment `t43369/t43380/t43381/t43386` is also
complete and negative. It obtains stationary/slow/full-cycle returns of
`2348.0/2188.2/2193.5`, versus `2644.6/2724.1/2747.4` for frozen hard CUSUM
and `2346.4/2255.3/2360.3` for robust. It loses every hard-CUSUM comparison on
both development seeds and even loses full-cycle return to robust by `166.7`.
No termination rate increases, so the problem is control quality: adaptation
is active on `71.6%` of full-cycle steps with mean action-delta L2 `1.064`,
while held-out oracle-action MSE remains `0.1524`. MSE distillation averages
incompatible specialist actions and creates off-manifold controls. Linear and
nonlinear residual blending are therefore closed; new confirmation seeds
`11100-11500` remain unopened. Any next controller must choose among actual
frozen policy actions using return advantage rather than regress an averaged
action target.

The next v8 diagnostic is prospectively frozen without training. It preserves
hard CUSUM whenever that router is eligible and replaces only its robust
fallback with the real controller maximizing posterior expected utility. This
tests whether the confirmed `18.3%` fallback rate is directly recoverable
without action averaging or another threshold sweep. It uses development
seeds `6100/6200`, the unchanged promotion gate, and hash-referenced baselines;
confirmation seeds `11100-11500` stay sealed.

V8 `t43448-t43450` fails. It matches hard CUSUM in stationary return
(`2645.5` versus `2644.6`) and is only `39.0` lower on slow switches, but
collapses to `2397.5` on full cycles, `349.8` below hard CUSUM and only `37.3`
above robust. It replaces specialists on `11.2%` of full-cycle fallback steps
and raises wrong routes to `21.8%`; episode return loss is strongly associated
with wrong routing (`r=-0.92`). Fallback is therefore protective. The final
capacity check is a privileged true-mode fallback oracle on `6100/6200`, with
all committed hard-CUSUM decisions preserved. It must improve both seeds and
gain at least 50 points on average without extra termination before any
return-advantage selector is trained. Sealed seeds remain unopened.

That privileged v9 capacity check `t43460-t43462` also fails. True-mode
fallback obtains `2677.3`, which is `70.1` below hard CUSUM; per-seed deltas
are `+3.4/-143.6`, with unchanged termination. Despite reducing wrong routes
to `6.25%`, brief specialist substitutions damage trajectory/controller
compatibility and subsequent causal routing. Even the full dynamic oracle has
only `+57.4` mean development headroom and is slightly below hard CUSUM on
seed 6200. The fallback return-selector route is closed, no learner or new
confirmation is launched, and `11100-11500` remain sealed. Frozen hard CUSUM
`cs4d025c80h8` remains the final BAPR controller: its fresh 10100-10500 result
is positive on all five seeds with a positive full-cycle paired confidence
interval, although it does not meet the stronger internal 70% oracle-recovery
target.

## BAPR-v4 persistent option rebuild (2026-07-18)

The next controller is now a separate shared persistent-option SAC rather
than another patch to the frozen v3 router. Four option coordinates and the
robust zero option FiLM-modulate one shared actor; all are trained jointly from
return on the structured-channel switching process. Learned decisions persist
for 64 steps and use the frozen `cs4d025c80h8` CUSUM settings. This directly
tests whether joint trajectory training fixes the state-distribution damage
seen when independently trained specialists were swapped for a few steps.

Smoke task `t43468` passed at iteration 4 / 512 steps, including fused causal
rollout, replay contexts, SAC/context gradients, strict evaluation, and full
checkpoint metadata. Formal task `t43470` completed the equal budget of 1400
iterations / 5.6M steps on `jtl311linux`. Promotion is capacity-first:
the true-mode persistent option must beat hard CUSUM on both development
streams by at least 50 points on average with no termination increase. Only
then is learned persistent routing judged. No sealed confirmation stream is
opened by this screen. Strict audits `t43635/t43636` and aggregation `t43637`
are queued with explicit dependencies on the final training marker and audit
groups; they will not launch on an intermediate checkpoint.

The full v4 chain `t43470/t43635/t43636/t43637` is now complete at iteration
`1399` / `5.6M` steps and fails the privileged capacity gate. On development
events `6100/6200`, true-mode persistent options score `2337.7/2417.9` on the
full cycle, about `372/368` below hard CUSUM. Learned options are lower still
at `1668.7/1716.9`. More importantly, the v4 robust zero context collapses to
`1231.1/1230.2`, roughly half the frozen robust return. There are no early
terminations, and oracle mode selection is exact, so the result rules out an
evaluation or inference explanation. Mode 2 localizes the strongest actor
capacity failure: the v4 oracle obtains only `1355-1616`, while the frozen
specialist oracle obtains `3226-3288` under identical event streams.

V4 therefore closes the single shared FiLM actor, not the broader persistent
option hypothesis. Its robust and option losses overwrite the same trunk and
output heads, and stored-context replay wastes transitions for the other
valid control contexts. The next bounded diagnostic uses a protected robust
actor/critic head, hard option-specific heads, and dual-context replay
relabelling so every transition trains both robust and its true physical-mode
option. It will screen privileged oracle capacity before learned routing or
new confirmation seeds are considered.

That v5 diagnostic is now frozen and submitted. Hard option heads initialize
as exact robust copies but have independent actor and critic parameters after
the first update. Each sampled transition appears once with zero context and
once with its true physical-mode context; robust and option behavior rollouts
alternate 1:1, and the old zero-context auxiliary loss is disabled to avoid
double-counting. The environment, structured stochastic process, 64-step
option persistence, estimator bootstrap, and v3 baselines are unchanged. The
legacy RE-SAC/bus regularization sign remains positive; as in v4, this MuJoCo
screen leaves the BAPR-v2 common critic shift disabled. Smoke `t44379`
completed all paths in 56 seconds.
The first formal launch `t44380` stopped before training because the frozen
bootstrap files were excluded from remote staging; their hashes were preserved
in the protocol snapshot and the clean formal task was resubmitted as `t44385`.
That task and strict audits `t44381/t44382` completed at iteration `1399` /
`5.6M` steps with no early terminations.

V5 improves substantially over v4 but fails the oracle capacity gate. On
event streams `6100/6200`, full-cycle v5 oracle scores `2518.2/2517.5`, still
`191.2/267.9` below hard CUSUM; learned persistent options score
`2035.0/2222.0`. V5 robust recovers from v4 to `2019.2/1882.3`, but remains
below the frozen equal-budget robust policy. Oracle selection is exactly
correct, so this is not an estimator or evaluation failure.

Per-mode returns localize the optimization mismatch. Option 1 exceeds the old
dynamic oracle by roughly `970-983`, option 3 is close, but options 0 and 2
remain `889-1157` lower. In each relabelled `2B` batch, robust receives `B`
examples and each option only about `B/4`, all losses use a single full-batch
mean, all heads share one alpha, and the total update count stays at 250 per
iteration. Thus each option receives roughly one quarter of a fixed
specialist's replay-draw budget. V6 will test only this explanation: balanced
per-head examples, independent temperatures, and update-budget matching at the
same `5.6M` environment steps. It will not change the environment, estimator,
CUSUM, bus path, or positive RE-SAC regularization sign.

## BAPR-v6 optimizer-equivalent result and final data ladder (2026-07-19)

The complete v6 chain `t45429/t45430/t45431/t45437` fails the privileged
capacity gate. Full-cycle oracle returns are `2517.2/2581.1` on development
events `6100/6200`, still `192.1/204.3` below frozen hard CUSUM. Learned
persistent returns are `2388.5/2373.6`. No episode terminates early and oracle
mode selection is exact.

V6 repairs robust training (`2471.6/2424.5`, gains of `+452/+542` over v5)
but leaves the option ceiling almost unchanged. The remaining mismatch is
unique data rather than gradient count: balanced updates give each option the
same total replay draws as a specialist, but each option has only one quarter
of the true-mode transitions and one quarter of the same-mode replay support.
Mode 2 remains about `840-907` below the independent controller bank.

One final capacity-only v7 ladder is frozen: 5600 iterations, 4000 samples per
iteration, 250 updates per iteration, and a 4M replay. It supplies `22.4M`
total switching steps so each option receives approximately `5.6M` unique
mode steps while preserving the v6 per-head optimization budget. It uses the
same bapr_v6 controller, environment, CUSUM, estimator, audit seeds, and
oracle-first gate. Failure ends the joint persistent-option route.

New scheduler submissions no longer reserve arbitrary `4-9GB` VRAM for an
unseen BAPR task. They start at `2048MB`; a formal-shaped smoke and its queued
formal/audit descendants share one VRAM-only resource family, so live observed peaks
replace the cold-start estimate with 20% headroom before dependent launch.

## BAPR-v7 unique-data-equivalent capacity result (2026-07-20)

The final v7 ladder `t47131/t47132/t47133/t47134` completed at iteration
`5599` / `22.4M` environment steps. It retained v6's independent heads and
optimizer-equivalent updates, but provided every physical option approximately
`5.6M` unique true-mode transitions and `1M` same-mode replay support: the
same data scale as an independently trained hard-CUSUM specialist. Strict
audits again reached the full 1000-step horizon and privileged routing was
exact.

| Event | Source | Full cycle | Full - hard CUSUM |
|---:|---|---:|---:|
| 6100 | hard CUSUM | 2709.3 | +0.0 |
| 6100 | v7 robust | 2065.0 | -644.4 |
| 6100 | v7 oracle option | 2654.2 | -55.1 |
| 6100 | v7 learned option | 2209.5 | -499.9 |
| 6200 | hard CUSUM | 2785.4 | +0.0 |
| 6200 | v7 robust | 2079.3 | -706.1 |
| 6200 | v7 oracle option | 2804.8 | +19.4 |
| 6200 | v7 learned option | 2385.6 | -399.8 |

The predeclared oracle gate requires a positive margin on both events and at
least `+50` on average. V7 therefore fails despite matching independent-head
data support; learned routing also remains well below the fixed controller
bank. This closes the jointly trained persistent-option architecture under the
structured-channel protocol. Future work must change the control formulation
or benchmark hypothesis rather than add another router, gate, residual, or
capacity sweep to this branch.

## BAPR-v8 independent-training-seed validation (preregistered 2026-07-20)

V8 does not introduce another controller variant. It freezes the best confirmed
independent-controller formulation, `cs4d025c80h8`, and tests whether its
previous gain survives independent policy-training seeds and same-protocol
baselines. The benchmark remains HalfCheetah `structured_channel`; the physical
mode, stochastic channel, 1000-step strict horizon, frozen estimator parameters,
and hard-CUSUM decision rule are unchanged.

Training seeds are fixed to `0-4`. Each seed launches seven separate GPU tasks:
one switching SAC robust policy, one switching ESCP policy, one switching
RE-SAC policy, and four stationary SAC specialists. Every controller receives
exactly `1400 x 4000 = 5.6M` environment steps and `349500` gradient updates.
RE-SAC preserves the proven positive regularization sign with
`weight_reg=beta_ood=0.01`; its optimizer uses the original implementation-scale
`1e-5` learning rate, while SAC and ESCP use `3e-4`. A completed task publishes
an immutable hash-checked evaluation bundle before deleting its replay file.

For each training seed, controller utilities are calibrated only on event seeds
`2100/2200`. Final evaluation streams `11100-11500` are sealed and disjoint from
calibration. Each policy seed/event seed pair is audited on all four stationary
modes, slow switching at dwell 500, and a full cycle at dwell 250 using paired
environment randomness. SAC, ESCP, RE-SAC, hard-CUSUM BAPR, and the dynamic
specialist oracle are evaluated in the same audit code. The independent
statistical unit is the policy-training seed, not an evaluation episode or
disturbance stream.

Promotion requires all four conditions: the paired 95% CI of BAPR minus the
strongest SAC/ESCP/RE-SAC baseline is positive; BAPR wins at least four of five
training seeds; mean stationary loss is no worse than 100 return; and full-cycle
termination rate is no more than 0.05 above the strongest baseline. Failure
ends the current hard-CUSUM claim rather than triggering post-hoc threshold
tuning on the sealed streams.

Scheduler chain `t48225-t48290` was submitted on 2026-07-20. The first staging
wave exposed a launch-wrapper omission: `task_num/test_task_num` retained their
default value 40, which the four-mode environment rejected before iteration 0.
No checkpoint or return was produced by those failed launches. The wrapper and
its protocol-signature checks now require both values to equal 4; corrected
automatic retries were verified at iteration 0 with a 4000-step checkpoint.
At that verification point 31/35 GPU training tasks were running, four were
queued, and every CPU calibration/audit task remained dependency-blocked.

On 2026-07-21, all 35 final controller bundles passed budget and SHA-256
validation. Four calibration tasks and 20 strict audits completed, while seed-2
calibration `t48262` failed because result synchronization exposed its bundle
manifest before `specialist_mode_3/logs/protocol_signature.json` had reached
the shared CPU workspace. This was a staging race, not a training or checkpoint
failure. CPU dependencies now wait for all four immutable files in every bundle
(manifest, parameters, training state, and protocol signature), and the audit
loader verifies each bundle hash before constructing an agent. Corrected retry
`t49829` loaded the complete 5.6M-step seed-2 controller bank on `node006`; its
five downstream audits remain file-gated until calibration completes.

## BAPR-v8 independent-training-seed result (2026-07-21)

All five calibration tables, 25 sealed-stream audits, and final aggregation
completed. The final aggregation task was initially restricted to busy `local`
CPU (`free 0/16`); the JSON-only analyzer was decoupled from JAX/model-bundle
loading, opened to `local,node001-node006`, and completed as `t48290` on
`node003`.

| Controller | Stationary | Slow switching | Full cycle |
|---|---:|---:|---:|
| SAC | 1674.8 +/- 213.8 | 1682.8 +/- 185.1 | 1689.6 +/- 208.1 |
| ESCP | 2039.5 +/- 528.9 | 1987.2 +/- 484.1 | 2007.1 +/- 523.9 |
| RE-SAC | -188.6 +/- 34.9 | -189.6 +/- 35.7 | -188.7 +/- 35.0 |
| hard-CUSUM BAPR | 1474.1 +/- 367.8 | 1441.9 +/- 352.5 | 1400.4 +/- 432.5 |
| dynamic specialist oracle | 2267.3 +/- 352.7 | 2226.5 +/- 453.3 | 2168.1 +/- 413.9 |

The preregistered promotion gate fails. Full-cycle BAPR minus SAC is `-289.2`
with paired 95% CI `[-742.7,+164.3]` and 2/5 seed wins; BAPR minus ESCP is
`-606.7` with CI `[-1519.2,+305.8]` and 1/5 wins. Against the strongest baseline
within each training seed, BAPR is `-686.5`, CI `[-1430.4,+57.3]`, with only
1/5 wins. There are no early terminations. The oracle remains substantially
above BAPR, so the controller bank has adaptation headroom, but the frozen
estimator/CUSUM router does not recover it consistently across policy seeds.
The anomalous negative RE-SAC return is a separate baseline-reproduction issue
and cannot be used as evidence for BAPR; BAPR already fails against both SAC
and ESCP without relying on that comparison.

## Shared-regime BAPR redesign and headroom screen (2026-07-21)

The hard-CUSUM/expert-routing branch is frozen after the v8 promotion failure.
The replacement path uses one shared robust actor plus a bounded residual,
empirical mode-conditioned transition variance, and a sticky causal posterior.
Aleatoric variance is represented by the transition model rather than Q-head
dispersion; BOCD, Q-std switching, and independently switched policy banks are
excluded.  The RE-SAC regularization coefficient retains its proven positive
sign (`weight_reg=beta_ood=0.01`).

The controller now has three explicit stages.  Robust pretraining learns the
fallback controller.  Inference pretraining learns the heteroscedastic context
model while still acting through the robust controller.  At the adaptation
boundary, the replay buffer is cleared and the residual output is initialized
to exactly zero, so adaptive training starts from the same robust action.  The
base actor is then frozen and only the bounded residual and critic are updated.
Learned deployment additionally falls back to the robust action unless its
estimated advantage is positive; the oracle screen deliberately disables that
fallback to measure an unrestricted upper bound.

Before training a learned estimator, a frozen seed-0 HalfCheetah
`mean_variance` headroom protocol compares equal-budget SAC, ESCP, RE-SAC,
shared robust BAPR, and true-mode oracle residual BAPR.  Every role receives
`1400 x 4000 = 5.6M` environment steps and `350000` updates.  Completion
requires a hash-checked compact bundle containing parameters, training state,
and the protocol signature.  The subsequent strict audit must compare dynamic
oracle context with robust and all four forced fixed contexts.  Learned
adaptation is permitted only if the dynamic oracle beats every fixed context
and the equal-budget robust controller, with the correct context optimal on at
least three of four stationary rows.

Scheduler tasks `t49867-t49871` were submitted at high priority without node
binding, Slurm, or auto-adopt.  Their initial placement is SAC/ESCP on
`jtl311linux` GPUs 0/1 and RE-SAC/robust/oracle on `node007` GPUs 0/1/2.
Cold-start VRAM is conservatively set to 2GB and checkpoint resume remains
managed by each immutable controller command.

The strict audit chain is already dependency-gated in scheduler as
`t49872-t49897`: 25 independent CPU evaluations (five controller roles by five
sealed event seeds) plus one JSON/CSV aggregation task.  Evaluation is allowed
only on `node001-node006`, requests zero VRAM, and each task waits for all four
files of its producer bundle rather than merely a task status.  The aggregator
waits for all 25 atomic audit manifests.  At submission all 26 tasks correctly
reported `BLOCK waiting for prerequisite file(s)`; none was launched before
training completion.  The oracle audit loads one checkpoint and evaluates the
dynamic true-mode context plus fixed contexts `0-3` on identical stationary
tasks and switching event streams.

## Equal-budget regime-control headroom reset (2026-07-22)

The next BAPR step no longer tunes another estimator against the same uncertain
controller target. It first isolates whether persistent mode information has
causal control value under an identical architecture and budget. New
`RegimeSAC` robust and oracle arms share actor, critic, target critic,
temperature, initialization, replay, and update code. Robust receives a zero
four-vector; oracle receives the true one-hot mode used by the current physics
step. Encoder, residual, gate, BOCD/CUSUM, expert bank, and RE-SAC
regularization are absent from both arms.

The frozen `structured_channel` benchmark keeps robot physics and the affected
actuator subset fixed for each 250-transition dwell. Only equal-strength
Gaussian actuator noise is sampled per step. The 30-run matrix is three
environments (`HalfCheetah`, `Ant`, `Walker2d`) by two arms by five independent
training seeds, each at 5.6M transitions and 350k updates. The exact protocol
and preregistered promotion rule are in
`reports/regime_control_headroom_protocol_2026-07-22.md`.

Scheduler tasks `t50115-t50144` cover training, `t50145-t50174` are one
CPU-only five-event audit per checkpoint, and `t50175` is the final aggregate.
Every audit waits for bundle manifest, parameters, training state, and protocol
signature; the aggregate waits for all 30 audit manifests. No task uses Slurm,
auto-adopt, or a pinned GPU node. Cold-start VRAM was 2GB.

The first dispatch exposed a scheduler/runtime interaction rather than an
algorithm failure: 16 simultaneous JAX cold starts on node007 caused five
`ptxas`/SIGABRT failures before iter 0. The scheduler preserved the signatures
and created child retries `t50177-t50181` on the jtl GPU nodes. All retained
node007 tasks crossed iter 0, and the active set returned to 30 unique training
signatures. Learned-estimator work remains forbidden until the frozen oracle
gate passes in at least two of three environments.

All 30 training bundles and 30 strict audits subsequently completed and passed
hash/budget validation (`iter=1399`, `next_iter=1400`, 5.6M transitions, 350k
updates). The final equal-budget result is:

| Environment | Robust switching | Oracle switching | Oracle relative gain | Robust worst | Oracle worst | Oracle relative gain | Gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| HalfCheetah | 1945.0 +/- 326.3 | 1317.7 +/- 324.4 | -32.3% | 1553.9 +/- 451.5 | 832.2 +/- 402.8 | -46.4% | fail |
| Ant | 2930.8 +/- 173.7 | 3291.4 +/- 238.3 | +12.3% | 2716.3 +/- 242.4 | 2906.6 +/- 176.2 | +7.0% | fail |
| Walker2d | 2177.4 +/- 73.0 | 2200.6 +/- 18.3 | +1.1% | 192.7 +/- 54.2 | 172.2 +/- 38.7 | -10.6% | fail |

The learned-estimator gate therefore fails with 0/3 passing environments. Ant
does contain a positive signal: switching improves by +360.6, paired 95% CI
[+22.4,+698.8], and all four stationary mode means improve. It misses the
predeclared worst-mode threshold because that gain is only 7.0% with an
interval crossing zero. HalfCheetah shows significant negative conditioning
value, while Walker2d has negligible switching gain and 100% termination for
both arms. Under this exact benchmark and budget, the learned estimator is not
the next bottleneck: even privileged current-mode input does not provide the
required broad controller headroom.

Recovery also exposed an operational false positive. The scheduler's bare
`OOM` substring matched the terminal word `HEADROOM`, turning successful
`exit_code=0`/`DONE` tasks into failed retries and preventing result sync. The
matcher was changed to require a standalone `OOM` token, successful attempts
were reclassified, and all bundles were synchronized without retraining. A
local independent aggregation reproduced the decision and human-readable
table exactly; JSON differences from the remote CPU node were only last-bit
floating-point rounding.


## Event-grouped cross-context diagnosis (2026-07-22)

The equal-budget result still confounded environment headroom with failure of
the shared conditioned controller. A checkpoint-only diagnostic therefore
re-evaluated the separately trained robust checkpoint and the oracle checkpoint
under true, zero, fixed_0 through fixed_3, and cyclically wrong contexts. One
scheduler task contains all eight cases for one environment, training seed,
and sealed event seed. This event grouping is necessary because a local
Python 3.11/JAX rollout did not numerically reproduce the earlier remote
Python 3.10 trajectory even with the same checkpoint and seed. All paired
controller and context comparisons are now made inside one process and runtime.

The frozen matrix contains 75 event tasks (three environments by five training
seeds by five event seeds). Every task validates both compact bundles at
next_iter=1400 and 5.6M steps, writes eight case directories atomically, and
checks that true mode m exactly matches fixed_m while cyclic mode m exactly
matches fixed_(m+1) in stationary evaluation. All 75 unique manifests passed.
The reproducible protocol and result are in
reports/regime_cross_context_protocol_2026-07-22.md and
reports/regime_cross_context_diagnostic_2026-07-22.md.

| Environment | Robust switch | True switch | Zero switch | Best fixed switch | True-robust | True-best fixed | Diagonal | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| HalfCheetah | 1877.7 | 1317.7 | 391.3 | 1176.7 | -29.8%, CI [-1162.1,+41.9] | +12.0%, CI [+53.9,+228.1] | 3/4 | Dynamic mode information is useful, but shared conditional training degrades the control base |
| Ant | 2838.5 | 3291.4 | 2132.9 | 2524.0 | +16.0%, CI [+357.8,+547.9] | +30.4%, CI [+536.9,+997.7] | 4/4 | Valid adaptation-positive environment |
| Walker2d | 2177.4 | 2204.4 | 2125.5 | 2192.6 | +1.2%, CI [-77.4,+131.4] | +0.5%, CI [-44.7,+68.3] | 2/4 | Invalid controller benchmark because robust and true both terminate in 100% of streams |

HalfCheetah is not an estimator-ambiguity or no-specialization result. True
context beats zero by +926.3 with paired CI [+234.7,+1618.0], beats the best
constant context in every training seed, and has a large fixed-context spread.
The failure occurs because the conditional controller loses the robust policy
capability before inference is considered. Ant establishes real oracle
headroom in all five seeds and is the only current environment on which a
learned estimator is scientifically justified. Walker2d must pass an absolute
survival gate before further controller comparisons.

Scheduler tasks t51044-t51118 covered the 75 audits. Five initial node005
attempts failed at process startup because 11 simultaneous JAX CPU clients
exhausted the pthread limit; child retries t51120-t51124 completed. CPU claims
were raised from 4 to 32 cores, limiting each 192-core node to six concurrent
JAX processes, and the submitter now sets single-thread XLA/TF/BLAS flags. No
failures occurred after that correction. The fresh aggregate t51127 passed
all pairing and identity checks.

The next algorithm work is split rather than universal: (1) Ant may train a
causal mode estimator against the already successful frozen oracle controller;
(2) HalfCheetah first needs a frozen robust actor plus mode-specific bounded
adapters/residuals with a no-degradation constraint, still using true mode,
and may not train an estimator until that oracle controller beats robust; (3)
Walker2d needs a milder, survival-valid stochastic regime before any BAPR
claim. This replaces further BOCD/gate tuning.




## Frozen-base independent-adapter fork launched (2026-07-23)

The HalfCheetah controller-isolation test is implemented and running under the
preregistered protocol in
`reports/regime_adapter_fork_protocol_2026-07-22.md`. Every branch starts from
the same audited seed-8 robust controller at iter 1399 / 5.6M transitions. The
robust arm receives 2.8M additional transitions. Each candidate bank has four
independent fixed-mode residual actors and fully independent critics, target
critics, alpha states, optimizers, replays, and tasks; each branch gets 0.7M
additional transitions, so the four-branch aggregate is also 2.8M. The base
actor is frozen and hash-checked. Residual caps 0.25, 0.50, and 1.00 are
evaluated with two calibration streams and five sealed paired streams.

Scheduler graph `t51149-t51180` contains 13 GPU training tasks, three CPU
calibrations, 15 CPU audits, and one aggregate. Only training was explicitly
dispatched; all downstream work is guarded by bundle/calibration manifest
files. GPU bootstrap found a CUDA numerical-check false positive, not a model
mismatch: the actor matched exactly and the widened zero-context critic had
0.0967 maximum float32 GEMM error on a 127.6 Q scale. Exact parameter-block and
zero-coordinate checks remain strict; the forward probe now uses global-scale
tolerance. Retries are `t51183,t51184,t51188,t51189,t51195`. The final replacement
`t51199` excludes node007 while remaining unpinned across the other four GPU
nodes, and launched on jtl110gpu2. This also motivated explicit single-thread
XLA/BLAS/JAX/TF child settings after a node007 `pthread_create` failure. No algorithm or budget was changed.

Promotion remains deliberately hard: utility routing must beat the 8.4M-step
continued robust policy by at least 10% switching and 5% stationary, win at
least 4/5 paired streams, beat every fixed adapter, and remain termination
noninferior. Failure means the HalfCheetah bottleneck is not merely shared
actor/critic interference; passing permits five independent training seeds
before any learned estimator. Ant causal-estimator work remains a separate,
already justified branch because Ant alone passed the cross-context oracle
headroom diagnosis.

## HalfCheetah independent-adapter development result (2026-07-23)

All 13 GPU branches, three calibrations, 15 sealed audits, and the final
aggregate completed. Independent local validation reproduced every manifest
and score. Residual caps `0.50` and `1.00` passed the development gate; the
smaller `0.50` cap was frozen for confirmation. With identity routing it
improved mean stationary return from 2196.9 to 2583.0 (+17.6%) and switching
return from 2212.0 to 2569.0 (+16.1%), won all five paired streams, beat the
best fixed adapter, and added no terminations.

This changes the HalfCheetah diagnosis. Persistent mode information does have
control value, but the old shared conditional actor/critic destroyed the robust
base through negative transfer. Freezing that base and giving each mode an
independent bounded residual and critic recovers the headroom. The result is
not yet a learned BAPR claim because routing still receives true mode and seed
8 selected the configuration.

The preregistered follow-up is in
`reports/regime_adapter_confirmation_protocol_2026-07-23.md`. It freezes
`delta=0.50` and map `[0,1,2,3]`, treats seed 8 as development-only, and makes
seeds `16,24,32,40` the primary independent confirmation units. Five new event
streams are averaged within each training seed. A causal learned router remains
blocked until the adapter bank beats robust and each seed's best fixed adapter
with positive paired intervals under the frozen gate.

The confirmation scheduler graph is `t51224-t51269`: 20 new unpinned GPU
branches, 25 CPU audits restricted to `node001-node006`, and one aggregate.
Seed-8 bundles are reused, not retrained. The bulk cold start again exposed
node007 pthread pressure and sub-0.2% CUDA forward-reduction differences. Both
are operational rather than algorithmic: retries retain checkpoint-safe
signatures, and the forward diagnostic now uses documented `5e-6` action and
`0.2%` global-Q bounds while exact parameter/hash checks remain binding.
Three signatures whose retries repeatedly returned to node007 received
checkpoint-safe non-node007 replacements `t51294`, `t51296`, and `t51298`.
All 20 unique training signatures then entered running state. No running branch
was migrated or restarted for load balancing.

## HalfCheetah adapter multi-seed confirmation result (2026-07-26)

All 25 bundles, 25 audits, and the aggregate completed. Independent local
validation passed after fixing a provenance portability bug: remote and local
absolute workspace roots differed, but every recorded manifest SHA256 and size
was identical. Provenance is now compared by workspace-relative path plus
content hash.

The seed-8 development gain did not replicate against the continued robust
controller. On untouched seeds `16,24,32,40`, identity adapters average `-8.5%`
stationary and `-10.3%` switching, with one of four wins; the promotion gate
fails. Seed8 was misleading because its robust continuation degraded relative
to the frozen base, while robust continuation improved sharply for seeds
24/32/40.

The mechanism diagnosis is more favorable than the primary comparison.
Identity adapters improve the frozen base in all four holdout seeds:
stationary `+299.8`, 95% CI `[+63.6,+535.9]`; switching `+236.1`, 95% CI
`[+95.8,+376.4]`. Correct-mode routing is stationary-optimal in 13/16 mode
rows. The failure is sample/optimization efficiency: each independent
critic/optimizer receives only 0.7M post-fork transitions, while one shared
robust controller receives all 2.8M and obtains roughly twice the gain.

Do not train the causal router yet. The next diagnostic gives every adapter the
robust controller's full 2.8M post-fork per-controller budget. The compact
iter-1575 bundles omit replay, so they cannot be extended without a second
replay reset. The clean protocol restarts each branch once from the common
iter-1400 source and trains continuously for 700 iterations. This deliberately
compute-unmatched upper bound determines whether longer independent
specialization can work at all. A positive result then motivates a shared
all-mode critic/backbone with lightweight mode-specific residual heads; a
negative result rejects the independent-adapter route.

## Equal-per-controller adapter diagnostic prepared (2026-07-26)

The implementation and frozen gate are documented in
`reports/regime_adapter_equal_controller_protocol_2026-07-26.md`. The graph has
16 GPU branches, 20 strict CPU audits, and one aggregate. Every fixed-mode
adapter starts from the validated 5.6M-transition robust source and trains
continuously to 8.4M transitions. The four-controller bank therefore consumes
16.8M unique aggregate transitions when the shared pretrain is counted once,
versus 8.4M for robust. This is an upper-bound mechanism diagnosis, not a fair
paper comparison.

Source and robust bundles for all four holdout seeds passed budget and
content-hash validation. The runner verifies the source boundary, final update
count, fixed-mode rollout log, frozen base hash, changed residual/critic hashes,
and atomic compact bundle contents. Fresh audits use event seeds `77100-77500`
and treat the four training seeds, not 20 event streams, as the independent
units. A pass authorizes a shared all-mode backbone redesign; a failure rejects
independent adapter optimization.

Scheduler graph `t51411-t51447` contains training `t51411-t51426`, audits
`t51427-t51446`, and aggregate `t51447`. Twelve GPU producers launched across
local and the three jtl nodes. The final four `t51422-t51425` were admitted to
the otherwise idle node007 and launched one per GPU after input staging.
Sampled launch logs on jtl and node007 verified `iter=1400/5.6M` resume with
empty replay; node007 showed no pthread startup failure. CPU audits remain
correctly blocked on their seed-specific bundles.

The initial node007 GPU2/3 attempts stopped only because their copied actor
forward difference (`5.6177e-6`) narrowly exceeded the old `5e-6` CUDA
diagnostic tolerance; critic error was zero. The tolerance is now `1e-5`,
while exact parameter/hash, zero-context, zero-residual, and frozen-base checks
remain unchanged. Scheduler children `t51465` and `t51466` replaced those two
attempts and resumed from iter 1400 on node007 GPU3 and GPU2. Node007 is now
running one producer on each of its four GPUs.

## Equal-per-controller adapter result (2026-07-26)

All 16 branches reached iter 2100 / 8.4M transitions, all 20 sealed audits
finished, and aggregate `t51447` completed. The deliberately compute-unmatched
upper bound fails:

| Seed | Robust stat | Identity stat | Gain | Robust switch | Identity switch | Gain |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 2295.2 | 2551.5 | +11.2% | 2163.8 | 2412.3 | +11.5% |
| 24 | 1993.4 | 1608.7 | -19.3% | 1932.3 | 1593.2 | -17.5% |
| 32 | 2431.3 | 2089.8 | -14.0% | 2395.5 | 1904.8 | -20.5% |
| 40 | 2436.7 | 2386.9 | -2.0% | 2300.8 | 2213.9 | -3.8% |

At the policy-seed level, identity minus robust is `-5.7%` stationary
(`-129.9`, 95% CI `[-542.7,+282.9]`) and `-7.6%` switching (`-167.0`, 95% CI
`[-615.8,+281.7]`), with one win in four on each metric. Identity routing also
does not reliably beat the best fixed adapter. All termination rates are zero.

There is real but insufficient specialization: the diagonal controller is best
among adapters in 12/16 stationary mode rows, and identity improves the frozen
base in every seed by `+20.9%` stationary and `+16.6%` switching. Continued
robust training improves the frozen base more. Seed 24 already loses across all
four stationary modes and has almost no extra switch penalty, proving that the
main failure cannot be the learned estimator, gate, or detection latency.
Seed 32 combines weak stationary adapters with an additional `8.9%` switching
penalty; seed 40 is near stationary parity but incurs a `7.2%` switching
penalty. Only seed 16 is positive.

This privileged true-mode bank consumes four times the robust post-fork data.
Its failure rejects the independent-adapter optimization route rather than
authorizing more router work. Do not train a learned router or continue tuning
these heads. A future control redesign must train mode-conditioned behavior
jointly under cross-controller state distributions and explicitly preserve the
robust policy objective. Full protocol and row-level diagnosis:
`reports/regime_adapter_equal_controller_protocol_2026-07-26.md`.

Post-hoc diagnosis narrows that decision. Both arms reset replay and optimizer
state at the fork, so continuity is not the culprit. The adapter instead
freezes a frequently undertrained 5.6M-step base and learns only a bounded mean
residual, while robust continues its complete actor. Identity improves the
frozen base in 15/16 mode rows, but robust full-actor gains are `+556/+678`
for the two strongest failing seeds 24/32. Source quality and adapter-relative
outcome have descriptive correlation `r=0.91` across the four seeds.

The critic objectives are also mismatched. Robust uses the minimum ensemble
target; BAPR-v2 uses independent per-head targets while this protocol disables
the RE-SAC regularizer, OOD penalty, and LCB actor. Adapter Q scales end
`1.6x-3.4x` above robust without corresponding returns. In addition, residual
mode freezes policy log standard deviation while continuing to optimize alpha,
and three poor branches collapse alpha below `1e-15`. The RE-SAC positive
regularization sign remains correct but is inactive here because
`bapr_v2_reg_weight=0`.

The environment audit remains valid: persistent 250-step actuator masks,
equal per-step execution noise, fixed morphology/gravity, no observation noise,
and exact oracle routing. Training-seed variance dominates event-stream
variance, so this is principally a controller-optimization failure, not an
evaluation or mode-generation bug. The next admissible diagnostic is a
minimum-target, entropy-matched residual from a converged robust base, trained
on switching-state distributions. No estimator or gate work is justified
before that oracle passes.

## Mature-base min-target residual result (2026-07-26)

The preregistered minimal validation in
`reports/regime_adapter_latebase_min_protocol_2026-07-26.md` completed. It
isolated the two strongest previously failing seeds, 24 and 32, started every
adapter from its completed 8.4M-step robust controller, changed the BAPR critic
to the same ensemble-minimum target as robust SAC, froze the inherited entropy
temperature, and cloned one canonical residual initialization to all four
fixed-mode branches. Each controller received 0.7M additional transitions.
Ten strict audits used five held-out event streams per policy seed.

The correction removes the earlier catastrophic negative transfer but does not
pass the oracle gate:

| Seed | Robust stat | Identity stat | Gain | Robust switch | Identity switch | Gain | Diagonal optimal |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 2009.1 | 2247.0 | +11.8% | 1935.2 | 2101.7 | +8.6% | 4/4 |
| 32 | 2410.9 | 2485.4 | +3.1% | 2384.8 | 2418.1 | +1.4% | 2/4 |

Seed 24 now has broad specialization: the identity oracle improves all four
modes by 8.6%-19.2%. Seed 32 improves modes 0-2 by only
`+0.4%,-0.9%,+1.9%`; only mode 3 has useful headroom at +9.6%. The stronger
seed-32 robust policy already captures nearly all behavior available to the
fixed-mode residuals. Its switching difference also varies across zero over
the five event streams. Frozen-base equivalence is exact and all termination
rates are zero, so neither result corruption nor survival bias explains the
failure.

The mean `+7.5%` stationary gain is not a pass: the sealed rule requires at
least +5% and at least 3/4 diagonal-optimal rows in every policy seed. This
privileged true-mode, extra-compute upper bound fails both seed-32 conditions.
Do not train a learned router, posterior, or gate on this branch. The result
narrows the historical diagnosis: immature bases, independent critic targets,
temperature collapse, and inconsistent initialization caused much of the
earlier severe loss, but the remaining adaptation headroom in the current
structured-channel HalfCheetah benchmark is mode- and policy-seed-dependent
rather than a stable BAPR advantage.

## Actuator-polarity benchmark headroom screen launched (2026-07-26)

The next branch changes the benchmark before changing BAPR. The sealed
`actuator_polarity` family gives four persistent actuator-calibration regimes:
low-half, high-half, even, or odd motor channels have their sign reversed for
the full 250-step dwell. Gravity, morphology, reward, absolute actuator
authority, and per-step Gaussian execution noise (`std=0.02`) are identical
across modes. This creates conflicting control mappings at the same state
without resampling robot physics every step.

Environment audits passed on HalfCheetah, Ant, Hopper, and Walker2d, including
the odd-dimensional Hopper action space. A two-iteration end-to-end
`RegimeSAC` oracle smoke completed fused rollout, replay, updates, and
checkpointing. The scientific screen uses equal-budget robust and true-mode
oracle arms at 5.6M transitions, training seeds `8,16,24`, and three sealed
event streams per seed. It has 24 GPU producers (`t52036-t52059`), 24
file-gated CPU audits (`t52060-t52083`), and aggregate `t52084`; `jtl311linux`
is excluded.

This is a benchmark positive-control screen, not a BAPR result. At least three
of four environments must show at least 15% oracle gain in both switching and
worst-mode stationary return, positive paired policy-seed intervals, at least
3/4 improving modes, and noninferior termination. No learned estimator is
authorized before this gate passes. Full preregistration:
`reports/regime_polarity_headroom_protocol_2026-07-26.md`.

## Actuator-polarity oracle-headroom result (2026-07-27)

All 24 training jobs and 24 strict audits completed at the sealed 5.6M-step
budget. The aggregate global gate fails with 0/4 formally passing
environments. This does not mean all four environments lack adaptation
headroom:

| Environment | Switching gain | Worst-mode gain | Mode wins | Policy-seed pattern | Diagnosis |
|---|---:|---:|---:|---|---|
| HalfCheetah | +256.5% | +875.6% | 4/4 | 3/3 positive | large but underpowered |
| Ant | +85.1% | +141.3% | 4/4 | 3/3 positive | large but seed-variable |
| Hopper | +3.0% | +14.3% | 3/4 | mixed signs | no reliable headroom |
| Walker2d | +0.7% | +13.8% | 2/4 | mixed signs | no reliable headroom |

HalfCheetah and Ant have consistent within-checkpoint evidence: every one of
their nine policy-seed/event-stream switching pairs favors the oracle. Their
paired 95% intervals still cross zero because the inferential unit is the
training seed (`n=3`) and between-policy variance is large. Hopper and
Walker2d are not merely underpowered: both arms terminate in every stationary
and switching evaluation, and gains are small or change sign by seed.

The environment concern raised after the earlier BAPR failures is now narrowed.
`actuator_polarity` uses a persistent 250-step mode, fixed robot physics, and
per-transition Gaussian actuator noise; it does not resample morphology or add
observation noise every step. It creates genuine controller headroom in
HalfCheetah and Ant, but not a universal four-environment benchmark at the
current severity.

No learned BAPR result follows from this screen. The next checkpoint-only test
must establish that true context beats zero, every fixed context, cyclically
wrong context, and shuffled context, then measure how much oracle gain remains
with 1/5/10/25/50-step context delay. This separates mode-conditioned control
value from an unattainable zero-latency oracle. Only candidates that retain
headroom under causal delay proceed to a fresh five-policy-seed confirmation.

Hopper and Walker2d require a new development protocol with milder invertible
actuator transforms and an absolute survival gate. They may not be rescued by
tuning BAPR on the failed full-polarity setting. If the HalfCheetah/Ant
cross-context and delay audits pass, the admissible algorithm is a
heteroscedastic transition model plus sticky semi-Markov posterior feeding a
policy trained on uncertain/delayed beliefs. BOCD/LCB, critic-Q uncertainty,
always-on residual gates, and independent adapter banks remain rejected by the
previous experiments.

## Polarity context-delay audit launched (2026-07-27)

The next phase reuses the completed HalfCheetah and Ant checkpoints without
training. In one paired event task it compares the robust controller and the
oracle checkpoint under true, zero, fixed 0-3, cyclically wrong, shuffled, and
1/5/10/25/50-action delayed true contexts. Switching traces now record the
context mode actually presented to the policy; zero context is represented as
`-1`, and delayed context holds the previous mode for exactly the declared
number of actions.

This is an attainability audit, not another BAPR variant. The three existing
policy seeds remain exploratory. An environment advances only if true context
beats robust and the per-seed best fixed-context envelope on all three seeds,
at least 3/4 stationary rows are diagonal-optimal, and delays of 10 and 25
actions retain at least 70% and 50% of oracle headroom while still beating
robust on every seed. Passing authorizes a fresh independent five-seed
robust-versus-oracle confirmation; it does not yet authorize estimator
training. Full protocol:
`reports/regime_polarity_context_delay_protocol_2026-07-27.md`.

The scheduler graph is `t54573-t54590` for the 18 CPU audits and `t54591`
for the file-gated aggregate. An initial source-staging collision was fixed in
scheduleurm by declaring the shared NFS workspace used by `node001-node006`
and deduplicating immutable input staging through the same cross-process
guard. The original audit tasks then launched; no replacement tasks or
checkpoint copies were created.

## Polarity context-delay result and fresh confirmation (2026-07-27)

All 18 checkpoint-only audits passed their hash, budget, event-stream, and
action-level context checks. HalfCheetah passes the continuation gate:
dynamic true context scores `2633.9`, compared with `738.8` for robust and
`1394.5` for the per-seed best fixed-context envelope. It is diagonal-optimal
in 4/4 stationary rows. Delays of 10 and 25 actions retain `87.9%` and `64.0%`
of instantaneous-oracle headroom and beat robust on all three exploratory
training seeds.

Ant remains blocked despite a high zero-delay oracle score. Its delay-10 and
delay-25 retention falls to `59.8%` and `15.2%`; delay 25 does not beat robust
on every seed. This is an estimator-feasibility failure under the sealed
causal-delay rule, not permission to tune a faster detector post hoc.

The next protocol is therefore HalfCheetah only. It trains equal-budget robust
and true-mode oracle controllers from scratch on five new seeds
`101,211,307,419,523`, then evaluates three sealed streams
`91001,91002,91003`. Passing requires at least 10% gains in switching and
worst-mode return, positive paired 95% training-seed intervals for both, at
least 3/4 improving stationary modes, and no more than a five-point switching
termination penalty. A learned posterior remains blocked until this fresh
confirmation passes. Full preregistration:
`reports/regime_polarity_confirmation_protocol_2026-07-27.md`.

The submitted graph is `t54602-t54611` for fresh GPU training,
`t54612-t54621` for per-checkpoint CPU audits, and `t54622` for aggregation.
The two first launches load the correct `actuator_polarity`, fixed-250 dwell,
5.6M-step configuration from empty confirmation output roots. Remaining
training tasks are unpinned and wait only for GPU capacity; audits and analysis
are file-gated.

## Fresh polarity confirmation passes (2026-07-28)

The independent HalfCheetah confirmation completed with all ten sealed
controller bundles and 10/10 strict audits. The robust arm scores
`1033.8 +/- 318.8` on switching, while the true-mode oracle scores
`2308.0 +/- 475.4`. The paired policy-seed gain is `+1274.2` (`+123.3%`,
95% CI `[+750.2,+1798.1]`) with 5/5 seed wins. Worst-mode stationary return
improves from `682.2 +/- 289.9` to `2012.3 +/- 486.4`; the paired gain is
`+1330.1` (`+195.0%`, 95% CI `[+932.7,+1727.5]`), again with 5/5 wins.

All four stationary modes improve, every mode has 5/5 seed wins and a
strictly positive paired 95% interval, and both arms have zero termination.
The result passes every preregistered gate and confirms that the persistent
actuator-polarity HalfCheetah benchmark has substantial, reproducible
adaptation headroom. It resolves the earlier ambiguity: poor BAPR performance
on the old benchmarks cannot be used to conclude that online adaptation has no
value, but neither does this privileged oracle result count as a learned BAPR
result.

The authorized next step is a frozen-controller causal-estimator screen:
heteroscedastic transition likelihood, separate aleatoric/epistemic
uncertainty, and a sticky posterior evaluated on untouched event streams.
The posterior must directly drive the already validated conditioned policy;
there is no BOCD/LCB/Q-std gate, residual bank, or hard option selector.
Only a successful estimator screen permits end-to-end
posterior-conditioned SAC training.

## Causal polarity-posterior result (2026-07-28)

The first learned-estimator screen completed on all five untouched policy
seeds and three sealed event streams. It fails decisively. Switching return is
`1023.4` for robust and `2327.7` for the true-mode oracle, but only `799.4`
for the causal soft posterior and `727.8` for posterior MAP. The learned arms
recover `-22.6%` and `-27.4%` of oracle headroom and beat robust on only 1/5
and 0/5 seeds. Their termination gap is zero.

The soft estimator reaches mode accuracy `0.537`, Brier score `0.839`, median
delay `10`, and P90 delay `369.4`; therefore posterior-conditioned policy
training is blocked. The problem is not lack of adaptation headroom and not an
HMM threshold alone. Mode 1 is nearly perfectly recognized, while mode 0 is
systematically assigned to other heads because the empirical forward-model
variance and mean errors rank the wrong conditional likelihood highest.

A diagnostic leave-one-seed-out affine classifier over the frozen likelihood
vectors reaches `0.848` one-step accuracy and `0.957` with ten causal
transitions. Thus useful transition information survives in the model, but
the raw Gaussian likelihood is misspecified. The next admissible experiment
is a supervised evidence-calibration screen: freeze the forward model, fit an
affine causal temporal calibrator on exploratory seeds `8,16`, select its
memory only on seed `24`, and rerun the untouched five-seed audit. It is not
permission to tune on seeds `101,211,307,419,523`, alter the policy, or return
to BOCD/LCB/Q-variance gates.

## Causal evidence-calibration result (2026-07-28)

The v2 affine evidence calibrator also fails under actual causal deployment.
The selected `EMA=0.9`, `ridge=0.001`, `temperature=0.25` model scores only
`0.537` switching mode accuracy with Brier `0.530`, median delay `23`, and
P90 delay `250`. Frozen-controller switching return is `746.3` for soft
belief and `602.8` for MAP, versus `1023.4` robust and `2327.7` true oracle.
Soft and MAP recover `-25.8%` and `-38.6%` of oracle headroom and win only
1/5 and 0/5 policy seeds.

Mode 0 is systematically aliased with the other actuator-polarity modes,
whereas mode 1 is almost perfectly recognized. The discrepancy with the
earlier leave-one-seed-out trace diagnostic is explained by closed-loop
covariate shift: that diagnostic classified trajectories generated by a
different posterior/controller path, while deployed mistakes change the next
action and state distribution. An affine correction to the forward
likelihood is therefore neither policy-invariant nor transferable.

No direct posterior-conditioned SAC training is authorized. The next
diagnostic is inverse system identification: learn to infer the executed
actuator command from consecutive observations, then score each known
persistent gain pattern against the commanded action. Privileged
`executed_action` is allowed only as an exploratory training target and is
excluded from online inference and all sealed audits. This tests whether the
remaining obstacle is the transition representation rather than adaptation
headroom, which is already confirmed by the five-seed oracle result.

The inverse system-ID screen is implemented as an independent v3 protocol.
Its fused data path exposes `executed_action` only when explicitly requested,
leaving every existing five-field rollout caller unchanged. Five independent
inverse MLP heads train on separate bootstrap minibatches from exploratory
controller seeds `8,16`; seed `24` selects only the sticky-filter parameters.
Online audit inputs are limited to consecutive observations and commanded
action. Tasks `t58371-t58377` form one GPU training producer, five
file-gated CPU audits on untouched seeds `101,211,307,419,523`, and one
file-gated aggregate. No policy is trained in this screen.

## Executed-action inverse system-ID result (2026-07-28)

The v3 screen passes decisively. On the five untouched controller seeds,
causal soft inference reaches `0.9992` mode accuracy, Brier `0.0013`, and
switch delays of `5` median / `11` P90 actions. It scores `2190.5` switching
return versus `1023.4` robust and `2327.7` true oracle. The paired gain over
robust is `+1167.0`, 95% CI `[+688.3,+1645.7]`, with 5/5 seed wins and
`89.3%` mean oracle-headroom recovery. MAP also passes at `2135.1`, 5/5
wins, and `84.5%` recovery. Neither learned arm adds termination.

Stationary soft performance (`2303.5` mean, `2005.7` worst mode) matches the
true oracle (`2320.3`, `2003.0`) and far exceeds robust (`1047.4`, `635.7`).
This is the first BAPR branch in the strict actuator-polarity protocol to pass
both causal inference and frozen-control gates.

The result identifies the previous failure precisely: adaptation headroom and
conditioned control were not the problem. Separate forward-likelihood heads
learned policy/trajectory-specific distributions and aliased mode 0.
Predicting one shared physical quantity and letting candidate actuator
transforms define the hypotheses generalizes across policy seeds and closes
the loop without that aliasing.

The remaining limitation is privileged supervision during exploratory
training: v3 targets the realized simulator `executed_action`. The next
registered ablation replaces it with
`clip(gain(true_training_mode) * commanded_action)`, so random per-step
execution noise is never observed as a label. The online estimator and all
five frozen gates remain unchanged. Direct learned-posterior policy training
is scientifically justified only after this supervision ablation, or must be
presented explicitly as asymmetric simulator-to-real training.

The no-realized-action v4 graph is `t58608-t58614`: one GPU model fit, five
file-gated CPU closed-loop audits, and one aggregate. Its collectors assert
the standard five-field rollout contract and never request
`executed_action`. The repeated five policy seeds make this a development
ablation only; a pass freezes the method and triggers a final confirmation
with newly trained policy seeds.

## Expected-action supervision result (2026-07-28)

The v4 ablation passes every frozen gate. Soft inference reaches `0.9975`
mode accuracy, Brier `0.0040`, and `5/12` median/P90 switch delay. Switching
return is `2171.3` versus `1023.4` robust and `2327.7` oracle. The paired
gain is `+1147.9`, 95% CI `[+604.7,+1691.1]`, with 5/5 seed wins and
`86.2%` mean oracle-headroom recovery. Stationary mean is `2312.1`, nearly
identical to oracle `2320.3`.

This is only 19.1 switching-return points below the realized-action v3
positive control (`2190.5`) and retains essentially the same inference
quality. Realized per-step simulator noise supervision is therefore
unnecessary. The frozen estimator uses only expected action transformed by
known exploratory training mode; deployment remains fully causal and receives
no mode or actuator metadata.

No further tuning on policy seeds `101,211,307,419,523` is permitted. The
final confirmation retrains robust and true-context controllers from scratch
on new policy seeds and evaluates this exact v4 model and filter on new event
streams. Any code, threshold, environment, or estimator change after seeing
that confirmation invalidates it.

## Independent frozen-method confirmation (submitted 2026-07-28)

The v4 estimator and filter are now frozen by SHA-256, and the final
confirmation uses new policy seeds `607,719,823,929,1031` plus new event
seeds `95001,95002,95003`. Ten equal-budget controller tasks
(`t58889-t58898`) train robust and true-context policies from scratch.
Five paired four-arm audits (`t58899-t58903`) compare robust, oracle,
frozen-soft, and frozen-MAP only after both corresponding bundles are
complete; `t58904` performs the final aggregate. At submission, the first
training task started on `local` and the remaining GPU work entered the
checkpoint-safe, node-reroutable queue. No final-confirmation result was
inspected before the complete sealed batch finished.

## Independent confirmation result (2026-07-29)

All final-confirmation tasks completed and all ten fresh controller bundles
validate at iteration `1399`, 5.6M transitions, and 350k updates. Frozen-soft
inference transfers cleanly: accuracy `0.995`, Brier `0.008`, switch delay
`4/10` median/P90, and no termination penalty.

Control does not pass the sealed five-seed gate. Switching return is `1340.1`
robust, `2283.5` true-mode oracle, `2161.5` soft, and `2156.1` MAP. Soft gains
`+821.4` on average but wins only 3/5 seeds, with paired 95% CI
`[-287.1,1929.8]`. Seed `719` has almost no oracle headroom; on seed `1031`
the oracle itself is `180.6` below robust. Thus the estimator is not the
remaining bottleneck: frozen-soft stays within `122.0` points of oracle on
average across all five seeds. The independently trained conditioned
controller is too seed-sensitive to support the final BAPR claim.

The preregistered result is recorded as
`estimator_passed_but_frozen_control_failed`; no threshold or seed is changed.
Further work, if pursued, starts a new development split and replaces the
free-standing conditioned actor with a zero-initialized residual over a paired
frozen robust actor plus an explicit robust fallback. The v4 estimator remains
frozen and these final seeds remain sealed from tuning.

## Robust-anchored residual development (2026-07-29)

The next controller is now implemented as `anchored_regime_sac`. It retains
an observation-only robust actor, adds a bounded mode residual initialized to
exactly zero, and isolates both actor and critic gradients between robust and
adaptive paths. The robust actor continues SAC training instead of being
frozen at an undertrained fork. Robust and adaptive critics are independent,
the adaptive critic is shared across modes, critic targets use SAC's minimum
operator, and entropy temperatures are context-specific.

This is a new development split, not a reinterpretation of failed final seeds.
Policy seeds are `1103,1213,1301`; calibration streams are `96001,96002`;
strict audit streams are `96101,96102,96103`. Each seed trains a fresh 5.6M
transition robust source, then forks into an equal 2.8M-transition robust
continuation and anchored continuation. Both finish at 8.4M transitions and
525k updates. The anchored branch alternates robust and oracle rollouts 1:1
and relabels each replay batch with paired robust/oracle contexts.

Fallback is explicit and same-checkpoint. Held-out calibration enables a mode
only if its true-mode residual beats both the anchored base and equal-budget
robust continuation by at least 2% without more than a two-point termination
penalty. Frozen posterior control requires confidence at least `0.85` and an
enabled MAP mode; otherwise gate zero recovers the anchored base action
exactly.

Unit tests pass for zero-output initialization, actor and critic gradient
isolation, source-controller copying across all five contexts, calibration
mask behavior, and the 16-task file dependency graph. A real two-iteration
HalfCheetah smoke records mode IDs `[0,1]`, rollout sources
`[robust,oracle]`, dual-context multiplier `2`, and a valid checkpoint. Full
protocol and promotion gates are frozen in
`reports/regime_polarity_anchored_residual_protocol_2026-07-29.md`.

The complete scheduler graph is `t59185-t59200`. Source tasks
`t59185-t59187` started on node007 GPU0/1/2 with the correct
`actuator_polarity`, fixed-250-dwell, 5.6M-transition command. The six branch
tasks and all seven CPU tasks are file-gated and did not launch early.

## Robust-anchored residual result (2026-07-29)

All 16 tasks completed with valid hashes and exact budgets. The branch does
not pass development: `anchored_base`, `oracle_residual`, and `learned_safe`
are respectively `-38.6%`, `-72.4%`, and `-37.9%` below the equal-budget
robust continuation on average, each with only 1/3 seed wins. Seed 1301 is a
positive case (`+7.8%` learned-safe gain, mode mask `0111`), but seeds 1103
and 1213 disable every residual mode and their true-mode residual returns are
negative.

This failure is upstream of mode inference. Frozen inference is essentially
perfect on seeds 1103 and 1301, and no arm terminates. The supposed robust
anchor was only gradient-isolated, not trajectory-isolated: adaptive
rollouts were replay-relabeled into the robust path, so the base actor evolved
under a different closed-loop distribution than its paired robust
continuation. Gate-zero performance therefore fell 39.5% and 82.0% on two
seeds. The shared bounded residual also failed to establish stable oracle
headroom on those seeds.

No untouched confirmation is launched. The next development experiment
starts from the completed robust-continuation checkpoints, freezes the base
actor exactly, and spends an equal additional interaction budget on either a
shared bounded residual or zero-initialized independent mode residual heads.
An equal-budget robust-long arm remains the comparator, and held-out
calibration may only enable residual modes that beat both the frozen base and
robust-long controller.

## Frozen-anchor v2 development launched (2026-07-29)

The follow-up now starts from the completed 8.4M-transition v1 robust
continuations and freezes each adaptive branch's base actor byte-for-byte.
Every branch receives the same additional 2.8M interactions and finishes at
11.2M transitions. Final validation rejects any adaptive checkpoint whose
base-policy hash differs from its fork hash.

Three registered variants separate residual capacity from anchor
preservation: shared residual caps `0.15` and `0.50`, plus independent
mode-specific additive mean/log-std residuals with zero output initialization
and no amplitude cap. All adaptive rollouts use oracle context during
development; held-out calibration and the unchanged frozen expected-action
estimator determine deployable fallback behavior.

Unit and regression tests pass. Full-shape 250-update GPU tests change
adaptive parameters without changing the base hash, and a real
two-iteration mode-residual run validates resume, oracle-only rollout,
checkpointing, and publication. Measured scheduler claims are 2.8 GB for the
shared residuals and 4.2 GB for the mode residual, rather than default 4/8 GB
claims.

The 33-task scheduler-only graph is `t63877-t63909`: 12 GPU producers, 9
file-gated calibrations, 9 file-gated strict audits, and 3 aggregates.
`jtl311linux` is excluded. Full registration:
`reports/regime_polarity_frozen_anchor_protocol_2026-07-29.md`.

## Frozen-anchor v2 result (2026-07-30)

All three registered variants completed the three-seed strict audit and
failed promotion. The frozen expected-action estimator is not the bottleneck:
mode accuracy is `0.9958-0.9996`, Brier score is at most `0.0561`, median
switch delay is `2-3` steps, and every estimator gate passes.

`shared_small` is the only non-catastrophic controller. Its true-mode oracle
residual averages `+31.8` switching return (`+8.3%`) but wins only `1/3`
seeds. Learned-raw wins `2/3` seeds but averages only `+9.1`; calibrated
learned-safe averages `+27.8` but wins only `1/3`. Calibration enables modes
`0` and `2` only for seed `1301`; seeds `1103` and `1213` fall back for every
mode. The paired 95% intervals include substantial losses, so this is not
stable headroom.

Increasing residual capacity makes control strictly worse. `shared_wide`
true-mode oracle control is `-30.4%` relative to robust and wins `0/3` seeds.
Independent unrestricted `mode_residual` heads collapse in every seed:
true-mode oracle control is `-132.2%`, with negative returns for all four
stationary modes. Learned control tracks these failures, showing that the
problem is residual-policy/critic optimization rather than mode inference.

Gate-zero control also misses the per-seed `95%` preservation requirement:
the frozen 8.4M-transition anchor is slightly below the independently
continued 11.2M robust controller on seeds `1103` and `1301`, while seed
`1213` moves in the opposite direction. No untouched five-seed confirmation
is launched.

The next controller experiment must therefore target conservative policy
improvement, not another estimator or gate. Keep the immutable robust actor,
drop unrestricted mode heads, and train a small bounded residual only when a
pessimistic critic lower bound predicts positive advantage over the exact
base action. The training objective must explicitly penalize per-mode
degradation and reject updates whose paired return falls below the base.
Until a true-mode oracle residual wins at least `2/3` development seeds,
learned-posterior training is not scientifically justified.

## Conservative frozen-residual v3 launched (2026-07-30)

The next development protocol now implements the registered conservative
policy-improvement test. The 8.4M-transition robust actor is immutable. Only a
zero-initialized bounded residual is optimized, with a smooth target-critic
LCB penalty relative to the exact base action. Every candidate actor update is
then checked per represented mode; a violating update and its optimizer state
are rolled back together.

Three variants distinguish a truly absent ascent direction from critic noise:
`strict_small` uses a 0.15 residual cap with zero regression tolerance and a
nonnegative LCB floor; `trust_small` keeps the cap but allows 0.005 tolerance
and a -0.01 floor; `trust_tight` uses the relaxed filter with a 0.075 cap.
Each branch still finishes at the same 11.2M-transition budget as the reused
v2 `robust_long` comparator.

Regression tests cover per-mode rejection, optimizer rollback, immutable base
hashes, old-v2 compatibility, scheduler dependencies, and new variant network
selection. A real seed-1103 bootstrap from the 8.4M checkpoint has exactly
zero action and Q error across zero plus all four one-hot contexts.

The scheduler-only graph is `t64314-t64343`: nine GPU branches, nine
file-gated calibrations, nine strict audits, and three aggregates. All nine
branches launched on `node007` and resumed at `iter=2100`; launch logs confirm
the registered constraints. GPU placement excludes both thermally unstable
`jtl110gpu2` and occupied `jtl311linux`. Full registration is in
`reports/regime_polarity_conservative_residual_protocol_2026-07-30.md`.

## Conservative frozen-residual v3 result (2026-07-30)

All 30 tasks completed at the registered 11.2M-transition budget, but the
experiment froze before testing its intended policy variants. Across all three
variants and all three seeds, actor update acceptance is exactly zero and the
adaptive-policy hash never changes. Therefore the strict, relaxed, 0.15-cap,
and 0.075-cap branches all evaluate the same frozen 8.4M actor.

The shared result is robust-long `1468.2/600.5/1902.8` versus frozen-base
`1410.7/772.3/1752.3` for seeds `1103/1213/1301`. Mean paired delta is `-12.1`
with only `1/3` wins; all calibration masks are `0000`. Oracle, learned-raw,
learned-safe, and base returns are identical. Estimator accuracy remains
`0.9958-0.9996` with `2-3`-action median delay, so inference is not implicated.

The failure is the global acceptance rule: every minibatch update to one shared
residual parameter tree must pass every represented mode. This can veto all
updates under cross-mode gradient conflict or a pessimistic target critic, and
it cannot certify a nonlocal path such as actuator sign reversal. Candidate
LCB/margin and rejection-cause metrics have now been added. The next action is
a few-update checkpoint-resume diagnostic, not another full sweep. If it
confirms finite cross-mode vetoes, replace the shared all-mode transaction with
bounded mode-specific residual heads and per-head rollback while retaining the
immutable robust fallback.

### Correction after candidate instrumentation

The preceding causal attribution to the global all-mode rule is superseded by
an exact short-run diagnosis. Tasks `t64456-t64458` show that every candidate,
in every mode, was non-finite before the rule evaluated regression or floor.
The source is the raw ensemble `std` in the LCB penalty: at the exact
zero-residual anchor its value is zero but its gradient at zero variance is
undefined. The first actor update became NaN and was correctly rolled back.

Replacing it with `sqrt(variance+1e-6)-sqrt(1e-6)` keeps the zero anchor and
makes its gradient finite. Clean reruns `t64460-t64462` have zero non-finite
rate. `trust_small` accepts 97.6% of candidates and changes its actor;
`trust_tight` accepts 100% and changes its actor. Only the strict zero-floor arm
remains frozen. Therefore v3 did not test conservative residual performance
and cannot support an algorithm conclusion.

## Stable-LCB conservative residual v4 launched (2026-07-30)

The corrected full rerun retains only the two relaxed variants and starts every
branch from the clean 8.4M source. Its 20-task scheduler graph is
`t64464-t64483`: six GPU branches, six file-gated calibrations, six file-gated
strict audits, and two aggregates. The producer tasks `t64464-t64469` are
running; all CPU tasks remain gated on published branch/calibration files.

The environment, 11.2M final budget, three development seeds, held-out event
seeds, robust-long comparator, frozen estimator, and promotion gates are
unchanged. Old v3 checkpoints are excluded. GPU placement allows only
`local`, `jtl110gpu`, and `node007`; CPU evaluation allows only
`node001-node006`. Full registration:
`reports/regime_polarity_conservative_residual_stable_protocol_2026-07-30.md`.

## Stable-LCB conservative residual v4 result (2026-08-01)

All 20 tasks `t64464-t64483` completed and synced. The smoothed standard
deviation fixed the v3 NaN defect: every branch changed its actor, candidate
updates stayed finite, and mean update acceptance was `0.959-1.000`. The
negative result is therefore algorithmic rather than an execution artifact.

`trust_small` true-context oracle residual improves paired robust return by
only `+29.8` on average, wins `2/3` development seeds, and has a paired range
of plausible effects spanning `-92.2` to `+137.4`. Its learned-raw and
calibrated learned-safe deltas are `+38.3` and `+9.2`; calibration masks are
`0000/0000/1010`. `trust_tight` is weaker: oracle residual delta `-5.6`,
learned-raw `-4.6`, learned-safe `+25.9`, with only `1/3` seed wins and masks
`0000/0100/1000`. Neither arm passes promotion.

The frozen expected-action estimator remains accurate (`0.9952-0.9996`) with
`2-3`-action median delay, so estimator error cannot explain the failure.
Instead, the target-critic LCB is poorly calibrated to realized return and a
bounded local residual cannot reliably represent the nonlocal policy change
needed for actuator sign reversal. Further residual-cap, LCB-floor, or gate
sweeps are retired.

This does not erase the strongest conditioned-policy result. In the first
sealed five-seed controller set, robust/oracle/learned-soft switching means
were `1033.8/2308.0/2171.3`; learned-soft beat robust on `5/5` seeds and
recovered `86.2%` of oracle headroom. In the independent final five seeds the
means were `1340.1/2283.5/2161.5`, but learned-soft won only `3/5`, so the
pre-registered final gate correctly failed. A retrospective ten-seed view is
informative but not confirmatory: robust/oracle/learned-soft means are
`1181.8/2305.6/2166.4`, learned-soft delta is `+984.6` with `8/10` wins, and
the approximate paired 95% t interval is `[494.8, 1474.4]`. The remaining
problem is rare controller-seed instability, not absence of adaptation
headroom or failure of online mode inference.

The next checkpoint-only diagnostic will evaluate the five development
robust/oracle controllers on identical fresh event streams, including every
individual controller plus deterministic mean-action and median-action
ensembles. This cleanly tests whether controller variance can be reduced
without another RL run. If an oracle ensemble preserves the large headroom
and removes the seed collapses, the next trainable algorithm is a single
posterior-conditioned student distilled from that ensemble; otherwise the
current conditioned-policy optimization remains too unstable for a main
claim.

## Policy-ensemble checkpoint diagnostic launched (2026-08-01)

The diagnostic is implemented without new RL training. It separately tests
the development controllers (`101,211,307,419,523`) and final controllers
(`607,719,823,929,1031`) on fresh event seeds `97001-97003`. Each group
contains all individual robust/oracle policies, deterministic coordinate-wise
mean and median robust/oracle action ensembles, and mean/median ensembles
driven causally by the frozen expected-action v4 posterior. Results from the
two controller groups remain separate because this is retrospective diagnosis,
not a new ten-seed confirmation.

The scheduler-only CPU graph is `t64515-t64521`: six audits restricted to
`node001-node006`, followed by one file-gated aggregate. No GPU, Slurm, or
auto-adopt path is used. The promotion rule requires both controller groups to
show at least 10% oracle and learned ensemble improvement on every event and
at least 70% learned recovery of oracle headroom. Only that outcome justifies
distilling an ensemble teacher into one posterior-conditioned student.

## Policy-ensemble diagnostic result and distillation screen (2026-08-01)

All seven ensemble tasks completed. Learned median is the stable reduction:
development/final switching returns are `2062.3/1970.9`, with zero
termination, and learned context recovers `93.6%/94.6%` of oracle-median
headroom. The earlier automatic recommendation of mean was an ordering bug;
after ranking passed reductions by their worst-group learned return, median is
selected.

The preregistered relative gains against matched robust action ensembles must
not be quoted as algorithm performance. Those robust ensembles collapse to
`227.4/407.9` in development and `-34.2/-93.1` in final, while mean individual
robust returns are `1037.8/1338.4`. The valid conclusion is that median oracle
aggregation stabilizes conditioned controllers and remains usable with the
causal posterior, not that it improves robust SAC by thousands of percent.

A new retrospective screen now tests deployable compression. It trains one
posterior-conditioned policy from the frozen median teacher using oracle,
learned-teacher, and individual-robust state distributions plus two DAgger
rounds. Three teacher sets (development five, final five, combined ten) and
three student initializations produce nine GPU tasks. Twenty-seven strict CPU
audits compare each student with the episode-paired distribution of individual
robust controllers on event seeds `99001-99003`; one aggregate is file-gated
on all audits. Only the combined-ten group can promote, requiring at least two
of three student seeds plus the seed chosen solely by supervised validation to
pass. Full registration is in
`reports/regime_polarity_policy_distillation_protocol_2026-08-01.md`.

## Policy-distillation screen result (2026-08-01)

The complete graph finished and synced: nine student models, 27 strict CPU
audits, and aggregate task `t64610`. Five failed scheduler records are
superseded launch lineages rather than missing arms. The scheduler staging bug
that caused those retries was fixed before the clean replacement runs.

The development, final, and combined median-teacher groups pass with `3/3`,
`2/3`, and `3/3` student seeds. The validation-selected combined student is
seed `1511`; it achieves switching return `1954.2` versus robust-population
return `1182.9` (`+771.4`, `15/15` episode wins), with zero termination and a
worst stationary-mode return of `1962.3`. It also exceeds the best individual
robust controller (`1921.5`) by `+32.7`, with `11/15` episode wins.

This passes the preregistered retrospective promotion rule and shows that the
causal median teacher can be compressed into one deployable policy. The claim
must remain narrow: the advantage over the best individual robust controller
is only about `1.7%`, and the validation-selected final-group student is tied
with its best robust controller (`-3.0`). This is not yet evidence that the
method beats SAC, ESCP, or RE-SAC. The next experiment is a genuinely new
five-seed confirmation with combined seed `1511` frozen solely by supervised
validation; no audit-based reselection is allowed.

## Frozen policy-distillation confirmation launched (2026-08-01)

The validation-selected combined student seed `1511` is now frozen by its
manifest, parameter, and selection-analysis hashes. Five previously unused
event seeds (`100019,100043,100069,100103,100151`) form the independent
confirmation split. The primary comparator is preregistered as
`robust_final_seed_719`; it cannot be reselected from confirmation returns.

Scheduler tasks `t64717-t64721` run the five paired 32-core CPU audits, and
file-gated task `t64722` performs the aggregate. No training, GPU task, Slurm,
or auto-adopt path is involved. The overall gate requires population wins on
`5/5` event seeds, wins over the fixed strongest robust controller on at least
`4/5`, positive event-clustered 95% t intervals for both comparisons, at least
80% learned-teacher headroom recovery, and zero switching termination. Full
preregistration is in
`reports/regime_polarity_policy_distillation_confirmation_protocol_2026-08-01.md`.

## Frozen policy-distillation confirmation result (2026-08-01)

All tasks `t64717-t64722` completed and synced. The frozen combined student
seed `1511` confirms its large improvement over the ten-controller robust
population: switching return `1899.4` versus `1178.9`, delta `+720.5`, wins
`5/5` event seeds, and event-clustered 95% t interval `[+659.7,+781.4]`.
Headroom recovery is `88.7%`, posterior mode accuracy is `0.9978`, and
switching termination is zero.

The preregistered overall gate nevertheless fails. Frozen strongest robust
controller `final_seed_719` scores `1933.0`; the student delta is `-33.6`, it
wins only `1/5` event seeds, and the clustered interval is
`[-103.8,+36.6]`. Thus the retrospective `+32.7` advantage over this
controller does not reproduce.

The failure is specifically distillation, not missing adaptation headroom.
The learned median teacher beats robust `719` by `+58.5` on `4/5` seeds with
clustered interval `[+5.0,+112.1]`; the oracle median teacher beats it by
`+111.9` on `5/5` with interval `[+72.0,+151.8]`. Robust `719` remains the
strongest individual robust controller on the fresh split. The student instead
trails the learned teacher by `-92.1` on all five event seeds, interval
`[-180.3,-4.0]`, despite accurate and fast mode inference.

The defensible result is now: causal adaptation plus median controller
aggregation is effective on the persistent actuator-polarity benchmark, but
the current supervised/DAgger student does not reliably compress that behavior
into one policy. These confirmation seeds are sealed. Further work must target
closed-loop, return-aware teacher-to-student compression on development splits
rather than retuning the environment, estimator, or confirmation threshold.

## Closed-loop policy-compression v2 registered (2026-08-01)

The next development sweep changes only deployable policy compression. The
environment, combined-ten median teacher, frozen causal estimator, robust
controllers, and strict audit remain fixed. It separates three explanations
for the v1 failure: insufficient network capacity (`wide_dagger`), conditional
head interference (`mode_heads`), and offline-loss/control mismatch
(`mode_heads_return`). Each uses four broader DAgger rounds and three fresh
student initializations.

Unlike v1, every phase is selected on independent closed-loop switching return,
with a termination penalty. The return-aware arm weights student-visited
samples by matched teacher-student regret, post-switch age, and action
disagreement. Phase-level immutable checkpoints preserve migration/resume
safety. The graph is nine GPU producers, 27 file-gated CPU audits, and one
aggregate; the five failed-confirmation event seeds remain sealed. Full
registration is in
`reports/regime_polarity_policy_distillation_control_v2_protocol_2026-08-01.md`.

The 37-task graph was submitted atomically as `t64744-t64780`: training
`t64744-t64752`, audits `t64753-t64779`, and aggregate `t64780`. The first GPU
producer launched on `local`; later stages remain file-gated and scheduler
managed.

When both `jtl311linux` GPUs became free, the queued v2 producers and submitter
were expanded to allow that node without hard-pinning it. All resume and
reroute semantics remain unchanged.

## Closed-loop policy-compression v2 result (2026-08-02)

All 37 tasks `t64744-t64780` completed and synced: nine GPU students, 27
strict CPU audits, and one aggregate. `mode_heads` is the only passing variant.
It passes with `3/3` student initializations, while `wide_dagger` passes only
`2/3` and its control-selected seed fails; `mode_heads_return` passes only
`1/3`. Thus broader capacity and the tested sample-level return weighting do
not solve the instability. Separating posterior-conditioned action heads does.

The candidate was fixed before development-audit inspection by the independent
control-validation score: `mode_heads/student_seed_1811`. On three development
events it scores `1932.7`, versus the ten-controller robust population at
`1174.3` and fixed robust `final_seed_719` at `1908.4`. The deltas are `+758.4`
and `+24.3`; it wins `2/3` events against robust 719, recovers `100.7%` of
learned-teacher headroom, and has zero switching termination. This establishes
a stable compression mechanism, but the valid margin over the strongest robust
controller is only about `1.3%` and is not yet confirmatory.

The candidate manifest, parameters, and development selection analysis are now
hash-frozen. A new untouched five-event confirmation uses seeds
`102019,102043,102069,102103,102151`; it does not reuse the earlier sealed
confirmation. CPU audits are `t64877-t64881`, with file-gated aggregate
`t64882`. Passing requires a positive event-clustered interval against both the
robust population and preregistered robust 719, at least `4/5` wins against
719, at least `80%` learned-teacher headroom recovery, and zero termination.

Future BAPR GPU submissions are restricted to `jtl311linux`. `local`,
`jtl110gpu`, `jtl110gpu2`, and `node007` are reserved for other work; the
current confirmation is CPU-only.

## Frozen mode-head confirmation result (2026-08-02)

All six confirmation tasks `t64877-t64882` completed and the five immutable
audit manifests validate. The frozen `mode_heads/student_seed_1811` scores
`1963.1` on switching, versus `1177.0` for the ten-controller robust population
and `1941.1` for preregistered robust `final_seed_719`. It beats the population
by `+786.0` on `5/5` events with clustered 95% t interval
`[+721.4,+850.7]`, recovers `116.9%` of learned-teacher headroom, and has zero
termination.

The overall confirmation nevertheless fails exactly one frozen gate. The
student beats robust 719 by `+22.0` (`+1.1%`) and wins `4/5` events, but its
event-clustered interval is `[-41.3,+85.3]`, not strictly positive. These event
seeds cannot be extended, reweighted, or reused to select another student.
Consequently no compute-matched SAC/ESCP/RE-SAC claim sweep is authorized from
this result.

The failure is no longer attributable to online inference. Student posterior
mode accuracy is `0.9995`, median switch delay is `6.4` steps, and the student
exceeds its learned median teacher by `+113.9`. The true-mode oracle median is
`2061.5`, only `+120.4` (`+6.2%`) over robust 719. Thus the remaining signal is
small relative to event variation and robust-controller seed variation: the
population mean is only `1177.0`, while one robust seed reaches `1941.1`.
The mode-head student is a strong and stable policy-compression/robustness
result, but this confirmation does not establish a statistically reliable
online-adaptation advantage over the strongest robust controller.

Do not continue compression training on this benchmark. A permissible next
analysis is a checkpoint-only context ablation on a separate diagnostic split
(`learned`, `oracle`, `uniform`, and fixed contexts) to determine whether the
student's remaining strength is genuinely posterior-dependent or mostly a
better robust policy. That diagnostic may explain the result but cannot reopen
the failed confirmation or select another model.

## Frozen mode-head context ablation launched (2026-08-02)

The permitted checkpoint-only mechanism diagnostic is registered on five new
event seeds `102301,102331,102367,102397,102451`. Each paired event compares
robust `final_seed_719`, oracle/learned median teachers, and the same frozen
student under causal learned, true-mode oracle, uniform, fixed `0-3`, cyclically
wrong, and shuffled contexts. No policy, estimator, threshold, or environment
parameter is updated.

CPU audits are `t64890-t64894`; file-gated aggregate is `t64895`. All producer
tasks launched on `node004` with `vram=0`. The diagnostic asks whether dynamic
oracle and causal learned context each beat uniform and every fixed context
with positive event-clustered intervals, and whether the matching fixed head
is stationary-optimal. Its outcome is explanatory only and cannot reverse or
extend the failed v2 confirmation. No BAPR GPU task was submitted.

## Frozen mode-head context ablation result (2026-08-02)

All six tasks `t64890-t64895` completed. The five immutable audit manifests,
the frozen student and estimator records, and the aggregate schema validate.
The diagnostic classifies the mechanism as
`causal_posterior_realizes_context_value`: the mode-head student is genuinely
context-dependent, not merely a stronger unconditional robust policy.

On five new diagnostic event seeds, robust `final_seed_719` scores `1933.9`,
the same student scores `2046.7` under true-mode oracle context and `1916.3`
under the causal learned posterior. Oracle context beats robust 719 by `+112.8`
on `5/5` events with clustered 95% interval `[+75.7,+149.9]`. Learned context
trails robust 719 by `-17.6`, wins `2/5`, and has interval `[-108.2,+73.0]`.

Both oracle and learned contexts beat uniform and every fixed context with
strictly positive intervals. The matching fixed head is stationary-optimal in
all `4/4` modes; wrong, fixed, uniform, cyclic, and shuffled contexts cause
large return collapse. Compression is also effectively solved: oracle
student/teacher returns are `2046.7/2053.6`, and learned student/teacher returns
are `1916.3/1895.8`.

The unresolved bottleneck is switch-local causal inference. Learned context
trails oracle by `-130.4` on all `5/5` events, interval
`[-236.7,-24.0]`, even though mode accuracy is `0.9984` and median/P90 switch
delay is only `9.4/16.6` actions. The oracle advantage over the strongest
robust controller is only `5.8%`, so transient post-switch mismatch is larger
than the available headroom. This explains why BAPR can contain a real adaptive
mechanism yet fail the superiority gate against a lucky, very strong robust
seed.

The prior confirmation remains failed and sealed. This diagnostic cannot
reopen model selection or authorize a SAC/ESCP/RE-SAC claim sweep. Do not spend
more compute on policy compression or fixed-context heads in this benchmark.
Any future algorithm branch must use new development splits and target the
post-switch transient directly, for example by training on stale/soft beliefs
and using an explicitly robust transition behavior until posterior evidence is
decisive. No further task was submitted from this diagnostic.

## Causal transient fallback launched (2026-08-02)

The next branch changes only switch-local deployment. The frozen
`mode_heads/student_seed_1811`, robust `final_seed_719`, causal estimator,
teacher bundles, environment, and dwell schedule remain unchanged. A causal
state machine starts in robust fallback, enters fallback when one-step mode
likelihood contradicts the currently believed mode, and returns to the
mode-head student only after posterior confidence and evidence remain stable.
It has no access to physical mode, actuator gain, executed action, switch
clock, or future transitions.

Four new development events select one of eight preregistered
threshold/stability configurations. Five disjoint independent events then
evaluate only that selected configuration against robust 719, unchanged causal
student, and true-mode oracle student. Stale/soft-belief training is authorized
only if fallback has a positive event-clustered interval against both learned
student and robust 719, wins at least `4/5`, recovers at least half of oracle
headroom, uses at most `20%` fallback actions, and has zero termination.

The 11 checkpoint-only CPU tasks were submitted atomically as `t64913-t64923`:
screens `t64913-t64916`, selection `t64917`, independent audits
`t64918-t64922`, and aggregate `t64923`. Every task has `vram=0` and is limited
to `node001-node006`; no GPU was claimed. Later stages are file-gated and
cannot launch before their immutable predecessor manifests exist. Full
registration is in
`reports/regime_polarity_policy_distillation_transient_fallback_v1_protocol_2026-08-02.md`.

## Causal transient fallback result (2026-08-02)

All tasks `t64913-t64923` completed and all frozen hashes and manifests
validate. Development selected `evidence_1p0_k1`: evidence threshold `1.0`,
entry/exit confidence `0.60/0.90`, and one stable transition before returning
to the adaptive student.

On five untouched audit events, the selected fallback scores `2035.6`, versus
`1929.6` for the unchanged causal student, `1919.2` for robust
`final_seed_719`, and `2032.9` for the true-mode oracle student. It beats both
the learned student and robust 719 on `5/5` events. The deltas and clustered
95% intervals are `+106.0 [+67.4,+144.7]` and
`+116.4 [+64.1,+168.7]`, respectively. Its difference from the true-mode
oracle is only `+2.7 [-41.0,+46.4]`. It recovers `102.4%` of oracle headroom,
uses fallback for just `1.088%` of actions, and has zero termination.

This identifies the failure precisely: mode-conditioned control and causal
inference work during persistent regimes, but a short stale-context interval
after each switch is larger than the available oracle headroom. A sparse,
strictly causal robust fallback removes that loss without changing the
environment, policy, or estimator. The continuation gate passes, but this is
still a single student initialization. Before any stale/soft-belief retraining
or external-baseline sweep, the frozen fallback must reproduce across the two
other existing `mode_heads` initializations on a disjoint event split.

## Frozen fallback cross-student audit launched (2026-08-02)

The passing `evidence_1p0_k1` configuration is now frozen without reselection
and applied to all three existing `mode_heads` initializations
(`1709,1811,1901`). Five new event seeds
`104301,104331,104367,104399,104451` produce a fully paired 3-by-5 audit.
Every student must independently beat both its unchanged learned arm and
robust `final_seed_719` with at least `4/5` event wins and a positive clustered
interval, recover at least half of its oracle headroom, use at most 20%
fallback actions, and terminate zero episodes. The overall gate is strict
`3/3`; no failing student may be removed.

Scheduler tasks `t64930-t64944` are the 15 CPU audits and file-gated `t64945`
is the aggregate. All have `vram=0` and are restricted to
`node001-node006`; no BAPR GPU task was submitted. Full registration is in
`reports/regime_polarity_policy_distillation_transient_fallback_cross_student_v1_protocol_2026-08-02.md`.

## Frozen fallback cross-student result (2026-08-02)

All tasks `t64930-t64945` completed without retry, and all 15 audit manifests
validate. The strict preregistered gate fails with only `1/3` students passing.
Seed 1709 passes every gate. Seed 1811 misses only the oracle-recovery floor
(`46.5%` versus `50%`). Seed 1901 recovers `63.6%` but misses the positive
clustered interval against its unstable learned arm; its five event deltas are
all positive, while one `+365.8` recovery inflates the event-level variance.

The formal failure must not be relabeled as a pass. However, the mechanism is
not initialization-specific in the practical comparison that matters most:
fallback beats robust `final_seed_719` by `+88.4`, `+78.5`, and `+79.3` for
student seeds 1709, 1811, and 1901. Every comparison wins `5/5` events and each
clustered interval is strictly positive, giving `15/15` positive
student-event clusters. Mean fallback use is only about `1.15%`, with zero
termination. Fallback also beats every learned-student event, although seed
1901's preregistered interval is too wide.

Thus explicit causal fallback is a reproducible improvement over the strongest
frozen robust controller, but the stronger claim of uniformly recovering half
of zero-delay oracle headroom is not established. Stale/soft-belief training
remains blocked. The next experiment is a checkpoint-only delayed-oracle
ladder on a new split to measure how much of the remaining gap is caused by the
unobservable first post-switch action versus reducible estimator/gate delay.

## Transient causal-ceiling diagnostic launched (2026-08-02)

The three students, estimator, robust 719, fallback configuration, environment,
and failed cross-student analysis are hash-frozen. Five new events
`105301,105331,105367,105399,105451` compare zero-delay oracle, oracle delayed
by `1,2,5,10` actions after each hidden switch, learned context, robust 719,
and the deployable fallback. Delay 1 is the relevant causal ceiling: it uses
the old head for the first unobservable post-switch action and reveals the new
mode only after that transition.

Transient training is reauthorized only if `delay1 - fallback` has at least
`4/5` wins, a positive clustered interval, and at least 2% robust-return
margin for `2/3` students, with zero termination throughout. Scheduler tasks
`t64954-t64968` are 15 CPU audits and `t64969` is the file-gated aggregate.
All have `vram=0` and are restricted to `node001-node006`; no GPU task was
submitted. Full registration is in
`reports/regime_polarity_policy_distillation_transient_causal_ceiling_v1_protocol_2026-08-02.md`.

## Transient causal-ceiling diagnostic result (2026-08-02)

All tasks `t64954-t64969` completed and the 15 immutable audit manifests
validate. The delayed-oracle ladder gives a clean causal bound. Delay 1 retains
`87.6%-98.5%` of zero-delay oracle headroom, while delay 5 remains above robust
719 and delay 10 is already below it. The adaptive benefit is therefore
switch-latency-sensitive, but the unavoidable first post-switch action is not
the main remaining failure.

The frozen `evidence_1p0_k1` fallback already recovers `64.8%-78.9%` of the
delay-1 causal headroom. It beats robust 719 for all three student
initializations by `+118.4`, `+91.9`, and `+89.2`; all three comparisons win
`5/5` events and have positive clustered intervals, with zero termination.

Only seed `1901` has a material paired `delay1 - fallback` gap under the frozen
rule. Seeds `1709` and `1811` have positive means but intervals crossing zero.
The preregistered authorization result is therefore `1/3`, not the required
`2/3`, and `authorize_transient_training=False`.

This closes the stale/soft-belief training branch. Freeze the explicit causal
fallback as the deployable BAPR mechanism; do not add another student,
threshold sweep, or posterior-training variant on this benchmark. Before a
final baseline comparison, consolidate the exact audited fallback into the
runtime path and verify behavioral identity. No new GPU or CPU task was
submitted from this result.

The audited state machine has now been consolidated in
`jax_experiments/common/causal_fallback.py`. The experiment protocol re-exports
the same pure update function, while `CausalFallbackGate` supplies reset and
checkpoint-state support for runtime use. Direct behavioral tests, compilation,
and a frozen aggregate rerun pass. This threshold remains estimator-specific:
it is valid for the frozen expected-action inverse-system-ID stack and is not
silently reused with legacy `BAPRRegime` likelihoods.

## Independent final baseline comparison launched (2026-08-02)

The deployable stack is frozen as `mode_heads` plus the expected-action
estimator, `evidence_1p0_k1` causal fallback, and robust seed 719. Five new
student initializations and independently trained SAC/ESCP/RE-SAC baselines use
registered seeds `2009,2113,2213,2311,2417`; evaluation uses five untouched
event seeds. Every baseline receives 5.6M transitions and 350k updates. RE-SAC
keeps the positive `weight_reg=beta_ood=0.01` sign and `1e-5` learning rate.

The final scheduler graph is `t65066-t65106`: 20 GPU training tasks run only on
`jtl311linux`, 20 audits are file-gated to `node001-node006`, and one aggregate
waits for all audit manifests. The first wave is running at three tasks per
GPU. Earlier staging-preflight records `t64981-t65021` never entered valid
training, and `t65022-t65062` were cancelled while queued; neither lineage is
part of the comparison. The full frozen claim gate and compute-disclosure are
in `reports/regime_polarity_fallback_final_comparison_v1_protocol_2026-08-02.md`.

## Independent final baseline comparison result (2026-08-03)

The complete graph `t65066-t65106` finished with 5/5 student models, 15/15
baseline bundles, 20/20 strict audits, and a valid aggregate. BAPR scores
`2015.3 +/- 32.1` switching and `2118.7 +/- 14.5` stationary, with zero
termination and `1.136%` fallback use. SAC scores `1222.5 +/- 772.4`; ESCP
scores `1858.5 +/- 710.6`; the RE-SAC arm collapses to `-375.9 +/- 69.2` and is
flagged as a reproduction failure.

The frozen primary gate fails. Against the stronger SAC/ESCP result in each
registered seed slot, BAPR's deltas are `-402.2,+534.6,-661.4,+437.4,-502.8`:
only `2/5` wins, mean `-118.9`, conservative 95% interval
`[-817.3,+579.5]`. Stationary retention passes at `95.5%` and termination is
tied, but these do not compensate for the return gate. Pairwise, BAPR beats SAC
on mean by `+792.8` with `4/5` wins and ESCP on mean by `+156.9` with `3/5`
wins; both uncertainty intervals cross zero.

This result rules out a general performance-superiority claim on the frozen
HalfCheetah polarity benchmark. The fallback/inference mechanism is no longer
the main bottleneck: it is sparse, causal, and non-catastrophic. The distilled
controller is consistently capped around 2k while independent SAC/ESCP runs
occasionally reach 2.4k-2.7k. Its low seed variance is conditional on shared
teacher data and cannot be presented as lower end-to-end RL training variance.

The negative RE-SAC row is also not evidence for BAPR. Although the run itself
matches its registered `1e-5`, `5.6M` transition, `350k` update configuration,
the local MuJoCo RE-SAC reference uses `3e-4`, approximately 1:1 update/data,
`beta_bc=0.001`, and critic/actor ratio 2. Reproducing that baseline requires a
separate protocol; it cannot alter the failed final gate. No post-hoc task was
submitted.

## Baseline-fidelity and independent-source diagnostics launched (2026-08-03)

Two isolated diagnostics now address the remaining ambiguity without tuning a
new BAPR estimator. First, the JAX RE-SAC path now implements the local
reference trainer's 10k pre-iteration random warmup, `3e-4` learning rate,
one update per transition, `beta_bc=0.001`, critic/actor ratio 2, clip norm 1,
and the intentional positive `weight_reg=0.01`. Actor cadence is derived from
checkpointed update count. A local end-to-end smoke verified the warmup
checkpoint and exact step/update accounting. The two-seed HalfCheetah/Ant
paper-fidelity graph is `t67092-t67116`; it cannot change the sealed final
comparison and only decides whether a fresh five-seed baseline reproduction is
warranted.

Second, `t67077-t67091` trains equal-budget robust SAC, ESCP, and four fully
independent fixed-mode SAC specialists on fresh development seeds. A privileged
true-mode oracle then switches among those independent controllers. New BAPR
training is authorized only if that optimistic oracle exceeds every
single-controller alternative by at least 10%, is diagonally optimal in at
least 3/4 modes, and passes on both seeds. This separates lack of benchmark
headroom from estimator/controller failure before any further algorithm
iteration.

All 24 GPU producers are restricted to `jtl311linux`; the 16 CPU audits and
aggregates are restricted to `node001-node006` and file-gated. The first six
source tasks are running three per GPU and have valid iteration-0 checkpoints.
No task uses Slurm, auto-adopt, another GPU node, or a learned estimator.

## Independent source-controller headroom result (2026-08-08)

The complete graph `t67077-t67091` finished and all source bundles and audit
manifests validate. The optimistic true-mode oracle is strongly useful on
average: for seeds 4021 and 4049 it improves switching return over the strongest
single controller by `+69.0%` and `+71.3%`, respectively, and the matching
specialist is stationary-optimal in all `4/4` modes for both seeds.

The preregistered gate nevertheless fails because seed 4021 improves the
worst stationary mode by only `+7.2%`, below the frozen `10%` floor. Seed 4049
passes every check, including `+40.0%` worst-mode headroom. The result is
therefore `1/2`, not a pass. It shows substantial mode-specialization space,
but not the required reproducible worst-mode margin; estimator training remains
blocked under this protocol.

## RE-SAC paper-fidelity JAX smoke result (2026-08-08)

All 12 training bundles and the repaired 12-audit evaluation matrix completed.
The initial audit lineage failed because `final_task_sweep` assumed every
continuous-gravity controller exposed a BAPR `latent_dim`; context-free SAC,
state-context ESCP, and RE-SAC do not. The evaluator now records the physical
protocol coordinate for those controllers. Tasks `t71868-t71879` completed the
corrected audits, and aggregate retry `t71885` completed after the analyzer was
made fail-closed on non-finite returns.

The paper ordering is not recovered. HalfCheetah stationary/switching returns
are SAC `2416.0/2227.4`, RE-SAC `750.4/781.6`, while ESCP is non-finite for
both seeds and all three event streams. Ant is SAC `1928.1/2195.9`, RE-SAC
`-798.5/-1167.9`, with ESCP again non-finite throughout. Accordingly this
two-seed diagnostic is `pass=False`; it blocks a five-seed JAX baseline run and
cannot support treating the current ESCP or RE-SAC ports as scientific
baselines.

Together these diagnostics narrow the next problem. The polarity environment
does contain a large oracle specialization benefit, but the controller source
is not reproducibly strong in its worst mode. Independently, the JAX adaptive
baselines are numerically or behaviorally invalid under the paper-fidelity
configuration. The next work should diagnose ESCP's first non-finite training
update and RE-SAC's actor/critic scale against the reference implementation;
it should not tune another BAPR estimator or launch confirmation seeds yet.

## RE-SAC/ESCP numerical-semantics diagnosis (2026-08-08)

The follow-up graph `t72153-t72173`, with probe retries `t72177-t72178`, is
complete. It localizes the legacy ESCP failure to the actor update: Ant first
becomes non-finite at iteration 0/global update 881, and HalfCheetah at
iteration 13/global update 13,346, while critic loss, critic gradients, and
alpha remain finite at the boundary. The corrected finite-guard ESCP is
numerically stable, but remains a state-only JAX approximation rather than the
original recurrent `(state,last_action)` implementation.

At the sealed approximately 1M-step diagnostic budget, HalfCheetah
stationary/switching returns are SAC `2416.0/2227.4`, ESCP `2102.7/1732.5`,
and RE-SAC `2175.4/2126.3`. Ant returns are SAC `1928.1/2195.9`, ESCP
`2333.2/2791.8`, and RE-SAC `2334.4/2309.5`. All corrected logs are finite;
RE-SAC's MuJoCo regularization shift is exactly zero as required by the B0
launcher. This establishes numerical interpretability, not superiority.

The released RE-SAC evidence also corrects an earlier assumption. Its existing
five-seed nonstationary artifact gives HalfCheetah SAC `4610.2 +/- 1100.2`
versus RE-SAC `5327.9 +/- 1752.9` (`+15.6%`, 4/5 paired wins), but Ant SAC
`3912.0 +/- 972.3` versus RE-SAC `3866.4 +/- 637.9` (`-1.2%`, 2/5 wins).
The Ant claim is improved worst-quartile return (`1093` to `1208`), not a
higher mean. The paper text, released launcher, and archived data do not all
describe the same hyperparameters; the launcher and completed artifacts are
the operational reference.

## Fresh released-B0 confirmation launched (2026-08-08)

The registered graph `t72958-t72998` contains 20 independent GPU training
tasks, 20 file-gated CPU audits, and one aggregate. It uses five fresh seeds,
8M environment steps, 499,500 updates, continuous gravity with
`exp(U[-3,3])`, the released 20k/4k mode clock, and environment-specific B0
controls. The first 40 training gravity tasks match the released repository
byte-for-byte. All 20 GPU tasks are running across `jtl311linux`, `node007`,
`jtl110gpu`, and `jtl110gpu2`; no task is pinned and checkpoint resume is
managed independently per seed. CPU audits cannot launch until their own
training bundle is complete, and the aggregate waits for all 20 audit
manifests.

The confirmatory report will compare stationary held-out mean, held-out worst
quartile, switching return, termination, and paired seed wins. ESCP is excluded
from this table until its recurrent history encoder is reproduced; mixing the
state-only approximation into an artifact-fidelity table would be misleading.

## Fresh released-B0 confirmation result (2026-08-09)

The graph `t72958-t72998` is complete with 20/20 training bundles, 20/20 strict
audits, and a valid aggregate. Every arm reached 8M environment steps and
499,500 updates. Four interrupted RE-SAC producers exposed a Flax checkpoint
metadata mismatch in the custom policy anchor; the loader was repaired and
tasks `t74441-t74444` resumed from their retained checkpoints. Their completed
bundles then released the dependent CPU audits without bypassing file gates.

On HalfCheetah, SAC versus RE-SAC is `1216.8 +/- 620.8` versus
`925.4 +/- 330.3` for stationary OOD, `-27.3 +/- 223.7` versus
`250.7 +/- 418.8` for worst quartile, and `225.0 +/- 134.5` versus
`348.5 +/- 323.9` for switching. Paired RE-SAC wins are `1/5`, `4/5`, and
`3/5`, respectively. Thus RE-SAC improves the robust tail and switching but
does not improve HalfCheetah mean OOD return.

On Ant, SAC versus RE-SAC is `656.7 +/- 48.8` versus `722.1 +/- 30.3` for
stationary OOD, `177.8 +/- 46.2` versus `278.6 +/- 79.5` for worst quartile,
and `302.7 +/- 153.2` versus `511.1 +/- 195.5` for switching. Paired RE-SAC
wins are `4/5`, `5/5`, and `5/5`. This is coherent positive evidence for
RE-SAC under the strict held-out protocol, especially for tail robustness and
switch recovery, but two environments are insufficient for a general dominance
claim.

The generated source of record is
`jax_experiments/results_resac_artifact_confirmation_analysis_v3/analysis.md`.
The legacy released scores use a different task/evaluation stream and remain a
provenance cross-check only. ESCP remains outside the confirmatory table until
the original recurrent history encoder is implemented.

## Recurrent ESCP corrected-baseline comparison launched (2026-08-09)

The original ESCP causal probe has now been reproduced as
`(state, previous_action) -> FC128 -> GRU64 -> context`, with reset-aware
history, timing-RMDM prototypes, released learning rates, twin-min SAC, and a
strict deterministic recurrent evaluation path. A Flax 0.10.6/0.12.6 NNX
state-layout difference was caught by the first scheduler run and fixed by
carrying the GRU's non-parameter RNG state through every recurrent graph
merge. Remote smoke `t76128` completed collection, update, evaluation, and
checkpointing on node007.

The sealed actuator-polarity graph compares frozen BAPR and SAC results with
five fresh recurrent-ESCP and five released-B0 RE-SAC runs at the same
single-controller budget of 5.6M steps and 350k updates. Current producers are
`t76144-t76153`; file-gated audits are `t76096-t76105`, and aggregate
`t76106`. The registration SHA-256 is
`5ef7fe17ba3003c562befe08448130364c49599358105b822b88f310472cc426`.
No performance claim is made while this graph is in progress.

## Recurrent ESCP corrected-baseline result (2026-08-09)

All ten strict audits completed. Original aggregate `t76106` could not launch
because its `stage_input_paths` incorrectly contained ten individual JSON
files, while scheduleurm accepts directories only. It was cancelled and
replaced by metadata-only staging fix `t79000`, which completed on node001.
The analyzer and registered source remained unchanged. A targeted scheduler
sync was required because 1,646 older completed results preceded it in the
global result-sync backlog.

Switching return is BAPR `2015.3 +/- 32.1`, recurrent ESCP
`1633.8 +/- 407.4`, released-B0 RE-SAC `1568.4 +/- 316.5`, and SAC
`1222.5 +/- 772.4`. Stationary return is respectively `2118.7 +/- 14.5`,
`1657.4 +/- 538.2`, `1640.0 +/- 344.5`, and `1257.6 +/- 766.4`; no method
terminates during the strict switching audit.

BAPR exceeds recurrent ESCP by `381.6` (`+23.4%`, `5/5` registered seed-slot
wins), RE-SAC by `446.9` (`+28.5%`, `5/5`), and SAC by `792.8` (`+64.8%`,
`4/5`). However, the preregistered per-seed strongest-corrected-baseline gate
is only `+93.5` (`+4.9%`, `4/5`) with conservative 95% interval
`[-298.0, 485.0]`, so `primary_pass=False`. This supports BAPR as the highest
mean method on the frozen actuator-polarity benchmark, but not a statistically
robust general superiority claim under the registered conservative gate.

The source-of-record outputs are
`jax_experiments/results_regime_polarity_corrected_baseline_analysis_v2/analysis.{json,md}`.
Remote and local reaggregation hashes match exactly: JSON
`8f20fbba708365ee3a27f3e02f34eaacbfc9009877a1f8471307cee94fbcb290`,
Markdown `868923cf217b577067be35fafb87cb58ab7e70c03033bc2f22b1055ceb9b1c32`.

## Frozen mechanism and deployment-cost audit (2026-08-09)

The checkpoint-only final mechanism graph `t79072-t79097` is complete. Across
all five frozen students, true context improves over robust context, the causal
learned-estimator-plus-fallback path improves over the frozen robust actor, and
fallback improves over learned-only inference. The registered mechanism gate
therefore passes. The audit does not select a new model, threshold, estimator,
or environment after seeing the corrected-baseline result.

The causal path remains materially below the zero-delay true-context ceiling,
and delay-10 context fails the per-student threshold in all five cases. This
supports a causal adaptation mechanism with a real inference-latency gap; it
does not support oracle-level mode recovery.

The complete BAPR construction consumed 147.059M environment interactions and
9.403M gradient updates, versus 84M interactions and 5.25M updates for the 15
final single-controller baseline runs. The deployed stack contains 883,266
float32 parameters (3.37 MiB): 74,508 in the robust actor, 732,190 in the
causal estimator, and 76,568 in the mode-head student. This is a deployment
performance/compression result, not an end-to-end sample-efficiency result.

## Fresh ten-seed deployment confirmation (2026-08-09)

The independently preregistered graph `t79307-t79387` contains 40 GPU
producers, 40 file-gated CPU audits, and one aggregate. It compares the frozen
BAPR pipeline against SAC, recurrent ESCP, and released-B0 RE-SAC on ten new
model seeds and five new event streams. The familywise decision requires Holm
adjustment, positive simultaneous lower bounds, at least 8/10 paired wins,
stationary retention of at least 95%, and no positive termination gap for each
named baseline comparison. The graph was submitted only after the registered
mechanism gate passed. No result is available yet.

Eight initial remote BAPR producers (`t79307-t79314`) exposed an incomplete
launch-stage closure and exited before student fitting. The missing frozen
mechanism artifacts were staged without changing their bytes, and scheduler
retries `t79390-t79397` passed registration validation and entered the student
pipeline. The local producer continued uninterrupted. Nodes without the
repaired closure were removed only from queued-task placement until they can be
staged; the registered experiment and output namespaces are unchanged.

## Untouched persistent-damping oracle-headroom screen (2026-08-09)

The registered v1 jobs failed before environment construction because the
generic CLI omitted `joint_damping_fault`; no v1 checkpoint was produced. The
v2 amendment changes only that parser choice. Its 24 GPU producers
`t79247-t79270` are running, followed by 24 file-gated CPU audits
`t79271-t79294` and aggregate `t79295`. The dead v1 audit/aggregate queue
`t79127-t79151` was cancelled after v2 launched.

A runtime factory audit confirms that v2 constructs
`PersistentDampingModeEnv`: gravity is identical across modes, all four joint
damping vectors are distinct with equal log-displacement norm, and every mode
has the same 0.02 action-noise standard deviation. The generic console line
`Varying: ['gravity']` is cosmetic and does not describe the selected
stochastic-mode physics. Learned-estimator work remains gated on dynamic oracle
beating equal-budget robust by at least 10% in at least three of four
environments without an unacceptable termination penalty.

## Fresh ten-seed deployment confirmation result (2026-08-26)

The delayed BAPR student retries `t84095-t84097`, their dependent strict CPU
audits `t79348,t79349,t79352`, and aggregate `t79387` completed. All ten fresh
BAPR model seeds and all three independently trained baselines are present;
the active BAPR queue is empty for this graph.

BAPR switching return is `1981.5 +/- 28.5`, versus SAC
`1021.1 +/- 639.8`, recurrent ESCP `1849.1 +/- 383.1`, and released-B0
RE-SAC `1633.1 +/- 550.8`. BAPR passes the registered SAC comparison with
`+94.1%`, `9/10` wins, and Holm-adjusted `p=0.00165`. It exceeds ESCP by
`+7.2%` with `7/10` wins and RE-SAC by `+21.3%` with `8/10` wins, but their
simultaneous lower bounds remain negative. The familywise registered decision
is therefore `primary_pass=False`.

This is stronger than the earlier five-seed result: BAPR has the highest mean
and very low deployment variance, but its roughly 2k shared-controller ceiling
still loses to occasional 2.3k-2.5k baseline runs. The source of record is
`jax_experiments/results_regime_polarity_deployment_confirmation_analysis_v1/analysis.{json,md}`.

## Causal independent-specialist router screen launched (2026-08-29)

The next development branch targets controller ceiling rather than estimator
thresholds. It combines the frozen expected-action posterior and causal robust
fallback with the independent fixed-mode specialist banks from source seeds
`4021,4049`. Their privileged dynamic-oracle switching returns were
`2762.6/3103.9`, leaving materially more headroom than the shared mode-head
student.

The checkpoint-only graph is `t84239-t84241`: two CPU audits compare matched
robust, true-mode dynamic oracle, posterior-MAP routing with fallback, and
posterior-soft routing with fallback on new development event streams
`153101,153113,153127`; the final task is file-gated JSON aggregation. A learned
router advances only if both source seeds exceed robust by at least 10%, reach
at least 2200 switching return, recover at least 70% of stationary and switching
oracle headroom, win every event stream, and add no termination penalty. No GPU
training, Slurm, or auto-adopt is used in this screen.

## Causal independent-specialist router result (2026-08-29)

All tasks `t84239-t84241` completed with both audit manifests and the aggregate.
The independent specialists retain large privileged headroom: dynamic-oracle
switching return is `2595.2` for source seed 4021 and `2989.0` for seed 4049,
versus matched robust `1501.4/1679.6`. The deployable routers fail sharply.
Posterior-MAP plus the frozen fallback scores `984.0/1085.0`; posterior-soft
plus fallback scores `862.5/1200.7`. Both lose all `3/3` event streams to their
matched robust controller, so `router_gate_pass=False`.

This failure is not lack of specialist capacity. The expected-action estimator
was trained on controller seeds `8,16`; under independent-specialist feedback,
switching mode accuracy falls from its original roughly `99.7%` to `89-94%`.
More importantly, the one-step evidence fallback is triggered roughly
`140-210` times per episode and occupies `23-31%` of actions, instead of the
roughly `1%` fallback rate in the shared mode-head pipeline. The next diagnostic
must separate estimator distribution shift from repeated robust/specialist
handoff and gate thrashing before any GPU training is authorized.

## Independent-specialist failure decomposition launched (2026-08-29)

The switching-only graph `t84245-t84247` reuses the same source specialists and
estimator; it does not train or select a new controller. Two new development
event streams compare the failed current gate against an ungated MAP router, a
causal persistent option with two-step contradiction debounce and three-step
posterior confirmation, true-mode actions under the current gate, and a
privileged true-mode specialist reached after exactly ten robust handoff steps.

This design distinguishes three outcomes. A successful deployable persistent
option advances unchanged. A failed deployable option with a successful
true-mode handoff control identifies estimator distribution shift and calls for
specialist-trajectory estimator fitting. Failure of the privileged handoff
control identifies controller state-distribution incompatibility and calls for
switch-matched specialist training. After frozen-input staging,
`t84245,t84246` entered `running` on `node005`; aggregate `t84247` remains
correctly file-gated. No v2 performance claim is available yet.

## Independent-specialist failure decomposition result (2026-08-29)

All tasks `t84245-t84247` completed and the local source manifests revalidated.
There were no terminations in any arm. Dynamic-oracle return is
`2792.9/3001.7`, versus robust SAC `1616.5/1669.0` for source seeds
`4021/4049`. Ungated posterior-MAP routing reaches `1983.7/2213.5`, improving
over robust on both seeds despite only `88.4-90.1%` selected-mode accuracy.

The fallback mechanism is the dominant failure. Posterior-MAP plus fallback
scores only `642.9/1154.8`, and even substituting true-mode specialist actions
under the same gate scores only `810.3/1300.5`. The gate inserts robust actions
for roughly `25-28%` of the rollout and triggers about `167-184` times per
episode. A true-mode controller with exactly ten robust steps per regime reaches
`2342.6/2885.4`; the handoff is viable in absolute return but recovers only
`61.7%` of oracle headroom on source seed 4021, below the registered 70% gate.

The v2 `posterior_debounced_option` also fails at `911.3/1278.7`, but inspection
found that it still entered fallback after a single posterior argmax change.
Its contradiction debounce did not govern that branch, leaving `45-56`
triggers per episode and `24-35%` robust actions. The immutable v2 result is
therefore evidence against repeated fallback, not a valid screen of a truly
persistent option.

## Sticky independent-specialist router launched (2026-08-29)

The frozen-checkpoint successor `t84249-t84251` tests atomic persistent option
switching on three new development streams. Robust SAC is used only until the
initial mode has been supported for three transitions. Afterwards the current
specialist remains active until a new mode receives consecutive supported
evidence; switching never inserts robust actions. Confirm-3 is the registered
primary arm, while confirm-2/5 are sensitivity arms. No model is trained and no
GPU is used in this graph. `t84249,t84250` are running on `node005`; aggregate
`t84251` is correctly waiting for both audit manifests.

## Sticky independent-specialist router result (2026-08-29)

All tasks `t84249-t84251` completed and revalidated. Confirm-3 improves over
matched robust SAC on every one of the six new event streams. For source seeds
`4021/4049`, return is `2152.4/2309.9` versus robust `1550.8/1546.7`, gains of
`+38.8%/+49.3%`, with no termination and only `0.49%/0.44%` initial robust
actions. Selected-mode accuracy rises to `92.27%/93.01%`.

The registered primary gate nevertheless fails because confirm-3 recovers only
`49.6%/47.6%` of dynamic-oracle headroom; seed 4021 also falls below the fixed
2200 return threshold. Confirm-2 is higher at `2265.1/2329.5`, but it is a
sensitivity arm and does not replace the preregistered confirm-3 decision. The
result establishes that persistent atomic routing repairs fallback thrashing,
while the frozen estimator/controller distribution gap remains material.

## Specialist-trajectory expected-action estimator launched (2026-08-29)

The v5 successor leaves the confirm-3 router unchanged and initializes from the
frozen v4 expected-action estimator. Only source seed 4021 contributes gradient
data, using robust SAC, four independent specialists, dynamic oracle, and old
confirm-3 trajectories. Source seed 4049 is held out from updates and used for
filter selection and policy-bank transfer. Online inference remains restricted
to `(observation, commanded action, next observation)`.

GPU smoke `t84253` completed on `jtl311linux` with ten real transitions and one
finite update (`loss=0.404483`). The formal graph is `t84254-t84257`: one
checkpointed GPU fit, two model-gated CPU audits, and one manifest-gated
aggregate. No v5 performance claim is available yet.

## Specialist-trajectory expected-action estimator result (2026-08-29)

All tasks `t84254-t84257` completed and the strict local analyzer revalidated
the estimator identity, source-bundle identities, shared switching traces, and
full action counts. The selected filter is frozen at hazard `0.002`, posterior
decay `0.98`, evidence scale `1.0`; the confirm-3 router remains unchanged from
v3.

On the training policy bank (source seed 4021), confirm-3 returns `2584.1`
versus robust SAC `1620.9` (`+59.4%`) and recovers `99.7%` of the dynamic-oracle
headroom. On the policy bank held out from estimator updates (source seed
4049), it returns `2746.1` versus `1467.7` (`+87.1%`) and recovers `77.5%` of
oracle headroom. Both banks win all `3/3` new event streams, have zero
termination, and use robust actions for only `0.30%/0.40%` of steps.

The held-out inverse-model filter reaches `100%` switching mode accuracy, with
median/P90 switch delays of `2/4` steps. The preregistered v5 primary gate and
held-out policy-bank gate both pass. This identifies specialist-policy
distribution shift in the previous estimator as the remaining closed-loop
failure, rather than insufficient specialist capacity or a need for more gate
tuning.

The v5 estimator parameters, filter, and confirm-3 router are now frozen. The
next confirmatory graph must train entirely new robust/specialist policy banks;
none of their controller or event seeds may be used to refit or select v5.

## Frozen-v5 fresh policy-bank confirmation launched (2026-08-29)

The independent confirmation graph is `t84258-t84288`. It trains five entirely
new policy banks at controller seeds `5003,5021,5039,5051,5077`; each bank has
one matched robust SAC actor and four fixed-mode specialists at the unchanged
1,400-iteration/5.6M-transition budget. Strict switching audits use untouched
event seeds `155301,155317,155333`. The frozen v5 estimator, selected filter,
and confirm-3 router are not updated or selected in this graph.

All 25 GPU producers launched across ten available cards. The five CPU audits
are independently file-gated on their same-seed controller bundles, and the
aggregate is gated on all five audit manifests. The primary gate requires all
five banks to satisfy the original v5 per-bank criteria: at least 10% gain over
matched robust, at least 70% dynamic-oracle recovery, return at least 2200,
three of three event wins, and zero termination. No v6 result is available yet.

## Historical artifact cleanup and retention policy (2026-08-29)

One hundred obsolete top-level `results_*` and `eval_bundles_*` directories
were removed after checking them against every active `t84258-t84288`
`ckpt_dir`, `result_dir`, and `wait_for_files` path; the overlap was zero. The
deleted working-tree artifacts occupied approximately 8.8GB. Small aggregate
`*_analysis_*` directories, this report, all source code, the frozen v5 model
and minimal audit records, and all current v6 namespaces were retained.

Ordinary `git gc --prune=now` removed 12.15GiB of unreachable loose objects
without rewriting commit history or expiring reflogs; the remaining Git object
store is a 222.22MiB pack. Filesystem free space increased from 60GB to 81GB.
The pruned v5 audit was reaggregated from its immutable event/bundle records and
still returns `validation=True, primary=True`.

For BAPR tasks submitted after `t84288`, training `ckpt_dir` is remote-only and
exists solely for failure recovery or migration. `result_dir` must point only
to logs, manifests, aggregate metrics, strict-audit outputs, or a minimal
inference bundle required by a dependent audit. Replay buffers, periodic
training checkpoints, critic/optimizer state, and full train-state histories
must not be pulled back as result artifacts. The already-running v6 graph
syncs one final eval bundle per controller because its queued audits require
it; those bundles can be deleted after the aggregate is frozen.

## Frozen-v5 fresh policy-bank confirmation result (2026-08-30)

All producers, audits, and aggregate `t84258-t84288` completed. The local
analyzer revalidated all five controller banks, frozen-v5 estimator records,
three shared switching streams per bank, and full action counts. The registered
primary decision is `primary_pass=False`: only seeds `5039` and `5051` satisfy
all per-bank gates.

Frozen v5 averages `2658.7`, versus matched robust SAC `2277.5` and dynamic
specialist oracle `2838.0`. It wins `4/5` controller seeds and `12/15` event
streams, with mean relative gain `+22.2%`, zero termination, `98.24%` selected
mode accuracy, and only `0.40%` initial robust actions. The paired mean
difference is `+381.2`, but its five-seed 95% interval is
`[-707.2,1469.6]`; this is not a statistically stable superiority result.

The failure is heterogeneous controller capacity rather than estimator
collapse. On seed `5003`, robust SAC scores `3182.9`, above even the true-mode
specialist oracle at `2796.1`, so no router over that bank has positive
headroom. Seed `5077` has only `7.7%` oracle headroom and an oracle score of
`2030.4`, below the fixed 2200 threshold. Seed `5021` has `12.4%` oracle
headroom; posterior MAP without hysteresis recovers `95.5%` of it and gains
`11.8%`, whereas confirm-3 recovers `66.9%` and gains `8.3%`. Across all five
banks the estimator remains stable at roughly `98.2%` selected-mode accuracy.

No estimator refit or threshold sweep is justified. The next checkpoint-only
diagnostic is a preregistered stationary mode-by-controller matrix on the three
failed banks. If individual specialists regress below their matched robust
actor, the next trainable controller must start from an immutable robust actor
and learn mode-specific no-regression options; if specialists are sound but
only switching loses, the remaining target is switch-transient training.

## Fresh-bank stationary controller-capacity diagnosis (2026-08-30)

Checkpoint-only tasks `t84368-t84370` and aggregate `t84371` completed on three
new stationary event streams. The full controller matrix rules out a mode-label
permutation: off-diagonal specialists are consistently poor, so nominal mode
IDs are aligned with the intended fixed-mode controllers.

Seed `5003` is a genuine no-headroom bank. Robust averages `3144.5`, while the
diagonal specialist oracle averages `2528.9` (`-19.6%`) and loses all `4/4`
modes. Seed `5021` is a mixed-capacity bank: specialists win modes 0 and 2 but
lose modes 1 and 3, producing `+17.2%` mean diagonal headroom but only `2/4`
diagonal wins. Seed `5077` has usable adaptation capacity: the diagonal oracle
gains `+21.1%`, wins `3/4` modes, and has zero termination, yet frozen-v5
switching gains only `+4.2%`; this bank is specifically a switching/transient
failure.

The fresh-bank failure is therefore not one scalar gate or estimator defect.
Always routing a predicted mode to its matching specialist is itself wrong when
that specialist is weaker than robust. Before training new controllers, the
next audit will freeze a robust-inclusive per-mode utility map on the stationary
calibration streams and evaluate it on untouched switching streams. This is a
checkpoint-only upper-bound test. If it cannot preserve robust performance on
no-headroom banks while recovering specialist gains on positive-headroom banks,
the next trainable design is an immutable robust actor plus bounded mode options
with an explicit per-mode no-regression objective.

## Robust-inclusive safe-utility holdout result (2026-08-30)

Checkpoint-only audits `t84382-t84386` and aggregate `t84387` completed on five
fresh policy banks. Per-bank controller maps were frozen on stationary
calibration streams using one fixed rule: enable the matching specialist only
if it beats robust by at least 5%, wins all three calibration streams, and has
zero termination. Switching evaluation then used untouched event streams
`155501,155517,155533`.

The frozen maps are `5003=[R,R,R,R]`, `5021=[S0,R,S2,R]`,
`5039=[S0,S1,R,S3]`, `5051=[S0,S1,S2,S3]`, and
`5077=[S0,R,S2,S3]`. This resolves the heterogeneous-capacity failure rather
than hiding it: seed `5003` has no specialist headroom and therefore reproduces
robust exactly on every holdout stream, while the other four banks selectively
retain only calibrated specialist gains.

The preregistered confirm-3 primary remains `False` because seed `5077`
recovers `67.2%` rather than the required 70% of safe-oracle headroom. The
simpler posterior-MAP safe-utility arm passes the frozen gate on all `5/5`
banks. Its mean return is `2842.2` versus robust `2225.0`, mean per-bank gain is
`+31.9%`, mean recovery on the four positive-headroom banks is `88.0%`, and it
has zero termination. It wins `4/5` banks and `12/15` event comparisons; the
remaining bank is the deliberate exact robust fallback. On seed `5077`, MAP
gains `17.1%`, recovers `72.9%`, and wins all three holdout streams, whereas
confirm-3 loses enough switch return to miss only the recovery threshold.

Posterior-MAP plus robust-inclusive per-mode utility is now the frozen
independent-specialist candidate; no additional confirmation hysteresis or
transient policy is justified. This result is a high-performance
mixture-of-experts result, not an equal-sample-efficiency result: every bank
contains one 5.6M-transition robust controller and four independently trained
5.6M-transition specialists. Any baseline comparison must disclose this
controller-training cost. The next confirmatory graph must use entirely new
policy-bank and holdout seeds with this rule unchanged; equal-budget shared-head
and distillation failures remain part of the limitations rather than being
silently replaced by the five-controller result.

## Independent safe-utility policy-bank confirmation result (2026-08-31)

All 35 fresh GPU controllers and five joint CPU audits completed for training
seeds `61003,61021,61039,61057,61079`. The frozen utility maps are
`[S0,S1,S2,S3]`, `[S0,R,S2,S3]`, `[R,R,R,R]`, `[R,R,R,R]`, and
`[R,S1,S2,S3]`. Three calibration streams selected each map and three disjoint
switching streams evaluated it. No arm terminated.

Posterior-MAP safe utility averages `2335.0`, versus matched robust SAC
`1950.5`, recurrent ESCP `1451.9`, and released-B0 RE-SAC `1484.1`. It beats
ESCP and RE-SAC on `5/5` policy seeds and `15/15` event streams; their paired
95% difference intervals are `[+271.6,+1494.8]` and
`[+568.7,+1133.1]`. Those comparisons pass the frozen gates. The matched
robust-SAC comparison does not: the mean difference is `+384.5`, but the
paired interval is `[-469.3,+1238.4]`, with only `3/5` seed wins and `8/15`
event wins. The preregistered comparative decision is therefore `False`.

The composition decision also fails, solely on seed `61021`. Its safe oracle
has only `+10.5%` headroom over robust, while posterior-MAP gains `+1.1%` and
recovers `10.8%` of that headroom despite `99.0%` mode accuracy. Seeds `61039`
and `61057` calibrate every mode to robust and therefore have no adaptation
headroom; seeds `61003` and `61079` are genuine positives with MAP gains of
`+154.4%` and `+22.3%` and oracle recovery of `85.9%` and `75.1%`.

This rules out estimator collapse as the common explanation. Independent
controller capacity is seed-dependent, and a roughly one-percent causal mode
error can consume a narrow safe-oracle margin because routing switches between
nonlocal policies. The construction also trains five controllers per seed:
28M interactions and 1.75M updates, versus 5.6M and 350k for each baseline;
frozen estimator pretraining is additional. It is retained as a diagnostic
mixture-of-experts upper bound, not promoted as the main BAPR algorithm.

The original aggregate task `t85434` failed before analysis because the frozen
analyzer called the shared v8 utility validator before binding the v9 event
seeds, causing a lookup for old calibration seed `155401`. All audit outputs
were already complete. The final aggregate was reproduced locally from the
264KB audit JSON set after the frozen registration validated and the existing
v9 binding routine was called; no registered source, controller, checkpoint,
or result was changed. The source of record is
`jax_experiments/results_regime_polarity_safe_utility_confirmation_analysis_v9/analysis.{json,md}`.

Before any estimator retraining, the next permitted diagnostic is checkpoint
only: on unused switching streams, compare the safe oracle with stale-policy
and robust-handoff delays of `1/2/4/8` steps against posterior-MAP. If a
two-to-four-step privileged delay already removes most headroom, this policy
bank has no deployable causal margin and estimator work stops. No new task was
submitted from the failed confirmation.

## Safe-utility causal-delay result (2026-08-31)

Checkpoint-only tasks `t85444-t85449` completed on three unused switching
streams. Zero-delay safe-oracle headroom is `+243.6%`, `+14.7%`, `0.0%`,
`0.0%`, and `+29.8%` for controller seeds `61003`, `61021`, `61039`, `61057`,
and `61079`. The preregistered headroom gate therefore fails at `3/5`, below
the required `4/5`.

All three positive-headroom banks retain at least 70% of their oracle margin
under the best privileged four-step delay arm. A short causal inference delay
is therefore not the common blocker. Posterior MAP recovers the required
margin on only `2/5` banks, but estimator retraining remains unauthorized
because the controller-capacity prerequisite failed first.

Seeds `61039` and `61057` are exact robust fallbacks: their independently
calibrated maps are `[R,R,R,R]` because every matching fixed-mode specialist
loses to robust. This is not a mode-routing or delayed-oracle implementation
error. Existing training logs do not support a simple late-checkpoint collapse,
and their ordinary online `Eval` metric is not the required fixed-mode
validation measure. The next development screen must instead test full
fixed-mode controller adaptation from an immutable matched robust checkpoint,
with no estimator, gate, residual, or environment change. It may proceed only
if both failed banks obtain at least `3/4` stationary mode wins over robust.

Full details are in
`reports/regime_polarity_safe_utility_causal_delay_v10_2026-08-31.md`.

## Robust-warm-start specialist development result (2026-08-31)

All 16 fixed-mode producers, four CPU audits, and the aggregate task
`t85687-t85707` completed at the frozen endpoint: iteration `2099`, 8.4M
transitions, and 525k updates. The two development seeds were exactly the v9
failure banks `61039` and `61057`; they are not confirmation evidence.

Both initializations pass every seed-level gate. `actor_only` obtains `4/4`
held-out stationary mode wins on both seeds and safe-oracle switching gains of
`+31.1%` and `+44.7%`; `full_state` obtains `4/4` and gains `+52.3%` and
`+47.5%`. Every switching event is a win and no evaluated arm terminates. The
controller-capacity failure in v9 is therefore not intrinsic to the
actuator-polarity benchmark: starting each specialist from a matched robust
actor makes useful specialization reproducible on both failed development
banks.

The v11 switching event seeds generated different rollout noise but the same
deterministic mode trace. This does not alter the stationary result or the
within-trace robust comparison, but it means the reported `3/3` events are not
three independent switch orders. The frozen v12 confirmation addresses this
by selecting the simpler `actor_only` initialization, using five new policy
seeds, and preregistering three explicit distinct four-mode schedules. Robust
and specialist result synchronization remains policy-only; full checkpoints
and replay stay on execution nodes.

## Actor-only robust-warm-start confirmation result (2026-09-01)

The five fresh robust sources, twenty actor-only specialists, corrected CPU
audits, and aggregate completed for seeds `71003,71021,71039,71057,71079`.
The original frozen audit entry point had a three-argument/two-argument
`validate_bundle` call mismatch; registered audit-only adapter tasks
`t87383-t87387` corrected that interface without changing training, policies,
event streams, or the scientific protocol. The aggregate task `t86778` then
completed from the validated audit JSON.

The preregistered confirmation passes at `4/5` seeds. Passing seeds
`71021,71039,71057,71079` each win all `4/4` held-out stationary modes and all
three switching schedules. Seed `71003` fails the stationary gate at `1/4`,
but its robust-inclusive safe policy bank still gains `+23.5%` over robust on
switching and wins all three schedules. Across all five seeds, the safe-bank
switching gains are `+23.5%, +56.9%, +139.0%, +65.3%, +36.3%`, with mean
`+64.2%`, median `+56.9%`, and zero termination. Mode-level stationary passes
are `4/5,4/5,5/5,4/5` for modes 0-3.

This confirms that actor-only robust warm-start fixes the policy-bank capacity
failure under three explicit, distinct switch orders. It does not yet confirm
causal adaptation: the positive result uses privileged true mode. The next
registered phase must keep these banks and the frozen v5 estimator immutable,
use new switching streams, and compare posterior routing with a four-step
delayed oracle. Estimator retraining is permitted only if the delayed oracle
retains the safe-oracle margin while the frozen estimator fails to recover it.

## Frozen-v5 transfer on confirmed policy banks (2026-09-01)

Checkpoint-only audits `t87703-t87707` and aggregate `t87708` completed on the
five confirmed v12 policy banks and three new explicit switch orders. The
aggregate was reproduced exactly after all five audit manifests, frozen input
records, action counts, and distinct trace hashes validated.

All `5/5` banks retain at least 10% fresh-schedule safe-oracle headroom. The
four-step privileged robust handoff passes the causal-retention gate on `4/5`
banks, so deployable causal margin is reproducible. Frozen-v5 posterior MAP
passes the stricter gain, recovery, event-win, accuracy, and termination gate
on only `3/5`; the preregistered diagnosis is
`frozen_v5_estimator_transfer_is_limiting`, and estimator/router development
is authorized.

The result is still strongly positive against robust SAC: posterior MAP
averages `3112.5` versus `2093.9`, wins all `5/5` seeds, and has a paired mean
difference of `+1018.5` with 95% interval `[+457.6,+1579.5]`. It misses
confirmation because seeds `71003` and `71021` recover only `64.9%` and
`55.4%` of their safe-oracle margins, below the frozen 70% gate. Their mode
accuracies remain `98.6-98.7%`.

A no-write replay of seed `71021` localized `70-90` errors per 5000 actions:
most occur in the first `0-7` actions after a regime switch, with maximum
wrong-controller runs of `5-9` actions and mean wrong posterior confidence near
`0.83`. The failure is therefore switch-local control under sticky evidence,
not persistent mode ambiguity. The next checkpoint-only development screen
uses raw-evidence conflict to enter a causal robust fallback and exits only
after the filtered posterior is stable for 1, 2, or 3 consecutive transitions.

## Evidence-conflict fallback router development result (2026-09-01)

Checkpoint-only audits `t87748-t87752` completed on three further unused switch
orders. Aggregate task `t87753` completed remotely but the scheduler did not
start result synchronization (`result_sync_attempts=0`); the frozen analyzer
was therefore run locally against the five already synchronized and validated
audit JSON sets. The stored analysis equals a fresh recomputation exactly.

An initial no-margin evidence trigger was rejected before registration because
it produced `72-116` false conflicts per 1000 actions and spent `18-40%` of
actions in robust fallback. A single declared development replay separated the
one-step evidence margins: stable-regime 99th percentile `1.84`, maximum
`3.36`, while switch-local values commonly reached `3-7`. V14 therefore froze
a log-likelihood conflict margin of `3.0` before scheduler evaluation.

The registered confirm-3 candidate passes the development gate on `4/5` seeds;
confirm-1 and confirm-2 pass `3/5`. Confirm-3 averages `3257.1` versus robust
`2166.5`, wins all five seeds, and has a paired mean difference of `+1090.6`
with 95% interval `[+684.8,+1496.3]`. It recovers `90.9%` of safe-oracle
headroom on average while using robust fallback for only `2.87%` of actions.
Its per-seed wrong-specialist action fractions are `0.38-1.01%`.

This is a selected development router, not confirmation evidence. The next
checkpoint-only phase must freeze confirm-3 and evaluate it alone on new event
seeds. It must retain at least four strict seed passes and improve over plain
posterior MAP on at least four seeds with a positive paired mean difference
before any fresh-policy-bank confirmation is justified.

## Frozen conflict-fallback holdout confirmation result (2026-09-01)

Checkpoint-only audits `t87788-t87792` and aggregate `t87793` completed on
unused event seeds `170201,170217,170233`. All five manifests, fifteen event
files, frozen input records, action counts, and distinct trace hashes validate;
the remotely produced analysis equals a fresh local recomputation exactly.

The basic adaptation result remains strong. Confirm-3 averages `3246.0` versus
matched robust SAC `2195.5`, wins all `5/5` policy seeds, and has a paired mean
difference of `+1050.4` with 95% interval `[+482.1,+1618.8]`. Fresh safe-oracle
headroom is present on `5/5` seeds, the four-step privileged delay passes on
`4/5`, and confirm-3 meets the strict robust-relative gain/recovery/event-win
gate on `4/5`. No arm terminates.

The preregistered holdout confirmation nevertheless fails. Confirm-3 beats
plain posterior MAP on only `3/5` seeds, below the required `4/5`; its paired
mean advantage is `+76.7` with 95% interval `[-162.7,+316.1]`. At event level
it wins only `7/15`. Although fallback reduces the wrong-specialist action
fraction by about `0.30` percentage points on average, event return improvement
does not track either fallback usage or that error reduction. The diagnosis is
`frozen_conflict_fallback_has_no_incremental_map_value`.

The v14 router is therefore not promoted and fresh-policy-bank confirmation is
not authorized. Further confirmation-step, confidence, or conflict-margin
sweeps are closed. The next permitted algorithm work is switch-focused
estimator training evaluated through plain posterior-MAP routing: optimize
short-horizon control regret around regime changes rather than global mode
classification accuracy, while keeping the v12 controller banks, environment,
event splits, and robust-inclusive utility maps frozen.

## Switch-weighted estimator development result (2026-09-01)

Task `t87925` trained a v5-initialized inverse model with the first eight
transitions of each regime weighted by 8 and a `0.10` true-mode evidence loss.
Policy seeds `71003,71021,71039` supplied training trajectories;
`71057,71079` were held out from updates. Tasks `t87926-t87930` audited five
frozen v12 policy banks on three unused schedules, and `t87931` aggregated the
15 event files. The scheduler did not start result synchronization for the
aggregate (`result_sync_attempts=0`), so only its 21KB analysis directory was
pulled manually. The remote payload equals a fresh local recomputation.

V16 passes the preregistered development rule: safe-oracle headroom, four-step
causal retention, and the strict robust-relative v16 gate each pass `5/5`
policy seeds. V16 beats frozen-v5 MAP on `4/5` seeds and both held-out policy
banks, with means `3174.4` versus `3120.5`. The paired improvement is `+53.9`,
but its 95% interval is `[-29.8,+137.7]`; seed-level deltas are
`-39.7,+53.1,+21.3,+129.6,+105.2`.

This control result does not validate the intended evidence mechanism. V16
reduces switch-window routing accuracy on every seed, increases the
wrong-specialist action fraction on every seed, and improves only `9/15`
individual events. Event deltas are unstable even within one policy bank (for
seed `71039`: `-212.2,+57.6,+218.6`). The frozen decision rule nevertheless
authorizes a fresh-policy-bank confirmation, but v16 must remain a development
candidate rather than a paper result. Confirmation must use entirely new policy
seeds and require both paired return improvement and non-degraded switch-local
control-error metrics before promotion.

## Fresh-policy-bank estimator confirmation result (2026-09-01)

The frozen v17 DAG completed for entirely new policy seeds
`81003,81021,81039,81057,81079`: five robust sources, twenty actor-only
specialists, five CPU audits, and one aggregate. Seven specialists initially
failed on `jtl110gpu` because that node lacked the transitive historical
registration closure; seed `81039` mode 3 encountered the same issue on
`jtl110gpu2`. Only the 6.8MB immutable registration/model closure was staged,
and replacement tasks kept the original signatures. Their logs confirm resume
from iteration `1400`, 5.6M transitions rather than fresh training.

All five source manifests, twenty specialist bundles, five audit manifests,
stationary payloads, switching action counts, and distinct event traces
validate locally. The stored aggregate equals an independent recomputation
exactly. Fresh controller capacity and causal headroom reproduce: policy-bank,
safe-oracle-headroom, and four-step-delay gates each pass `5/5`. The true-mode
oracle averages `2863.9` versus robust SAC `1689.8`, a paired gain of `+1174.1`
with 95% interval `[+659.2,+1689.1]`; frozen-v5 MAP also beats robust on every
seed by `+1046.5` on average.

V16 does not confirm on the fresh policy banks. It averages `2719.7` versus
frozen-v5 MAP `2736.2`, wins only `2/5` seeds, and has paired difference
`-16.5` with 95% interval `[-88.4,+55.4]`. The strict return gate passes only
`4/5`, below the frozen `5/5` requirement. More importantly, the mechanism gate
passes `0/5`: switch-window routing accuracy declines on every seed and the
wrong-specialist action fraction increases on every seed.

The preregistered diagnosis is `v16_fails_fresh_bank_seed_win_gate`.
`estimator_confirmation_pass` is false, v16 is not promoted to the final
baseline comparison, and the executed-action inverse-estimator family is
closed. No threshold relaxation or additional variant is authorized. The
source of record is
`reports/regime_polarity_fresh_bank_estimator_confirmation_v17_2026-09-01.md`.

## Frozen-v5 final equal-policy-budget comparison (2026-09-07)

The v18 final comparison completed for fresh policy seeds
`81003,81021,81039,81057,81079`, three stationary holdout streams, and three
balanced switching holdout streams. The first CPU audit lineage completed all
calibration work but failed while constructing provenance because the frozen
entry point referenced two module-level v5 model-path aliases that did not
exist. Amendment 1 supplied only those process-local aliases, recorded the
execution amendment, and reran the unchanged registered scientific protocol.
All five amended audit manifests and the aggregate then completed without a
training retry or checkpoint transfer.

The frozen BAPR candidate clearly passes the standard-baseline gate. On
switching holdout it averages `2803.1`, versus `1691.6` for matched robust SAC,
`1512.7` for recurrent ESCP, and `1646.4` for released-B0 RE-SAC. The paired
advantages are respectively `+1111.5`, `+1290.4`, and `+1156.7`; all three 95%
intervals are positive. BAPR wins all `5/5` policy seeds and `15/15`, `15/15`,
and `14/15` event comparisons against those baselines, with zero termination.
Oracle recovery and stationary-retention gates each pass `4/5` seeds.

The strong equal-policy-budget claim does not pass. BAPR averages `2803.1`
versus `2406.7` for causal SAC5, a `+396.4` (`+16.5%`) mean advantage, but the
paired 95% interval is `[-485.3,+1278.2]`; BAPR wins `4/5` seeds and `11/15`
events. Seed `81039` is the single large reversal: BAPR scores `2052.8` versus
SAC5 `2752.4`, recovers only `68.0%` of oracle headroom, and retains `91.0%` of
the strongest stationary single-policy baseline. The preregistered diagnosis
is therefore `adaptation_supported_but_not_equal_policy_budget_advantage`.

This supports the paper claim that causal regime inference plus a specialized
policy bank substantially outperforms standard single-policy SAC, ESCP, and
RE-SAC in this persistent stochastic regime benchmark. It does not establish
that BAPR is better than spending the same policy-training budget on a generic
SAC ensemble routed by the same posterior. Further work should first diagnose
the seed-81039 policy-bank reversal; it should not tune on the final v18
holdout or weaken the registered gate.

## Fresh-seed specialist initialization stability result (2026-09-08)

The v19 graph completed for development seeds `82003,82021,82039`. Two failed
producer lineages were replaced without changing their signatures. Eight CPU
audits initially failed because the frozen audit entry point looked for a
GPU-local protocol signature even though the same registered signature was in
the staged inference bundle. Amendment 1 changed only that lookup location;
all nine audits and the aggregate then completed. The ten failed scheduler
records are superseded lineages, not missing experiment cells.

`full_state` is the strongest arm but does not pass its preregistered selection
gate. It improves safe switching over actor-only by `+262.2` on average, wins
`2/3` policy seeds and `7/9` events, passes all three robust-relative bank
cells, and raises the worst-seed robust-relative gain from `+43.3%` to
`+53.8%`. Its minimum stationary retention against actor-only is only `85.2%`,
below the frozen `95%` threshold. `critic_warmup` is closed: it wins only
`1/3` seeds and `3/9` events and reduces the worst-seed gain to `+35.2%`.

Per-mode traces localize the remaining variance. Copying critic, target critic,
and alpha frequently produces a much stronger controller immediately after
the fixed-mode fork, confirming that fresh-critic shock is real. Continued
actor updates can then erode that behavior, especially for seed `82021` modes
2 and 3. This is not evidence for changing the environment, posterior, or
router, and prior frozen-anchor/bounded-residual experiments already showed
that a locally constrained residual cannot represent actuator sign reversal.
The next fresh-seed screen therefore keeps full-controller initialization and
tests validation checkpoint selection plus a lower actor-update frequency,
under the unchanged v19 comparison gate. Full diagnosis:
`reports/regime_polarity_specialist_stability_v19_diagnostic_2026-09-08.md`.

## Full-state specialist policy-stability v20 launched (2026-09-08)

V20 freezes three new policy seeds `83003,83021,83039` and three paired arms:
the final full-state policy, a validation-selected full-state policy, and a
validation-selected policy whose actor and alpha update every second critic
step. Every arm retains the same 5.6M-transition robust source plus 2.8M
specialist interaction budget and 175k critic updates. The independent trainer
validation stream selects snapshots; calibration and reporting use disjoint
event streams. The full v19 retention and seed/event-win gate is unchanged.

The scheduler-only graph is `t89788-t89836`: three robust GPU sources, 36
file-gated specialist GPU tasks, nine file-gated CPU audits, and one aggregate.
`local` is excluded from GPU placement. The three sources launched on
`jtl311linux` GPUs 0/1 and `node007` GPU 0; CUDA initialization and the frozen
actuator-polarity configuration are present in all launch logs. Tests establish
period-1 update equivalence to `SACBase`, period-2 actor masking, best-policy
selection, and full checkpoint/resume restoration. Full registration:
`reports/regime_polarity_specialist_policy_stability_v20_preregistration_2026-09-08.md`.

## Full-state specialist policy-stability v20 result (2026-09-09)

All 36 specialist bundles, nine CPU audits, and the aggregate completed. Two
failed specialist scheduler records were superseded by unchanged-signature
replacement tasks, so no experiment cell is missing. The frozen aggregate
selected no candidate: `full_state_best` and `period2_best` trail
`full_state_final` by 860.1 and 1168.9 mean switching-return points.

Post-run code and provenance inspection found an execution deviation affecting
both `best` arms. The common trainer evaluated all four stochastic modes and
fed their pooled mean into checkpoint selection, whereas the preregistration
specified matching fixed-mode validation. Most selected snapshots therefore
come from the first post-fork evaluation and remain close to the generic robust
policy. These arms do not validly test matching-mode checkpoint selection, and
`period2_best` also does not isolate actor thinning. The registered aggregate is
not changed after the fact.

The unaffected `full_state_final` control is strong: `3/3` bank cells, `12/12`
stationary seed-mode cells, and `9/9` switching events pass; safe switching
averages 3182.3 versus 1510.8 for robust SAC. Across V19 and V20, full-state
final training beats robust SAC on all six development seeds with a mean paired
gain of 1437.8. This is sufficient development evidence to freeze that
training recipe and test it once on entirely new confirmation seeds against the
same frozen-v5 posterior and matched standard/equal-policy-budget baselines.
Detailed diagnosis:
`reports/regime_polarity_specialist_policy_stability_v20_diagnostic_2026-09-09.md`.

## Full-state final independent confirmation v21 launched (2026-09-09)

V20 closes the development phase for the unaffected `full_state_final` arm.
Across V19 and V20 it beats the matched robust source on all `6/6` policy
seeds and all `18/18` switching events, with mean safe switching return
`3070.1` versus `1632.3` and a paired gain of `+1437.8` (95% CI
`[+1030.9,+1844.7]`). The invalid V20 validation-selection arms are excluded
from this conclusion and from V21.

V21 is an independently registered five-seed confirmation of that frozen
recipe. It uses policy seeds `84003,84021,84039,84057,84079`, fresh disjoint
calibration/stationary/switching event streams, the unchanged frozen-v5
executed-action posterior, and the same persistent actuator-polarity
environment. The comparison includes matched robust SAC, recurrent ESCP,
released-B0 RE-SAC, and the equal-policy-budget causal SAC5 control. The
specialists copy actor, critic, target critic, and alpha from the robust source,
reset optimizer/replay state, train for the registered fixed-mode budget, and
use only the final policy; there is no post-hoc checkpoint selection.

The scheduler-only DAG is `t90234-t90294`: 35 first-wave GPU producers, 20
source-dependent GPU specialists, five CPU audits, and one aggregate. GPU
placement excludes `local`. After the first dispatch all 35 producers were
running across `jtl110gpu`, `jtl110gpu2`, `jtl311linux`, and `node007`; the
remaining 26 tasks were correctly dependency-queued. Task `t90250` recovered
from a transient first-attempt SSH failure by rerouting to `jtl311linux`, so no
replacement or duplicate was submitted. Result synchronization is restricted
to compact controller/provenance/audit artifacts and excludes checkpoints and
replay buffers. Full registration:
`reports/regime_polarity_full_state_final_confirmation_v21_preregistration_2026-09-09.md`.

## Full-state final independent confirmation v21 result (2026-09-09)

All five corrected CPU audits and the preregistered aggregate completed. The
audit retries were caused only by omitted runtime aliases for the reused V18
artifact-path API; amendment 2 restored those existing paths without changing
training, checkpoints, evaluation data, or decision thresholds.

The frozen BAPR arm passes every registered gate. Mean switching return is
`3197.3`, versus `1448.8` for robust SAC, `1362.9` for recurrent ESCP,
`1691.0` for released-B0 RE-SAC, and `2397.8` for the equal-policy-budget
causal SAC5 control. BAPR wins all `5/5` policy seeds and all `15/15` event
comparisons against every comparator, with paired gains and 95% intervals of
`+1748.5 [+1368.7,+2128.2]`, `+1834.4 [+1011.4,+2657.4]`,
`+1506.3 [+880.6,+2132.0]`, and `+799.4 [+199.3,+1399.5]`, respectively.
No arm has a termination increase.

The causal mechanism also passes its controls: posterior routing accuracy is
`98.65%`; mean switching oracle recovery is `91.5%` and passes on `5/5`
seeds; stationary return is `3455.7`, with retention against the strongest
single-policy comparator passing on `5/5` seeds. The preregistered diagnosis is
therefore `full_state_final_bapr_supported_at_equal_policy_budget`, and the
strong final algorithm claim is supported for this persistent
actuator-polarity benchmark. V21 is now a sealed confirmation set and must not
be used for further tuning. Full result:
`reports/regime_polarity_full_state_final_confirmation_v21_2026-09-09.md`.

## Ant frozen-recipe transfer screen v22 (2026-09-11)

The three-seed Ant transfer screen is complete after an aggregation-only API
adapter; no training or audit artifact changed. The registered gate fails.
Safe true-mode switching averages `3559.2` versus `2624.4` for matched robust
SAC (`+35.6%`, `3/3` paired seed wins), so Ant has substantial adaptation
headroom. The blocker is controller reliability: stationary specialist cells
pass only `1/4, 1/4, 2/4` modes across seeds, and switching termination remains
nonzero. This policy bank must not be used to train an estimator.

The separate bus policy-bank screen also fails at `0/3` seeds. Mean stationary
and switching gains are only `+0.59%` and `+0.30%`; estimator training is not
authorized for that bank.

## Ant matching-mode checkpoint screen v23 launched (2026-09-11)

V23 tests the remaining unexecuted policy-stability hypothesis. It retrains
only Ant modes 0/1 from the matched V22 robust controller and compares correct
matching-mode best-checkpoint selection at actor periods 1 and 2. Frozen V22
final policies are reused for modes 2/3. Calibration, stationary holdout, and
switching holdout use new disjoint event streams; no estimator or router is
trained.

The final scheduler-only DAG is `t92615-t92633`: 12 GPU producers, six
dependency-gated CPU audits, and one aggregate. All producers started on
`jtl110gpu`, `jtl110gpu2`, `jtl311linux`, or `node007`, never local, and their
logs show the corrected V23 training entry point resuming from iteration 1400
and 5.6M source transitions. Only compact policy bundles and JSON are synced.

## Ant matching-mode checkpoint screen v23 result (2026-09-11)

All 12 specialists, six Linux CPU audits, and aggregate `t92633` completed.
The original audits had been routed to the Windows-native `jtl110cpu` aliases
and exited before evaluation on the POSIX environment prefix. Recovery tasks
`t92736-t92741` ran on `node001-node006`; their compact JSON results synced
successfully, and the aggregate completed on `node006`. No training artifact
was rerun during this recovery.

Neither registered variant passes. `matching_best` averages `3642.1` safe
switching return versus `2625.1` for robust SAC (`+37.9%`), but passes only
`6/12` stationary mode cells and has `22.2%` switching termination.
`period2_matching_best` raises the mean to `3923.0` (`+49.5%`) and passes
`7/12` stationary cells, but still has `20.0%` termination. Both variants pass
`0/3` policy seeds. In particular, seed 85003 retains broad stationary
instability, while seed 85021 obtains high return with catastrophic switching
termination. Correct matching-mode checkpoint selection and period-2 actor
updates therefore do not stabilize the Ant policy bank, and no estimator or
router experiment is authorized from V23. The next controlled algorithm step
is switch-state specialist training with an explicit termination-risk
objective and an immutable robust fallback.

## Ant switch-state recovery screen v24 launched (2026-09-11)

The HalfCheetah V21 result remains a valid positive result, but it does not by
itself establish transfer to Ant. V24 tests the localized Ant failure: fixed-mode
specialists had not trained on states produced immediately after a mode switch,
and their objective did not price catastrophic termination. Each specialist now
alternates a frozen robust-policy predecessor segment with a matching target-mode
segment while preserving simulator state. Only target-mode transitions enter
replay; `switch_state_risk` additionally applies a 500-point terminal penalty.
Evaluation retains the immutable robust actor for the first eight post-switch
steps and compares against the same matched robust control.

The registered gate requires all three policy seeds to pass, at least three of
four stationary modes per seed, at least 10% switching gain, all switching events
won, and zero termination under transient fallback. The valid scheduler DAG is
`t92780-t92810`: 24 GPU specialists, six dependency-gated Linux CPU audits, and
one aggregate. Three node007 producers aborted during simultaneous first-JIT
compilation with `pthread_create` exhaustion; their checkpoint-safe retry lineage
is `t92811-t92813`, with the last retry migrated to `jtl311linux`. All 24 cells
are now running on `jtl110gpu`, `jtl311linux`, or `node007`, never local, and
logs confirm exact resume from iteration 1400 and 5.6M source transitions. The
superseded `t92749-t92779` DAG omitted the 32KB V23 authorization artifact from
launch staging and stopped at registration validation before training; it
contributes no result.

## Ant switch-state recovery screen v24 result (2026-09-11)

All 24 unique specialist cells, six CPU audits, and aggregate `t92810`
completed. Every compact bundle records final iteration 2099, next iteration
2100, 11.2M physical environment steps, and 525k updates; all frozen-fallback
equivalence checks have zero action error. The three original node007
first-JIT failures are superseded by successful same-signature retries
`t92811-t92813`, so no cell is missing.

Neither arm passes the registered gate. Plain `switch_state` raises mean safe
switching return from 2546.9 to 3849.9 (`+49.8%`) but passes only 7/12
stationary seed-mode cells and terminates in 35.6% of switching episodes.
`switch_state_risk` lowers termination to 20.0%, but also lowers safe switching
to 3287.7 (`+28.7%`) and passes only 6/12 stationary cells. Both arms pass
`0/3` policy seeds. The risk arm's effect is real but inconsistent: its best
seed reaches 6.7% termination, while another mode-specific controller collapses
despite the scalar 500-point terminal penalty.

V24 therefore confirms two facts. Ant has substantial controller-adaptation
headroom, and matching the post-switch state distribution alone does not make
the policy bank reliable. The remaining failure is catastrophic-risk control,
not posterior inference: these audits use true mode and an immutable eight-step
robust fallback. The next authorized development experiment keeps the frozen
robust actor and switch-state curriculum, removes scalar reward shaping, and
tests a learned termination-risk critic with a relative candidate-versus-robust
policy constraint. No estimator or final confirmation is authorized from V24.

## Ant constrained termination-risk screen v25 launched (2026-09-11)

V25 keeps the V24 physical interaction budget, source controllers, switch-state
curriculum, and immutable eight-step deployment fallback. It replaces terminal
reward shaping with a learned discounted termination-risk critic. The
`risk_q_absolute` arm penalizes candidate risk directly; `risk_q_relative`
penalizes only pessimistic risk above the frozen robust action at the same
state. Actor and alpha updates remain frozen for the first 12,500 risk-critic
updates. To make that action comparison identifiable, each target-mode segment
uses a registered 50/50 candidate/robust behavior mixture while preserving the
same 4,000 target-mode and 8,000 physical steps per iteration.

The valid scheduler-only DAG is `t92842-t92872`: 24 GPU specialists, six
dependency-gated Linux CPU audits, and one aggregate. A same-signature pilot
`t92841` stopped during pre-training registration validation because V23's
small analysis artifact was omitted from staging; the closure was corrected
and the registration rebuilt before any V25 checkpoint or sample existed.
Pilot retry `t92842` then resumed exactly at iteration 1400 and 5.6M source
steps, crossed the first GPU JIT, and measured about 2.54GB total process VRAM,
supporting the 3.3GB scheduler reservation. GPU placement excludes `local`, and
result synchronization includes only compact policy/provenance bundles and
JSON. If neither arm passes all three development seeds, independent Ant
specialist optimization is closed rather than followed by another penalty or
gate sweep.

## Ant constrained termination-risk screen v25 result (2026-09-11)

All 24 unique training cells, six Linux CPU audits, and aggregate `t92872`
completed. The 24 compact manifests agree on final iteration 2099, next
iteration 2100, 11.2M physical environment steps, and 525k updates. Every
frozen-fallback equivalence check passes with zero action error. The failed
pre-training pilot `t92841` is superseded by completed same-signature retry
`t92842`; it produced no training state and is not counted twice.

Neither registered objective passes. `risk_q_absolute` averages 3397.5 safe
switching return versus 2495.2 for robust SAC (`+34.9%`), passes 7/12
stationary seed-mode cells, and terminates in 13.3% of switching episodes.
`risk_q_relative` is the better arm at 3616.9 (`+44.5%`), 8/12 stationary
cells, and 11.1% switching termination, but both arms pass `0/3` policy seeds.
The relative arm's seed-level safe switching gains are `+33.8%`, `+59.7%`,
and `+39.8%`; its termination rates are 0.0%, 20.0%, and 13.3%. Seed 85039
mode 0 remains catastrophic, with 93.3% stationary termination, despite the
learned relative-risk constraint.

V25 therefore rejects learned termination-risk regularization as a way to
stabilize independently optimized Ant specialists. The risk model improves
the aggregate tradeoff, but it does not remove seed- and mode-specific policy
collapse. This is still a controller optimization failure under true-mode
evaluation, not an estimator or routing failure. Per the registered stopping
rule, the independent specialist-bank line is closed. The next authorized
experiment is one jointly trained robust-plus-mode-conditioned controller,
retaining the immutable robust fallback and the relative termination-risk
constraint; no further scalar-penalty, risk-lambda, or gate sweep is justified.

## Ant joint mode-conditioned controller v26 launched (2026-09-11)

V26 replaces the failed independent specialist bank with one shared
true-mode-conditioned actor, critic, and relative termination-risk critic. It
retains the immutable robust fallback, the eight-step post-switch fallback,
and balanced candidate/robust collection. `joint_equal_budget` receives the
same total continuation budget as one V25 arm; `joint_data_matched` receives
four times that continuation so every mode sees the same amount of unique data
as a V25 specialist. The latter is a capacity diagnosis, not an alternative
selected after seeing the result.

The first launch attempt stopped before rollout for two independent integrity
reasons: the dynamic-mode runner incorrectly passed CLI value `-1` to a
four-value enum, and the initial `1e-6` functional-equivalence threshold was
tighter than cross-GPU float32 matmul reproducibility. No attempted branch
advanced beyond the source bootstrap. The CLI flag was removed and the
registered cross-device tolerance was set to `1e-4`; actor, critic, target
critic, fallback, and alpha remain checked separately for every mode. One
jtl311linux retry exceeded that bound by only `6.8e-6` in target-critic output,
so it was rerouted without changing the frozen registration.

The valid GPU tasks are `t92903-t92908`; all exclude local, resume at iteration
1400 and 5.6M physical steps, and have entered training. Existing audits
`t92882,t92884,t92886,t92888,t92890,t92892` and aggregate `t92893` remain
dependency-gated. The adoption gate requires all three seeds to pass, at least
three of four stationary modes per seed, at least 5% per-mode gain, at least
10% switching gain, all switching events won, and zero fallback termination.
If neither budget passes, joint Ant controller optimization closes rather than
starting another posterior, gate, or risk-weight sweep.

## Ant joint mode-conditioned controller v26 result (2026-09-12)

All six valid training tasks, six Linux CPU audits, and aggregate `t92893`
completed. Equal-budget bundles end at iteration 2099, 11.2M physical steps,
and 525k updates; data-matched bundles end at iteration 4199, 28.0M physical
steps, and 1.05M updates. All six immutable-fallback checks pass with zero
action error. The earlier failed or cancelled tasks stopped before rollout and
are superseded by valid producers `t92903-t92908`.

| Variant | Safe switching | Gain over robust | Stationary cells | Termination | Seed gates |
|---|---:|---:|---:|---:|---:|
| joint equal budget | 3117.2 | +19.5% | 9/12 | 6.7% | 1/3 |
| joint data matched | 3169.3 | +21.7% | 6/12 | 15.6% | 0/3 |

Neither variant passes. Equal-budget seed 85039 is the only complete seed pass;
seed 85003 collapses in mode 1 with 100% stationary termination. Giving the
shared controller four times as much continuation data does not repair this:
stationary reliability and switching termination both worsen. V26 therefore
rules out independent-head interference and per-mode data starvation as the
main Ant explanation. Even with privileged true mode, a shared conditioned
actor still learns seed-dependent high-return but termination-prone gaits, and
the relative risk critic does not reliably constrain them.

The Ant actuator-polarity environment does contain adaptation headroom, but the
frozen BAPR controller recipe does not transfer reliably from HalfCheetah.
Under the registered stopping rule, Ant controller optimization is closed and
no estimator is trained on this bank. The paper-facing evidence boundary is a
positive five-seed HalfCheetah result from V21 plus Ant as a negative transfer
case. Hopper and Walker2d require a separately registered, milder and
survival-valid benchmark before any new algorithm comparison; they must not be
rescued by tuning on the failed full-polarity tasks.

## Hopper/Walker2d survival-valid headroom screen v27 launched (2026-09-12)

V27 tests environment capacity before any further BAPR training. It reuses the
pre-existing `structured_channel` family without outcome-driven changes: four
persistent, positive and invertible actuator-gain patterns, gain 0.45, action
noise standard deviation 0.04, and fixed 250-step dwell. Robot morphology and
physics remain fixed within an episode. Equal-budget robust zero-context and
privileged true-mode controllers each receive 5.6M environment steps and 350k
updates on Hopper-v2 and Walker2d-v2, using three new training seeds and three
independent event seeds.

The scheduler-only DAG is `t92933-t92957`: 12 GPU training tasks, 12
dependency-gated Linux CPU audits, and one aggregate. All training tasks are
running on `jtl110gpu`, `jtl110gpu2`, `jtl311linux`, or `node007`; none uses
local. Logs confirm the spring backend, CUDA execution, registered
`structured_channel` environment, 1,400 iterations, and 4,000 samples plus 250
updates per iteration. Only compact evaluation bundles are synchronized; raw
run checkpoints remain remote.

Each environment is judged independently. It passes only if the oracle improves
both safe switching return and worst-mode stationary return by at least 15%,
wins for all three training seeds on both metrics, wins at least 3/4 stationary
mode means, has no more than a five-percentage-point switching-termination
penalty, and both arms stay at or below 10% absolute stationary and switching
termination. A passing environment may receive the frozen V21 BAPR next. A
failed environment is retained as evidence and will not be retuned on these
seeds.

## Hopper/Walker2d survival-valid headroom screen v27 result (2026-09-12)

V27 fails for both environments. Hopper robust/oracle switching returns are
2424.0/2463.9 (`+1.6%`), while worst-mode stationary returns are 127.7/167.7
(`+31.3%`); only 2/4 mode means and 2/3 switching seeds improve. Walker2d
robust/oracle switching returns are 2186.7/2141.0 (`-2.1%`) and worst-mode
returns are 162.3/149.8 (`-7.7%`); only 2/4 mode means improve and only 1/3
seeds wins each paired metric.

More importantly, both robust and oracle have 100% stationary and switching
termination. All 45 switching episodes per role and environment terminate
before the first scheduled switch at step 250; stationary mean survival is only
82.6/85.0 steps for Hopper and 83.9/95.7 for Walker2d. This benchmark is not
survival-valid and cannot test causal adaptation. No BAPR transfer is
authorized, and these development seeds will not be used to retune severity.

The original CPU audits were accidentally sent to Windows workers and failed on
POSIX command syntax. Linux retries then exposed a validation-only missing
robust sentinel (`action_task_id=-1`). Runtime amendment tasks
`t93018-t93023` completed without changing evaluation data, and aggregate
`t93031` produced the final `FAIL` artifact.

## HalfCheetah action-coordinate compensation audit v28 result (2026-09-12)

V28 performs no training. It reuses the five frozen V21 HalfCheetah policy
seeds and the frozen v5 causal estimator, selects one safe reference specialist
from independent calibration events, and tests whether mode-dependent action
sign compensation can replace policy switching on new stationary and switching
event streams.

| Arm | Switching return | Stationary return |
|---|---:|---:|
| robust SAC | 1424.8 | 1442.6 |
| reference, no compensation | 513.3 | 777.4 |
| reference + true-mode compensation | 4291.8 | 4283.8 |
| reference + causal v5 compensation | 4076.2 | 4266.1 |
| V21 policy bank + causal v5 | 3246.0 | 3478.0 |
| SAC5 + causal v5 | 2301.4 | 2519.5 |

Causal compensation beats the uncompensated reference by 3563.0 switching
return (95% CI 2961.1 to 4164.9), robust SAC by 2651.4 (2201.2 to 3101.6),
and the V21 policy bank by 830.2 (669.2 to 991.3). Every comparison wins all
five policy seeds and all 15 paired switching events. The true-mode transform
is exactly trajectory-equivalent to running the reference policy in its native
mode (maximum return error 0). The causal estimator reaches 98.84% mode
accuracy, a three-step median switch delay, and recovers 91.6%-97.0% of oracle
headroom for every seed.

The registered mechanism claim therefore passes. In this actuator-polarity
benchmark, the principal loss was not mode inference but switching among
independently optimized policies whose control coordinates and gaits were not
aligned. A single policy in one reference coordinate system plus causal command
compensation is both simpler and materially stronger. This result is specific
to the invertible polarity symmetry: HalfCheetah has no health termination, and
the result does not transfer automatically to bus uncertainty or non-invertible
actuator gain loss.

The first five audit tasks failed before producing results because the frozen
protocol imported a metadata-only v5 module that did not expose
`make_estimator`. Execution amendment 1 binds that symbol to the already frozen
v5 model implementation; it changes no seed, policy, estimator parameter,
rollout, or decision threshold. Valid Linux retries are `t93046-t93049` and
`t93051`; aggregate `t93044` completed. A mistaken Windows placement and its
cancelled retry produced no scientific output and are excluded.

## Ant oracle action-coordinate compensation audit v29 result (2026-09-12)

V29 performs no training. It reuses the three frozen V22 Ant policy seeds,
selects one reference specialist per seed using new calibration events, and
compares robust SAC, the uncompensated reference, true-mode action compensation,
and the true-mode V22 specialist bank on disjoint stationary and switching
events. Unlike HalfCheetah, every Ant arm records all health terminations and
time to first termination.

| Arm | Switching return | Stationary return | Switching term. | Stationary term. |
|---|---:|---:|---:|---:|
| robust SAC | 2588.6 | 2745.3 | 11.1% | 4.4% |
| reference, no compensation | -266.3 | 24.7 | 97.8% | 59.4% |
| reference + true-mode compensation | 4510.7 | 4698.5 | 4.4% | 6.7% |
| V22 dynamic specialist oracle | 4464.4 | 4522.0 | 28.9% | 12.2% |

Action compensation has substantial Ant return headroom: it beats robust SAC by
1922.1 switching return (95% CI 191.1 to 3653.0), wins all three policy seeds
and all nine seed-event cells, and gives per-seed gains of 55.8%, 103.3%, and
62.8%. It is exactly equivalent to running the selected reference policy in its
native mode: maximum executed-signal and paired-return errors are both zero.
The specialist bank adds no reliable advantage; compensation is higher on
average, but its three-seed interval versus the bank spans zero.

The preregistered estimator-authorization gate nevertheless fails. Seed 85003
has 13.3% stationary termination and seed 85021 has 6.7% stationary plus 13.3%
switching termination; only seed 85039 satisfies the absolute zero-termination
gate. The failure is reference-policy stability across stochastic event streams,
not action-transform correctness or lack of return headroom. Calibration used
15 native episodes per specialist, yet two specialists that were termination-
free there terminated on holdout. The dynamic bank is materially less safe,
confirming that switching independently optimized Ant gaits is not the repair.

Accordingly, V29 supports action-coordinate compensation as a mechanism on Ant
but does not authorize an Ant causal-estimator experiment. The current Ant
evidence remains a safety/reliability counterexample. Any later Ant confirmation
must use new policy seeds and a preregistered reference-policy reliability
criterion; the V29 holdout cannot be reused to select a more favorable mode.
Scheduler audits `t93054-t93056` and aggregate `t93057` all completed on Linux
CPU nodes and synchronized only compact JSON artifacts.

## Ant finite-horizon paired branch-risk screen v30 result (2026-09-12)

V30 follows the safety diagnosis in `markdown/GPT_diagnosis.md` without tuning
another gate or risk penalty. From new event streams it snapshots states visited
by each seed's frozen V29 reference policy, then uses common future actuator
noise to compare full true-mode-compensated continuation against full robust-SAC
continuation over horizons 1 through 250. It records physical termination,
rescue/harm, return, and torso-health margin; horizon ends are truncations. The
old V25/V26 compact bundles omit risk-critic parameters, so this first tests the
necessary fallback-headroom premise without pulling obsolete checkpoints.

All nine source trajectories survive 1,000 steps, and only two of 576 unique
candidate continuations terminate. The compensated candidate has 0.3% pooled
250-step termination, compared with 9.9% for full robust continuation. Robust
fallback is worse for all three policy seeds and all four actuator modes, harms
9.9% of candidate-surviving pairs, and loses about 582.5 return on average. Its
extra termination begins by horizon 16. Candidate compensation remains exactly
mode invariant with zero cumulative-return error and zero termination-trace
mismatches.

The risk-model gate fails: there are zero informative seeds under the frozen
minimum-positive rule, and the proposed fallback has negative rather than
positive net rescue headroom. The Ant fallback/shield route is therefore closed;
the two rare rescued candidate failures cannot be reused to tune a classifier.
Audits `t93066-t93068` and aggregate `t93069` completed on `node004`, `node006`,
and `node002`. Only compact JSON was synchronized; no local/GPU worker, Slurm,
auto-adopt, policy update, checkpoint pull, simulator state, or trajectory array
was used.

## HalfCheetah fresh-policy canonical compensation v31 result (2026-09-13)

V31 freezes actuator-polarity mode 0 as the reference, uses five entirely new
policy seeds and three new switching event streams, and gives the compensation
path and SAC/ESCP/RE-SAC comparators the same 8.4M interactions per seed. The v5
causal estimator is unchanged. All 25 GPU bundles, five amended CPU audits, and
aggregate `t93155` completed; only compact bundles and JSON were synchronized.

| Arm | Switching return | Stationary return |
|---|---:|---:|
| causal canonical compensation | 3400.6 | 3508.5 |
| true-mode canonical compensation | 3642.5 | 3631.5 |
| equal-budget SAC | 2594.3 | 2738.2 |
| equal-budget ESCP | 1901.8 | 1963.1 |
| equal-budget RE-SAC | 1916.1 | 1957.9 |
| canonical reference without compensation | 325.4 | 623.7 |

The independent confirmation is **FAIL** under its preregistered two-sided
gate. Causal compensation beats ESCP by 1498.8 (95% CI 611.8 to 2385.7), but
its differences versus SAC and RE-SAC are 806.3 (CI -30.2 to 1642.8) and
1484.5 (CI -23.8 to 2992.8). Both comparisons win 4/5 policy seeds and 12/15
events, but the CI requirements do not pass. This outcome must not be relabeled
as a successful confirmation despite the large mean gains.

The mechanism diagnostics remain strong: exact true-mode trajectory error is
zero, v5 mode accuracy is 98.26%, median detection delay is three steps, oracle
recovery passes 5/5 seeds, and stationary retention passes 4/5. The failure is
concentrated in seed 87021, where even true-mode compensation has only 2.1%
headroom over equal-budget SAC and causal compensation trails SAC by 73.0 and
RE-SAC by 554.5. V31 therefore points to reference-controller training variance,
not mode inference or compensation correctness. Amendment 1 only repairs NNX
state loading, the canonical-reference alias, and the already registered v5
estimator binding; it changes no scientific setting or output.

## HalfCheetah prospective power confirmation v32 result (2026-09-19)

V32 freezes the V31 algorithm and uses the failed V31 result only to plan a
clean ten-seed cohort; V31 results are not pooled. Ten new policy seeds and new
stationary/switching streams test equal-budget SAC, recurrent ESCP, RE-SAC,
true-mode compensation, and frozen-v5 causal compensation. Success requires a
positive paired mean with a positive two-sided 95% CI lower bound against all
three comparators, plus at least 8/10 seed wins and 24/30 event wins.

All 50 independent GPU runs, ten CPU audits, and the aggregate completed. Causal
compensation reaches 3297.1 switching return, compared with 2287.1 for SAC,
2046.0 for ESCP, and 2360.2 for RE-SAC. Its paired advantages are respectively
1009.9 (95% CI 391.7 to 1628.2), 1251.1 (553.5 to 1948.7), and 936.9 (337.1 to
1536.7). Thus the ten-seed mean advantage is positive and statistically resolved
against every equal-budget comparator.

The preregistered overall decision remains **FAIL** because the SAC comparison
wins 7/10 policy seeds and 23/30 seed-event cells, just below the frozen 8/10
and 24/30 consistency thresholds. ESCP and RE-SAC comparisons pass at 9/10,
25/30 and 9/10, 27/30. Oracle recovery and stationary retention both pass
10/10; frozen-v5 mode accuracy is 98.55% with a three-step median delay.

The three SAC non-wins are seeds 88021, 88039, and 88179. Their true-mode oracle
headroom over SAC is only +2.3%, -2.3%, and +4.1%, while causal recovery remains
95.4%, 95.4%, and 95.5%. The limiting factor is therefore canonical-controller
training variance, not mode inference or causal compensation. V31 and V32 must
still be reported separately, and neither failed registered decision may be
relabeled as PASS.
