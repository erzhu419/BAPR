# BAPR-v3 Oracle-First Stochastic-Control Protocol

Date: 2026-07-15

## Question

Before changing the BAPR estimator again, test whether a persistent stochastic
regime creates useful controller headroom at all. The learned estimator is not
part of this phase. A privileged one-hot mode teacher must first outperform its
own frozen robust base.

## Environment semantics

Robot morphology, gravity, nominal action gain, reward, observation, horizon,
and termination remain fixed across modes. A mode lasts 500 environment steps.
Only the parameters of a per-transition exogenous actuator distribution change:

| Family | Mode 0 | Mode 1 | Mode 2 | Mode 3 |
|---|---:|---:|---:|---:|
| packet loss probability | `0.00` | `0.08` | `0.18` | `0.32` |
| burst probability / torque std | `0/0` | `.03/.25` | `.08/.45` | `.15/.70` |

A packet-loss event zeros the complete command for one simulator step. A burst
event adds a Gaussian torque impulse for one simulator step. Event draws are
independent conditional on the persistent mode. The policy command, rather than
the disturbed action, remains in replay, so this is a hidden transition-kernel
change rather than observation corruption.

Both sequential evaluation and every fused training/evaluation rollout receive
the same five disturbance parameters: action gain, Gaussian noise std, packet
loss probability, burst probability, and burst magnitude.

## Environment audit

The 4096-sample audits pass for both Ant and HalfCheetah. All four modes retain
gravity `-9.81`, action gain `1`, and Gaussian noise std `0`.

| Env | Packet loss measured | Burst event rate measured |
|---|---|---|
| HalfCheetah | `0,.073,.178,.319` | `0,.026,.075,.143` |
| Ant | `0,.083,.187,.331` | `0,.028,.076,.146` |

Audit artifacts:

- `reports/bapr_v3_oracle_headroom_env_audit_2026-07-15.json`
- `reports/bapr_v3_oracle_headroom_env_audit_ant_2026-07-15.json`

## Controller ladder

Each run uses one checkpoint and two stages:

1. iterations `0-699`: train a context-free robust SAC base;
2. iterations `700-1399`: freeze that base, copy it into a direct conditioned
   branch, and train the branch with privileged one-hot mode context.

The same checkpoint is evaluated with `CONTEXT_ROBUST` and `CONTEXT_ORACLE`,
with the advantage gate disabled for both. Configuration: 5 critics, hidden
size 256, 4000 samples and 250 updates per iteration, three stationary episodes
per mode, two switching episodes, strict 1000-step horizon, 500-step switches.
There is no learned-context claim in this phase.

## Preregistered gate

For each final score, average the last five finite evaluations and compute

`gain = (oracle - robust) / max(abs(robust), 100)`.

A disturbance family proceeds to learned inference only if `gain >= 10%` for
both stationary and switching evaluation on both Ant and HalfCheetah. A failure
on any of those four cells rejects that family. Cross-run SAC/ESCP comparisons
are not used for this gate.

Do not download remote checkpoint PKLs or expand seeds before a family passes.

## Execution

Submitted as one scheduler JSONL batch with no node restrictions, no Slurm, and
no auto-adopt:

| Task | Family | Env | Initial placement |
|---|---|---|---|
| `t33462` | packet loss | Ant | `jtl311linux:GPU0` |
| `t33463` | packet loss | HalfCheetah | `jtl311linux:GPU1` |
| `t33464` | burst torque | Ant | `local:GPU0` |
| `t33465` | burst torque | HalfCheetah | `node007:GPU0` |

All four reached `running`. Checkpoint resume is enabled and migration-safe.
Before submission, all 19 BAPR-v3 focused tests passed. A full short training
smoke completed iterations `0-3`, then loaded the checkpoint at next iteration
`4` and completed iterations `4-5` through the fused packet-loss rollout.

## Oracle result

All four tasks completed iterations `0-1399` (`5.6M` environment steps) and
produced 70 finite ladder evaluations. The table reports the preregistered mean
of the final five evaluations. Gains are relative to the robust branch in the
same checkpoint, not to a separately trained baseline.

| Family | Env | Robust stationary | Oracle stationary | Gain | Robust switching | Oracle switching | Gain |
|---|---|---:|---:|---:|---:|---:|---:|
| packet loss | Ant | 1218.2 | 2058.7 | +69.00% | 2068.3 | 2758.6 | +33.38% |
| packet loss | HalfCheetah | 1432.8 | 2079.3 | +45.12% | 1465.1 | 1614.3 | +10.18% |
| burst torque | Ant | 1324.1 | 5325.6 | +302.20% | 3363.6 | 5873.9 | +74.63% |
| burst torque | HalfCheetah | 2454.5 | 2820.8 | +14.93% | 2241.1 | 2874.3 | +28.25% |

Both families technically pass the fixed 10% gate. Burst torque is promoted
first because all four cells clear the threshold with useful margin. Packet
loss remains a passing but provisional result: its HalfCheetah switching gain
is only 0.18 percentage point above the threshold, so it is not used to choose
the estimator before stronger evidence exists. No packet-loss checkpoint PKL
was manually downloaded.

## Learned-context continuation

The promoted burst checkpoints are continued without changing their frozen
robust base, oracle teacher, critic, replay, or controller. Only a fresh causal
context model is trained from iteration `1400` through `1999`. Its transition
mean is shared across modes, while each mode has a separate empirical residual
distribution. A sticky posterior must therefore infer persistent differences
in exogenous burst statistics instead of explaining each mode with a different
conditional mean.

| Task | Env | Placement | Required resume point |
|---|---|---|---:|
| `t34535` | Ant | `local:GPU0` | `iter=1400`, `5.6M` steps |
| `t34536` | HalfCheetah | `node007:GPU0` | `iter=1400`, `5.6M` steps |

Both scheduler-only tasks are unpinned and reached `running`. Their launch logs
confirm the exact resume point and a one-time context-signature reset while
preserving controller and replay state. Once the new signature is checkpointed,
ordinary restart or migration restores the learned context instead of resetting
it again. The focused no-fixture suite passes `20/20`, including compiled BAPR
and standard-SAC stochastic-mode rollouts.

## Shared-transition estimator result

Both continuations completed iteration `1999` (`8.0M` environment steps). The
table averages the final five fixed ladder evaluations from each checkpoint.

| Env | Protocol | Robust | Oracle | Learned | Oracle gain | Learned gain |
|---|---|---:|---:|---:|---:|---:|
| Ant | stationary | 1324.1 | 6191.6 | 1488.8 | +367.59% | +12.44% |
| Ant | switching | 3208.0 | 6625.8 | 3571.8 | +106.54% | +11.34% |
| HalfCheetah | stationary | 2454.5 | 2856.8 | 2460.1 | +16.39% | +0.23% |
| HalfCheetah | switching | 2397.8 | 2417.9 | 2421.1 | +0.84% | +0.97% |

Ant still has large controller headroom, but learned context recovers only
`3.38%` of stationary and `10.65%` of switching oracle gain. HalfCheetah has
some stationary headroom but essentially no final-window switching headroom.
Its learned result is therefore not evidence for successful adaptation.

The estimator itself is the failure. Last-100 training posterior accuracy is
`0.309` on Ant and `0.301` on HalfCheetah, close to the four-mode chance level
of `0.25`; true-mode probability is `0.262/0.258`. All four learned transition
residual variances collapse to nearly the same value (`0.223-0.226` on Ant,
`0.247-0.250` on HalfCheetah), even though the posterior can be incorrectly
confident. Controller, critic, residual, and gate training flags remain zero in
the student stage, so teacher forgetting is not the explanation.

An offline audit of 200k Ant replay transitions confirms that raw forward-model
error swamps the burst signal. A clean-mode nonlinear inverse model instead
predicts executed action from `(s,s')` and separates the commanded-action
residuals: mode means are `0.161,0.162,0.172,0.201`; clean-q90 exceedance rates
are `0.100,0.104,0.132,0.221`. A 500-step Gaussian window classifies the four
modes at `0.599,0.471,0.869,1.000` (`0.735` mean). Modes 0 and 1 remain close,
but the switching ladder's mode-0/mode-3 extremes are identifiable.

## Inverse-dynamics continuation

The next estimator learns `g(s,s') -> executed action` only from clean mode-0
transitions, where executed and commanded actions agree. At inference, the
commanded-action residual supplies mode evidence. Mode likelihood and posterior
gradients are stopped through the inverse mean; mode-specific variances are
owned by empirical EMA updates. Student rollouts use the frozen robust actor so
an incorrect posterior cannot change the policy and contaminate its own future
evidence. The controller, critic, robust base, and oracle teacher remain frozen.

Focused tests pass `24/24`. A copied real Ant checkpoint resumed at
`iter=2000`, `8.0M` steps, and replay size `1,000,000`; only context reset. Its
first continuation iteration had finite loss/variance and all four controller
training flags equal to zero. A second start reported a matching context
signature and preserved the learned context rather than resetting it again.

Scheduler-only, unpinned continuation tasks:

| Task | Env | Source checkpoint | Target |
|---|---|---:|---:|
| `t35195` | Ant | iter `1999` | iter `2599` |
| `t35193` | HalfCheetah | iter `1999` | iter `2599` |

No remote checkpoint PKL was manually downloaded. The previous protocol
signature is preserved as `protocol_signature_shared_empirical.json` before the
continuation writes its new signature.

The first Ant dispatch, `t35192`, was cancelled before completing an iteration:
its target node held an older iter-1399 copy, and the scheduler incorrectly
treated checkpoint-directory presence as freshness. The scheduler now compares
each target copy with the newest scanned mtime/size and includes that source
generation in staging cache keys (`95` focused resume/launch/staging tests
pass). The training command also requires `start_iteration >= 2000`. The
unpinned replacement `t35195` staged the current checkpoint to `node007` and
its launch log confirms `iter=2000`, `8.0M` steps, and replay size `1,000,000`.

## Inverse-dynamics result

Both continuations completed iteration `2599` (`10.4M` environment steps).
The final-five in-training ladder is:

| Env | Protocol | Robust | Oracle | Learned |
|---|---|---:|---:|---:|
| Ant | stationary | 1427.7 | 4999.0 | 2299.5 |
| Ant | switching | 3247.1 | 6509.5 | 4595.0 |
| HalfCheetah | stationary | 2454.5 | 2856.8 | 2576.6 |
| HalfCheetah | switching | 2397.8 | 2417.9 | 2603.9 |

The inverse residual variances no longer collapse and remain ordered by burst
severity. Exact online four-mode accuracy in the training diagnostics remains
about `0.30`, so the final checkpoint was not accepted from this single fixed
evaluation stream. It proceeded to paired event-stream auditing.

## Final checkpoint event-stream audit

The accepted audit contains `30/30` evaluations: two environments, three
context sources (`robust`, `oracle`, `learned`), and five paired disturbance
event seeds (`1100-1500`). Every output contains four stationary task rows,
five switching episodes, and 5000 switching trace rows. Every row loads
`next_iter=2600`, `10.4M` steps. These are five event streams for one trained
policy seed (`seed=0`), not five independent training seeds.

An earlier audit batch is invalid and excluded. Node007 initially staged source
without `jax_experiments/analysis`, then reused a stale source archive; after
that was fixed, non-node007 workers loaded iter-2000 checkpoint copies while
node007 had iter-2600. Scheduler code-tar keys now include both include paths
and a source-file `(path,mtime,size)` snapshot. Resume placement now requires
the newest checkpoint generation rather than any same-path copy, and the
evaluator aborts below `--min-checkpoint-next-iter 2600`. The accepted outputs
are from corrected tasks `t35762,t35765-t35793`; no checkpoint PKL was manually
downloaded.

Means and sample SDs are across the five paired event streams. Confidence
intervals are two-sided 95% paired t intervals for `learned - robust`.

| Env | Protocol | Robust | Oracle | Learned | Learned - robust (95% CI) | Positive streams | Oracle recovery |
|---|---|---:|---:|---:|---:|---:|---:|
| Ant | stationary | 1138.8+/-297.1 | 5274.8+/-243.9 | 3897.7+/-772.7 | +2758.9 `[+1801.1,+3716.6]` | 5/5 | 66.7% |
| Ant | switching | 3255.6+/-102.8 | 6546.9+/-204.2 | 5310.8+/-337.9 | +2055.2 `[+1574.7,+2535.6]` | 5/5 | 62.4% |
| HalfCheetah | stationary | 2400.0+/-41.1 | 2719.7+/-138.3 | 2624.3+/-93.5 | +224.2 `[+95.3,+353.2]` | 5/5 | 70.1% |
| HalfCheetah | switching | 2366.5+/-49.0 | 2707.3+/-281.2 | 2570.9+/-60.7 | +204.4 `[+79.9,+328.9]` | 5/5 | 60.0% |

This is the first inverse-estimator round with a positive learned-versus-robust
paired interval in all four cells. It does not yet establish successful online
mode adaptation. The learned switching diagnostics remain weak:

| Env | Exact mode accuracy | Expected-mode correlation | Expected-mode MAE | Switch AUC |
|---|---:|---:|---:|---:|
| Ant | 0.520 | 0.222 | 1.361 | 0.486 |
| HalfCheetah | 0.489 | 0.139 | 1.431 | 0.460 |

In the switching traces, the learned posterior chooses mode 0 for 88.9% of
true-mode-0 Ant steps and 77.7% of true-mode-3 steps. HalfCheetah shows the same
pattern (`93.2%` and `86.4%`). The posterior shifts slightly with event
severity, but exact task recognition cannot explain the full return gain.

The eval information path was audited before interpreting this result.
`set_eval_task()` writes only the privileged oracle latent;
`ProbabilisticRegimeContext.policy_context()` ignores that latent in learned
mode and uses only its causal posterior. A regression test changes the oracle
mode while holding learned state fixed and requires identical learned context.
The evaluator now additionally passes an all-zero privileged latent to every
non-oracle rollout. The focused BAPR-v3 suite passes `28/28`.

## Fixed-context counterfactual

The remaining causal question is whether online evidence helps, or whether one
approximately static conditioned branch simply dominates the weak frozen base.
The preregistered control holds oracle policy context at mode `0`, `1`, `2`, or
`3` while stationary physics tasks and the 0/3 switching stream continue
normally. It uses the exact same checkpoint and event seeds as the accepted
audit. Learned context must beat the best fixed branch before the gain is
attributed to online adaptation.

Smoke `t35799` passed, after which `t35800-t35818` completed the 20 unique Ant
controls. Four first attempts (`t35802,t35807,t35810,t35811`) encountered
`LLVM ERROR: pthread_create failed` under 16 simultaneous JAX compilations;
automatic retries `t35819-t35822` completed the unchanged signatures. All
accepted rows load iter `2600`. Future controls reserve 5.2GB per task to cap
node007 at eight simultaneous compilers; this is an admission-control change,
not an evaluation change.

| Ant context | Stationary mean+/-SD | Switching mean+/-SD |
|---|---:|---:|
| robust base | 1138.8+/-297.1 | 3255.6+/-102.8 |
| dynamic oracle | 5274.8+/-243.9 | 6546.9+/-204.2 |
| learned online | 3897.7+/-772.7 | 5310.8+/-337.9 |
| fixed mode 0 | 5318.4+/-454.6 | 6256.5+/-251.0 |
| fixed mode 1 | **5655.7+/-430.4** | **6600.2+/-94.6** |
| fixed mode 2 | 5197.8+/-157.6 | 6399.4+/-144.2 |
| fixed mode 3 | 5286.5+/-477.3 | 6276.1+/-181.1 |

Learned loses to fixed mode 1 on all five event streams: stationary difference
`-1758.1`, 95% paired t CI `[-2768.3,-747.8]`; switching difference
`-1289.4`, CI `[-1708.7,-870.2]`. Against the best fixed branch selected
separately within each event stream, learned loses `-1923.2` stationary and
`-1306.5` switching, again 0/5 wins. Dynamic oracle also loses to the per-stream
best fixed policy in stationary evaluation by `-546.0`, CI
`[-910.6,-181.5]`; its switching difference is `-70.3`, with a CI crossing
zero.

The stationary cross-context matrix makes the confound explicit. Rows are the
true burst mode; columns are the fixed policy context:

| Physics mode | Context 0 | Context 1 | Context 2 | Context 3 | Best |
|---:|---:|---:|---:|---:|---:|
| 0 | 5498.4 | **6996.5** | 5479.3 | 6663.8 | 1 |
| 1 | 5827.3 | **6680.3** | 6408.4 | 5283.0 | 1 |
| 2 | 5582.1 | 5447.3 | 5312.1 | **5590.6** | 3 |
| 3 | **4365.6** | 3498.9 | 3591.6 | 3608.6 | 0 |

The semantically matching diagonal context is optimal only for mode 1. The
clean-mode policy is best in the strongest-noise mode, while mode-1 context is
best in clean physics. Thus the large robust-to-oracle gap mostly measures an
undertrained frozen base versus a later-trained direct conditioned branch; it
is not usable mode-adaptation headroom. The learned estimator partially selects
that stronger branch, which explains positive return despite weak mode metrics.

This rejects the current Ant adaptation claim.

HalfCheetah independently gives the same answer. The first bulk attempt
(`t35823-t35845`) was cancelled after dynamic scheduler backfill admitted 14
simultaneous JAX compilers and several processes exited before producing valid
outputs. The accepted rerun used one five-task mode wave at a time:
`t35846-t35850`, `t35852-t35856`, `t35857-t35861`, and `t35864-t35868`.
All 20 outputs were synchronized and validated at iter `2600`, with four task
rows, five switching episodes, and 5000 trace rows each.

| HalfCheetah context | Stationary mean+/-SD | Switching mean+/-SD |
|---|---:|---:|
| robust base | 2400.0+/-41.1 | 2366.5+/-49.0 |
| dynamic oracle | 2719.7+/-138.3 | 2707.3+/-281.2 |
| learned online | 2624.3+/-93.5 | 2570.9+/-60.7 |
| fixed mode 0 | **2831.9+/-141.1** | **2818.9+/-96.9** |
| fixed mode 1 | 2646.1+/-77.2 | 2650.6+/-130.5 |
| fixed mode 2 | 2798.7+/-105.9 | 2725.5+/-131.0 |
| fixed mode 3 | 2668.3+/-82.9 | 2481.3+/-409.3 |

Learned loses to the fixed mode-0 branch by `-207.6` stationary (95% CI
`[-450.0,+34.7]`, 1/5 positive streams) and `-248.0` switching
(`[-397.2,-98.7]`, 0/5). Against the best fixed branch selected separately for
each event stream, learned loses by `-261.7` (`[-395.4,-128.1]`) and `-295.5`
(`[-381.1,-209.9]`), with 0/5 wins in both protocols. Dynamic oracle also loses
to the per-stream best fixed branch by `-166.3` stationary
(`[-324.0,-8.6]`, 0/5); its switching difference is `-159.1`, with a wide CI
crossing zero and only 1/5 wins.

The HalfCheetah stationary cross-context matrix is likewise non-semantic:

| Physics mode | Context 0 | Context 1 | Context 2 | Context 3 | Best |
|---:|---:|---:|---:|---:|---:|
| 0 | 2803.6 | 2747.6 | **2878.4** | 2756.8 | 2 |
| 1 | **2918.9** | 2764.5 | 2785.9 | 2693.5 | 0 |
| 2 | **2924.7** | 2578.6 | 2786.9 | 2699.1 | 0 |
| 3 | 2680.4 | 2493.6 | **2743.7** | 2523.9 | 2 |

The matching diagonal is optimal in `0/4` rows. Therefore both Ant and
HalfCheetah reject the current burst-torque adaptation interpretation. The
conditioned actor is a stronger later-trained controller, but its context
coordinates do not encode mode-specific optimal control. The frozen robust
branch received 700 actor-update iterations; the warm-started direct branch
then received another 700 while the base stayed frozen. Robust-to-conditioned
return differences are consequently training-budget confounded.

`scripts/analyze_bapr_v3_static_context_audit.py` reproduces both tables,
validates all file cardinalities and checkpoint metadata, and computes the
paired intervals. This phase remains evaluation-only and does not open
multi-policy-seed or cross-algorithm claims.

## Budget-matched persistent-mode mechanism screen

The next experiment removes the identified controller-budget confound before
reopening learned inference. It uses two environments (Ant and HalfCheetah) and
two persistent mode families:

- `deterministic_mean`: gravity/action-gain means remain fixed for each 500-step
  dwell, with no per-step actuator noise;
- `mean_variance`: the same persistent mean shifts plus mode-conditioned
  actuator noise sampled independently each step.

For each environment/family pair, `robust_long` applies all 1400 actor-update
iterations to one context-free base. `oracle_direct` applies 700 updates to the
same base architecture, copies it exactly into the conditioned branch, then
applies the remaining 700 updates under privileged mode context. Both receive
5.6M transitions, 350k critic updates, and 1400 effective actor-update
iterations. The two runs share seed 0 and all pre-iteration-700 settings; only
the scheduled recipient of the final 700 actor updates differs.

Scheduler tasks `t35869-t35876` cover the full eight-task matrix. They were
submitted without node binding, Slurm, or auto-adopt and launched across
`jtl311linux`, `local`, `node007`, and `jtl110gpu`. Checkpoint resume remains
wired for every task.

The preregistered gate is stricter than robust-to-oracle return alone. At the
final checkpoint, dynamic oracle must beat the equal-budget robust controller
and every fixed policy context in both stationary and switching evaluation;
the matching context must be row-optimal in at least three of four stationary
physics modes. Only a family that passes this gate proceeds to learned latent
training. The burst-torque inverse estimator is not reused merely because it
beat the undertrained frozen base.

## Budget-matched mechanism result

All four producer pairs reached iteration `1399` (`5.6M` environment steps per
arm). Each strict final audit then evaluated six controllers (`robust_long`,
dynamic oracle, and fixed contexts `0-3`) on five paired event streams. Every
pair therefore has `30/30` validated outputs. Event seeds `1100-1500` are five
evaluation streams for one training seed (`seed=0`), not five independently
trained policies.

| Family | Env | Robust stationary / switching | Oracle stationary / switching | Oracle - robust (stationary / switching) | Beats every fixed context | Diagonal optima | Pair gate |
|---|---|---:|---:|---:|---|---:|---|
| deterministic mean | Ant | 1985.2 / 2823.9 | 1718.4 / 2439.6 | -266.8 / -384.3 | fail / fail | 1/4 | **FAIL** |
| deterministic mean | HalfCheetah | 1386.6 / 1479.8 | 2368.3 / 2515.2 | +981.7 / +1035.5 | fail / pass | 2/4 | **FAIL** |
| mean + variance | Ant | 1570.2 / 2714.8 | 1415.9 / 2410.1 | -154.4 / -304.6 | fail / fail | 1/4 | **FAIL** |
| mean + variance | HalfCheetah | 2615.3 / 2443.7 | 3036.4 / 2861.4 | +421.1 / +417.7 | pass / pass | 4/4 | **PASS** |

The HalfCheetah mean-plus-variance cell is a clean positive mechanism case.
Dynamic oracle beats equal-budget robust on every event stream, beats each
fixed context in paired mean under both protocols, and all four stationary
physics rows prefer their matching context. Its oracle-minus-robust 95% paired
intervals are `[+379.9,+462.3]` stationary and `[+312.4,+523.1]` switching.
This establishes useful mode-conditioned controller headroom for that one
environment/protocol.

It does not pass the preregistered family-level gate. The same mean-plus-
variance family fails on Ant: oracle is below robust by `-154.4` stationary
and `-304.6` switching, and only `1/4` stationary rows are diagonally optimal.
The deterministic family also fails because Ant is negative and HalfCheetah
has only `2/4` diagonal optima; in HalfCheetah stationary evaluation, dynamic
oracle is additionally `2.3` points below fixed context 1. Thus no tested
family passes on both environments, and learned-estimator training remains
unauthorized.

This screen narrows the failure mode. Equalizing actor-update and transition
budgets removes the earlier frozen-base confound, but useful semantic context
is environment-dependent rather than a general property of these mode
definitions. HalfCheetah with persistent mean shifts plus per-step stochastic
disturbance has adaptation headroom; Ant does not learn a corresponding
mode-specialized policy under the same objective. The next algorithm step, if
continued, must first repair Ant controller specialization or explicitly scope
the claim to environments that pass the oracle/fixed-context screen. Training
a more elaborate latent estimator cannot repair a failed privileged-controller
upper bound.

HalfCheetah recovery did not alter the protocol. The deterministic pair was
finalized from its immutable source archive without retraining. For the
mean-plus-variance pair, the robust arm was already complete and the oracle arm
resumed from iteration `700` (`2.8M` steps), then reached iteration `1399` and
the same `5.6M`-step budget. Both final manifests and physical provenance were
revalidated before auditing. Protocol-integrity failures now terminate instead
of entering an impossible retry loop; corrected archived-source recovery is an
explicit, fail-closed operation.

Accepted audit tasks are `t35958-t35967` (Ant), `t36476-t36480`
(deterministic HalfCheetah), and `t36484-t36488` (mean-plus-variance
HalfCheetah). Strict pair reports were generated by
`scripts/analyze_bapr_v3_budget_matched_fork_pair.py`; the focused protocol
suite passes `8/8`. No remote checkpoint PKL was manually downloaded.

## Ant controller-specialization screen

The failed Ant cells are not uniform controller degradation. Averaged over the
five accepted event streams, the equal-budget shared `direct` oracle changes
the stationary return in modes `0-3` as follows:

| Family | Mode | Robust | Oracle | Oracle - robust |
|---|---:|---:|---:|---:|
| deterministic mean | 0 | 1787.1 | 1762.4 | -24.7 |
| deterministic mean | 1 | 1295.3 | 1501.8 | +206.5 |
| deterministic mean | 2 | 2774.2 | 1790.2 | -984.0 |
| deterministic mean | 3 | 2084.2 | 1819.3 | -264.9 |
| mean + variance | 0 | 1297.0 | 1241.5 | -55.5 |
| mean + variance | 1 | 1488.1 | 1574.9 | +86.8 |
| mean + variance | 2 | 1435.7 | 972.0 | -463.7 |
| mean + variance | 3 | 2060.1 | 1875.1 | -185.0 |

Thus the shared conditioned actor learns a useful mode-1 branch but destroys
the mode-2 controller in both families. In the mean-plus-variance cross-context
audit, fixed context 1 is best for all four physics rows. This is consistent
with shared-network interference or branch collapse, not with an absence of
mode-dependent dynamics. The legacy `expert` policy is not a valid four-mode
control: it reads only the first latent coordinate, so modes 1-3 share the same
scalar route.

The new `categorical_expert` policy gives each of the four modes a fully
independent two-layer actor and output heads. A one-hot privileged context
selects exactly one expert; a soft posterior forms a normalized nonnegative
mixture. Every expert is initialized as an exact copy of the robust actor at
the teacher boundary, and the existing confidence gate still falls back to
the frozen robust controller. Focused tests verify exact warm-start equality,
one-hot routing, soft mixing, and the required expert/latent cardinality.

Four controller variants are screened on Ant seed 0 under both persistent-mode
families:

| Variant | Purpose |
|---|---|
| `cat_mean` | Isolate the effect of independent per-mode capacity. |
| `cat_lcb` | Add the existing conservative LCB actor objective (`beta=-2`). |
| `cat_anchor_0p1` | Penalize action deviation from the robust actor with weight `0.1`. |
| `cat_anchor_1p0` | Apply a strong robust-action anchor with weight `1.0`. |

All pairs retain the original causal comparison: one immutable shared
iteration-699 snapshot, 5.6M transitions and 1400 effective actor updates per
arm, followed by `robust_long` versus privileged `oracle_direct`. The real Ant
transition smoke `t36492` completed iterations `0-3` and 2048 steps on GPU,
saved the final checkpoint, switched from robust stage to teacher stage at
iteration 2, and recorded `v2_conditioned_warmstarted=1`. The preceding
`t36491` never entered training because its smoke-only command supplied an
unsupported CLI batch-size flag; it was corrected rather than retried.

Scheduler tasks `t36494-t36501` are the eight formal pairs. They were submitted
without node or GPU binding and launched across `local`, `jtl311linux`,
`node007`, and `jtl110gpu2`. Each task owns a dedicated resumable checkpoint
directory. Automatic result-directory synchronization is deliberately disabled
for these large producer checkpoints.

This screen does not authorize a learned estimator. After all pairs finish,
strict five-stream stationary/switching audits must again show that dynamic
oracle beats its equal-budget robust arm and every fixed context in both
protocols, with at least `3/4` diagonal-optimal stationary rows. Only a variant
that clears that privileged-controller gate can proceed to learned online mode
inference or multiple training seeds.

### Categorical boundary-audit correction (2026-07-16)

The `cat_lcb/deterministic_mean` and `cat_anchor_1p0/mean_variance` producers
both completed iteration `1399` and `5.6M` steps. Their parent tasks
`t36496/t36501` and short retries `t36627/t36628` were reported failed only
because the original finalizer required the robust base path and a one-hot
categorical expert path to produce byte-identical 4000-step Ant trajectories.
Both boundary policy canaries had exactly zero mean/log-std difference and the
task-id schedules were identical; the long trajectories diverged after the
two algebraically equivalent actors followed different floating-point graphs.

The corrected fail-closed rule keeps byte-exact rollout equality for the
original direct actor. A categorical pair may instead pass only with an
identical task schedule, exact finite policy canaries for both branches, and
the expected robust/oracle warm-start flags. Scheduler protocol failures no
longer auto-retry, and finalize-only tasks now self-stage the validator and
emit an explicit completion marker. Tasks `t36638/t36639` finalized the two
pairs on their original node/GPU without training; both manifests report
`training_reentered=false`, final iteration `1399`, and validation mode
`categorical_policy_equivalence`. The same correction then finalized the four
completed node007 pairs as tasks `t37198-t37201`, also on their original GPUs
and without re-entering training. The two `cat_mean` producers remained in
their original training runs at that point.

## Final categorical-controller screen result (2026-07-16)

All eight categorical producer pairs reached iteration `1399` and `5.6M`
environment steps in both arms. The strict audit validated six controllers on
each of five paired event streams, giving `30/30` accepted outputs per pair and
`240/240` overall. Event seeds `1100-1500` are repeated evaluation streams for
one trained policy seed (`seed=0`), not five independent training seeds.

| Variant | Family | Robust stationary / switching | Oracle stationary / switching | Oracle - robust (stationary / switching) | Beats every fixed context (stationary / switching) | Diagonal optima | Pair gate |
|---|---|---:|---:|---:|---|---:|---|
| `cat_mean` | deterministic mean | 2351.7 / 2948.7 | 2428.5 / 2658.6 | +76.8 / -290.2 | fail / fail | 2/4 | **FAIL** |
| `cat_mean` | mean + variance | 1652.2 / 2562.7 | 1690.2 / 2724.0 | +38.0 / +161.3 | fail / fail | 1/4 | **FAIL** |
| `cat_lcb` | deterministic mean | 2264.5 / 3314.4 | 2388.2 / 3252.6 | +123.7 / -61.8 | fail / fail | 1/4 | **FAIL** |
| `cat_lcb` | mean + variance | 2197.6 / 3307.0 | 1423.1 / 2906.5 | -774.5 / -400.5 | fail / fail | 2/4 | **FAIL** |
| `cat_anchor_0p1` | deterministic mean | 2995.2 / 3145.6 | 2375.9 / 3065.3 | -619.3 / -80.3 | fail / fail | 1/4 | **FAIL** |
| `cat_anchor_0p1` | mean + variance | 944.3 / 2217.9 | 1519.5 / 2444.0 | +575.2 / +226.0 | fail / fail | 2/4 | **FAIL** |
| `cat_anchor_1p0` | deterministic mean | 1520.7 / 2955.7 | 3162.3 / 3781.7 | +1641.6 / +826.0 | pass / fail | 2/4 | **FAIL** |
| `cat_anchor_1p0` | mean + variance | 1151.0 / 2714.8 | 1435.8 / 2541.9 | +284.8 / -172.9 | fail / fail | 1/4 | **FAIL** |

Independent actor capacity is therefore not sufficient. Dynamic oracle beats
its robust arm in stationary mean for six of eight pairs, but in switching
mean for only three of eight. Only `cat_anchor_1p0/deterministic_mean` beats
all fixed contexts in stationary mean, and no pair does so in switching mean.
No pair exceeds `2/4` matching stationary context optima.

The strongest-looking return result is also the clearest warning against using
robust-to-oracle gain alone. In `cat_anchor_1p0/deterministic_mean`, oracle
gains `+1641.6/+826.0` over robust, yet fixed context 2 is best in physics modes
0, 2, and 3. During switching, fixed context 1 exceeds dynamic oracle by
`14.7` mean return. The privileged task label selects a higher-return bank than
the weak robust arm, but the bank is not semantically aligned with its four
physics modes. Other pairs show the same collapse toward one or two generally
useful experts: their best-context rows are respectively `1,1,0,3`, `0,0,0,0`,
`0,0,1,1`, `0,1,0,1`, `0,0,0,2`, `0,1,1,1`, and `1,1,1,1`.

This rules out shared actor parameters as the sole Ant blocker. The remaining
controller-side hypotheses are shared critic/target/temperature interference
and a training objective that rewards globally useful branches without making
the task-indexed branches identifiable. The categorical actors are independent
and exact robust warm starts, but they still use one context-conditioned critic,
target critic, entropy temperature, and global replay stream. A second, not
mutually exclusive explanation is limited Ant adaptation headroom: a reactive
feedback policy may already absorb these scalar gravity/action-gain changes,
so the chosen labels need not identify four distinct optimal controllers.

The environment and audit do not support blaming per-step morphology changes
or context leakage. Gravity and action gain remain fixed for the full 500-step
dwell; only the configured actuator disturbance is sampled per step. Dynamic
oracle reads the actual physics mode before each action. Fixed-context controls
keep the same switching physics stream and freeze only the policy context. All
controllers use the same strict horizon and paired event stream.

The preregistered consequence is final: **no learned estimator and no
five-training-seed expansion are authorized from this Ant screen**. Another
gate/BOCD/posterior variant cannot repair a privileged controller that loses to
a fixed branch. If Ant work is resumed, the next useful diagnostic is not a new
BAPR estimator. It is a fully independent per-mode specialist ladder, including
separate policy, critic, target critic, and entropy temperature, with balanced
mode updates and switch-state initialization. Only if assembled dynamic
specialists beat robust and every fixed specialist with at least `3/4`
diagonal rows should online inference be trained. If that ladder also fails,
the Ant mode family lacks the controller headroom required for this adaptation
claim and should be reported as a negative case.

The accepted pair-analysis tasks are `t38688`, `t38689`, `t38691`, `t38692`,
and `t38695-t38697`, plus `t38699`. Audit replacements preserved the same
event signatures: `t38686` replaced a local runtime-probe failure, `t38690`
replaced the validator temporary-file race, and `t38694` replaced a canceled
GPU-lock waiter. No checkpoint PKL or replay buffer was manually downloaded;
only the final JSON and Markdown analyses were synchronized.

## Fully independent Ant specialist result (2026-07-16)

The final controller-side diagnostic removes all remaining shared optimization
state. Four stationary specialists were continued from iteration `699` to
`1399` with separate policy, critic, target critic, entropy temperature,
optimizers, and replay sampling. The equal-budget robust controller also
reached iteration `1399`; every arm therefore consumed `5.6M` environment
steps. The assembled dynamic oracle selects the specialist matching the true
physics mode, while fixed-specialist controls hold one specialist constant on
the identical paired switching stream.

All ten strict audit groups pass the current immutable validator: five event
seeds (`1100-1500`) for each disturbance family, one training seed (`seed=0`),
the exact six-controller set, stable source/checkpoint bundle hashes within
each family, and complete stationary and switching outputs.

| Family | Robust stationary / switching | Dynamic oracle stationary / switching | Dynamic - robust (stationary / switching) | Dynamic beats every fixed specialist (S / W) | Diagonal optima | Gate |
|---|---:|---:|---:|---|---:|---|
| deterministic mean | 1927.6 / 2877.9 | 1753.6 / 2446.7 | -173.9 / -431.1 | fail / fail | 2/4 | **FAIL** |
| mean + variance | 1665.1 / 2668.8 | 1563.8 / 2664.0 | -101.3 / -4.9 | fail / fail | 2/4 | **FAIL** |

For deterministic means, the paired dynamic-minus-robust 95% intervals are
`[-692.3,+344.4]` stationary and `[-652.0,-210.3]` switching, with `2/5` and
`0/5` event-stream wins. For mean plus variance they are
`[-477.3,+274.7]` and `[-170.9,+161.1]`, with `2/5` wins in both protocols.
A fixed specialist remains stronger than the dynamic assembly: fixed mode 0
is nearly the robust controller and is best for three deterministic-mean
physics rows, while fixed mode 2 reaches `2961.3` switching return in the
mean-plus-variance family, above the dynamic oracle's `2664.0`.

The audit jobs did move between nodes, but the accepted outputs did not. The
original tasks `t38876-t38880` and `t38882-t38886` generated all ten valid
`group.json` files on `node004-node006`, and their logs contain the exact
`INDEPENDENT SPECIALIST AUDIT COMPLETE` marker. At that time the scheduler's
success-pattern list did not recognize this new marker, so completed jobs were
falsely classified as failed and automatically retried. Final retry records
`t38908-t38917` landed on `node003-node006`; each emitted only
`Complete valid specialist audit already exists` after validating the existing
group. These CPU nodes share
`/home/zhengliang01/scheduleurm_work/BAPR`, so no cross-node copy selected a
different checkpoint and no retry recomputed or overwrote an audit. The
scheduler now recognizes both completion strings.

This result rules out shared actor, critic, target, alpha, optimizer, and replay
interference as sufficient explanations for the Ant failure. Under the current
four Ant task labels, independently optimized stationary controllers still do
not form a semantically aligned controller bank. Consequently the
preregistered oracle-headroom gate fails for both families: no learned
estimator and no five-training-seed Ant expansion are authorized. Ant should
remain a negative mechanism case unless its task family is redesigned; the
validated bus and HalfCheetah mean-plus-variance cells remain the positive
adaptation evidence.

## Strict stochastic-regime rerun (launched 2026-07-16)

The original stochastic oracle screen is now being repeated without its
controller-budget confound. This rerun returns to the two bus-like mode
families whose hidden mode controls a per-transition event distribution while
gravity, morphology, and nominal action gain stay fixed:

- `packet_loss`: command-drop probabilities `0,.08,.18,.32`;
- `burst_torque`: event probability/torque-standard-deviation pairs
  `0/0,.03/.25,.08/.45,.15/.70`.

The new entry point writes to
`results_bapr_v3_stochastic_headroom_fork_v1`; it cannot reuse the exploratory
`oracle_v1` checkpoints. Each pair uses the validated shared-checkpoint fork:
700 base iterations, then either another 700 robust updates or 700 privileged
conditioned updates. Both compared controllers therefore reach iteration
`1399` and `5.6M` transitions. Large checkpoint PKLs remain on their producer
nodes rather than being synchronized automatically.

| Task | Family | Env | Initial placement |
|---|---|---|---|
| `t38918` | packet loss | Ant | `jtl311linux:GPU0` |
| `t38919` | packet loss | HalfCheetah | `jtl311linux:GPU1` |
| `t38920` | burst torque | Ant | `node007:GPU0` |
| `t38921` | burst torque | HalfCheetah | `node007:GPU1` |

All four scheduler records are unpinned and reached `running`. Their live
commands confirm `stochastic_mode`, the requested family, a fixed 500-step
dwell, and the direct privileged controller. Focused protocol tests pass
`26/26`.

Completion alone does not promote either family. Five paired event streams
must next compare equal-budget robust, dynamic true-mode oracle, and fixed
contexts `0-3`. A family passes only if dynamic oracle beats robust and every
fixed context in stationary and switching means, with at least `3/4` matching
stationary diagonal optima. Only then may a Bernoulli packet-loss or Gaussian
burst likelihood estimator be trained; learned BAPR and five-training-seed
claims remain out of scope until that gate passes.

## Strict stochastic-regime rerun result (2026-07-17)

All four producer pairs reached iteration `1399` / next iteration `1400` and
`5.6M` environment steps in both branches. The strict audit then validated
five paired event streams (`1100-1500`) and all six controllers for every
family/environment pair: equal-budget robust, dynamic true-mode oracle, and
fixed contexts `0-3`. This is `120/120` accepted outputs from one training seed
per pair. The large checkpoint and replay PKLs remain on their producer nodes;
only compact JSON and Markdown summaries were synchronized.

| Family | Env | Robust S / W | Oracle S / W | Oracle - robust S / W | Beats every fixed S / W | Diagonal | Gate |
|---|---|---:|---:|---:|---|---:|---|
| packet loss | Ant | 522.7 / 2015.9 | 1042.6 / 2312.2 | +519.9 / +296.3 | fail / fail | 1/4 | **FAIL** |
| packet loss | HalfCheetah | 2847.1 / 2755.0 | 2845.0 / 2910.7 | -2.1 / +155.7 | fail / fail | 1/4 | **FAIL** |
| burst torque | Ant | 1850.8 / 2558.3 | 970.9 / 2384.3 | -879.9 / -174.0 | fail / fail | 0/4 | **FAIL** |
| burst torque | HalfCheetah | 2506.8 / 2521.0 | 2811.7 / 2777.7 | +304.9 / +256.8 | fail / fail | 1/4 | **FAIL** |

There is real controller headroom in two cells. Packet-loss Ant's paired
oracle-minus-robust 95% intervals are `[+342.9,+696.9]` stationary and
`[+128.8,+463.8]` switching, with `5/5` wins. Burst-torque HalfCheetah's are
`[+229.0,+380.7]` and `[+73.1,+440.4]`, also with `5/5` wins. Packet-loss
HalfCheetah improves switching by `+155.7` but is tied in stationary return;
burst-torque Ant is decisively worse than robust in both protocols.

None of those robust comparisons establishes mode adaptation. Dynamic oracle
does not beat all fixed contexts in any cell, and the stationary specialist
matrices have only `0/4` or `1/4` matching diagonal optima. The best-context
rows are:

- packet-loss Ant: `1,1,3,0`;
- packet-loss HalfCheetah: `0,0,0,1`;
- burst-torque Ant: `3,2,1,1`;
- burst-torque HalfCheetah: `2,3,2,2`.

The original oracle-first gains were therefore partly a controller-budget
effect. Under equal budgets, the four context labels do not identify four
mode-specific optimal controllers. A Bernoulli/Gaussian mode estimator cannot
repair that failure, so neither learned inference nor a five-training-seed
expansion is authorized.

The remaining ambiguity is controller training versus benchmark structure.
Global packet-loss probability and zero-mean torque variance are scalar
severity levels; a reactive robust policy can compensate after disturbances,
and one conservative controller may legitimately dominate several modes. The
next diagnostic is restricted to the two positive-headroom cells above: train
four fully independent stationary specialists, including independent actor,
critic, target critic, alpha, optimizer, and replay paths. If their true-mode
assembly still fails the fixed-specialist and `3/4` diagonal gates, the next
environment must use qualitatively different action-channel regimes (for
example joint-group dropout, actuator delay, or persistent signed bias) rather
than further estimator tuning.

Training tasks `t38918-t38921` were finalized in place by `t41352-t41355`
without re-entering training. Strict audit groups are `t41367-t41376` for
burst torque and `t41377-t41386` for packet loss. The compact validated reports
are under
`jax_experiments/results_bapr_v3_stochastic_headroom_audit_v1/analysis/`.
The first burst aggregate completed scientifically but lacked a terminal
`DONE` marker, so scheduler records `t41461/t41462` falsely retried it; the
third retry was cancelled and packet aggregate `t41464` used an explicit
marker. No output or checkpoint was recomputed by those aggregate retries.

## Independent stochastic-specialist diagnostic (launched 2026-07-17)

The final attribution screen is running only on the two cells with clear
robust-to-oracle headroom. Each mode receives an independent policy, critic,
target critic, entropy temperature, optimizer state, and replay continuation
from the exact pair's iteration-699 shared checkpoint.

| Tasks | Family / env | Modes | Placement | Resume |
|---|---|---|---|---|
| `t41476-t41479` | packet loss / Ant | 0-3 | `jtl311linux`, two per GPU | mode 0: `702`; modes 1-3: `700` |
| `t41480-t41483` | burst torque / HalfCheetah | 0-3 | `node007`, one per GPU | mode 0: `702`; modes 1-3: `700` |

Bootstrap tasks `t41468/t41469` only validated and copied the completed source
pairs; they did not train. GPU smoke tasks `t41472/t41475` advanced mode 0 by
two iterations and are part of the formal budget, not discarded runs. All
eight formal logs confirm `stochastic_mode_fixed_id=0/1/2/3`, the correct
family/environment, and target iteration `1400`. Measured smoke VRAM was about
`1.24 GB`, so the declared `1.5 GB` permits the intended concurrency. Only a
compact completion manifest is configured for result sync; specialist bundles
and replay/checkpoint PKLs remain on their data nodes.

Passing this screen still requires dynamic assembly to beat robust and every
fixed specialist in both protocols and at least `3/4` stationary diagonal
optima. Failure will close the scalar-severity benchmark and trigger a redesign
toward qualitatively different action-channel modes rather than another BAPR
estimator variant.

## Independent stochastic-specialist result (2026-07-17)

All eight specialists completed at iteration `1399`, next iteration `1400`,
and `5.6M` transitions. Node-local preparation tasks `t42122/t42123` then
revalidated the robust bundle plus four specialist bundles for each cell. They
also proved that policy, critic, target critic, entropy temperature, and all
three optimizer states are distinct across modes while every branch shares the
same iteration-699 source snapshot. Strict paired groups are `t42134-t42138`
for packet-loss Ant and `t42145-t42149` for burst-torque HalfCheetah; aggregate
tasks are `t42150/t42151`.

| Family / env | Robust S / W | Dynamic specialist oracle S / W | Dynamic - robust S / W | Beats every fixed S / W | Diagonal | Gate |
|---|---:|---:|---:|---|---:|---|
| packet loss / Ant | 497.2 / 2021.5 | 737.9 / 2159.6 | +240.7 / +138.0 | fail / fail | 2/4 | **FAIL** |
| burst torque / HalfCheetah | 2503.9 / 2558.9 | 2595.2 / 2481.1 | +91.3 / -77.8 | fail / fail | 1/4 | **FAIL** |

Packet-loss Ant retains real stationary headroom over robust: the paired
dynamic-minus-robust interval is `[+65.6,+415.7]` with `5/5` event-stream wins.
Its switching interval is `[-37.7,+313.7]` with `4/5` wins, so the switching
gain is not decisive. More importantly, fixed mode 0 is stronger than the
dynamic assembly in stationary mean (`853.7` versus `737.9`), and fixed mode 1
is stronger in switching (`2246.5` versus `2159.6`). The stationary row winners
are `0,0,0,3`: only physics modes 0 and 3 select their matching specialists.
The apparent four-mode family therefore behaves like at most a low/moderate
regime plus one severe regime, not four semantically distinct controllers.

Burst-torque HalfCheetah is more conclusive. The dynamic assembly has only a
small, non-significant stationary advantage over robust and is worse under
switching. Fixed mode 1 reaches `2782.3 / 2743.4` and is the best specialist
for all four stationary physics rows; the row winners are `1,1,1,1`. A single
moderately conservative policy handles the whole scalar variance range better
than true-mode switching.

This closes the remaining controller-training ambiguity. Fully separating
actor, critic, target, alpha, optimizers, and replay does not create a
mode-aligned controller bank. The scalar packet-loss probability and
zero-mean torque variance labels describe disturbance severity, but they do
not reliably change the identity of the optimal controller. A learned mode
estimator, HMM, BOCD, or gate cannot repair this structural failure, so no
estimator training or five-policy-seed expansion is authorized for these
families.

The next benchmark must use qualitatively different persistent action-channel
mechanisms rather than four severity levels. The first fail-closed screen
should compare modes such as front/rear actuator-group attenuation, persistent
signed actuator bias, and action delay, with modest per-step aleatoric noise in
every mode. Robust and fully independent true-mode specialists must pass the
same fixed-specialist and diagonal gates before any learned BAPR inference is
trained.

The first prep submissions `t42120/t42121` were cancelled before launch because
a local-only `wait_for_files` check could not see remote-only bundles. Audit
submissions `t42124-t42133` were also cancelled before evaluation after exposing
local seed-cwd and node007 SSH argument-length constraints. The accepted tasks
use a short outer launcher, execute against immutable source archives on the
checkpoint-owning nodes, and emit explicit `DONE` markers. Only compact JSON
and Markdown were synchronized; no checkpoint or replay PKL was copied.

## Structured action-channel screen (launched 2026-07-17)

The failed scalar-severity families are replaced by one isolated
`structured_channel` protocol. Robot physics stays fixed for an entire run.
Every mode has the same per-step Gaussian actuator noise (`0.04`) and the same
impairment magnitude (`gain=0.45` on exactly half of the action dimensions),
but the persistently impaired subset differs: low half, high half, even
indices, or odd indices. Thus all four labels have equal disturbance severity
while requiring compensation on different control channels.

Focused tests cover sequential step execution, fixed dwell, switching setup,
and exact Ant/HalfCheetah gain masks. A 4096-sample audit on both environments
also verifies fixed gravity, four distinct equal-severity masks, and empirical
action mean/variance against the configured transition kernel.

Fresh equal-budget fork tasks are:

| Task | Env | Placement | Protocol root |
|---|---|---|---|
| `t42217` | Ant | `jtl311linux:GPU0` | `structured_channel_headroom_fork_v2` |
| `t42218` | HalfCheetah | `jtl311linux:GPU1` | `structured_channel_headroom_fork_v2` |

Both live logs confirm the `structured_channel` family, the correct CUDA
device, scan-fused rollout construction, and entry into training. They start
from scratch at a new root: 700 shared robust iterations followed by equal
700-iteration robust and true-mode-conditioned branches. No node was requested
at submission time; scheduler placed the two tasks on the available cards.

The superseded `v1` records `t42213/t42214` failed before training because the
new family was initially omitted from the `train.py` CLI choice list. Their
automatic retries correctly failed closed when the source hash changed. No
iteration or training checkpoint from `v1` is reused; `v2` preserves a fresh
immutable source snapshot.

Completion does not authorize learned inference. The next strict audit must
first show robust-to-oracle improvement in stationary and switching return.
Only a subsequently trained fully independent specialist bank that beats all
fixed specialists and reaches at least `3/4` diagonal stationary optima can
authorize an estimator or a five-seed expansion.

## Structured action-channel strict result (2026-07-17)

Producer tasks `t42217/t42218` completed without retries. Their shared base,
equal-budget robust branch, and oracle branch all validate at iteration `1399`,
next iteration `1400`, and `5.6M` transitions. Strict paired audit tasks
`t42483-t42492` produced all `60/60` controller outputs; node-local aggregate
task `t42590` validated both pairs. Only compact CSV, JSON, and Markdown audit
artifacts were synchronized; training checkpoint and replay PKLs remain on
`jtl311linux`.

| Env | Robust S / W | Oracle S / W | Oracle - robust S / W | 95% CI S / W | Diagonal | Gate |
|---|---:|---:|---:|---:|---:|---|
| Ant | 926.7 / 1987.4 | 1215.6 / 2102.9 | +288.9 / +115.4 | [-64.7,642.5] / [-23.2,254.0] | 1/4 | **FAIL** |
| HalfCheetah | 2354.8 / 2245.9 | 2641.1 / 2224.7 | +286.3 / -21.2 | [222.4,350.2] / [-176.4,133.9] | 3/4 | **FAIL** |

Ant is not promoted: fixed context 3 reaches `1504.6` stationary versus the
dynamic oracle's `1215.6`, and the stationary row winners are `3,3,1,3`.
This is another non-identifiable controller bank despite qualitatively distinct
channel masks.

HalfCheetah is the first clear MuJoCo specialization result in this branch.
The oracle improves stationary return by `12.2%`, wins all five paired event
streams, beats every fixed context in aggregate stationary and switching
means, and has stationary row winners `0,3,2,3`. Its strict gate fails only
because true-mode context 1 is not the best controller for physics mode 1 and
the `0 <-> 1` switching mean is `0.9%` below robust.

The trace diagnosis rules out mode-detection delay: this audit uses privileged
true modes. Across 25 paired 500-step segments, oracle-minus-robust is positive
for mode 0 (`+95` to `+119`, depending on segment position) but negative for
mode 1 (`-112` to `-157`). The last 400 steps after a switch remain `-85`
below robust in mode 1, so the deficit is a shared-controller optimization or
capacity failure rather than a transient switch cost.

Bootstrap task `t42594` therefore created four byte-validated HalfCheetah forks
from the exact iteration-699 shared base and published the existing robust
bundle. Tasks `t42595-t42598` now train modes 0-3 with fully independent policy,
critic, target critic, alpha, optimizer state, and replay continuation. All
four resumed at iteration `700`, `2.8M` transitions; scheduler placed two tasks
on each `jtl311linux` GPU. No Ant specialist or learned estimator was submitted.

Promotion remains fail-closed. The independent true-mode assembly must beat
equal-budget robust and every fixed specialist in both stationary and
switching evaluation, retain at least `3/4` diagonal stationary optima, and
repair mode 1. Only then is a learned causal mode estimator or five-training-
seed expansion authorized.

## Independent structured-channel specialists (2026-07-18)

Tasks `t42595-t42598` completed at iteration `1399`, next iteration `1400`,
and `5.6M` transitions after exact resumes from iteration `700` / `2.8M`
transitions. Each mode owns an independent policy, critic, target critic,
temperature, optimizer state, and replay continuation. Prep task `t42989`,
paired audits `t42990-t42994`, and aggregate task `t43036` validated all five
bundles without downloading checkpoint or replay PKLs.

The strict identity assembly still fails the old gate. It improves stationary
return over robust by `+315.8` (95% CI `[+258.6,+373.0]`, `5/5` wins), but its
switching difference is `-40.2` (95% CI `[-341.3,+260.9]`, `2/5` wins). The
stationary specialist matrix has row winners `[0,2,2,3]`, not `[0,1,2,3]`.
In particular, the independently optimized mode-1 specialist is not the best
controller for physics mode 1; specialist 2 is. This falsifies the previous
assumption that environment mode identity must equal controller identity.

## Frozen control-equivalence result (2026-07-18)

Task `t43037` froze the many-to-one controller map `[0,2,2,3]` using only the
predeclared calibration streams `1100-1500`. Five disjoint holdout streams
`2100-2500` were then evaluated by tasks `t43038,t43040-t43043`; task `t43046`
performed the strict aggregate. `t43040` had one transient SSH claim timeout
before launch, but no duplicate evaluation. The calibration winner agrees with
the holdout winner in all `4/4` physics rows and selects three distinct
specialists.

| Controller | Stationary mean +/- SD | Switching mean +/- SD |
|---|---:|---:|
| equal-budget robust | 2316.9 +/- 53.4 | 2263.5 +/- 120.4 |
| identity true-mode oracle | 2650.1 +/- 54.8 | 2256.9 +/- 131.0 |
| mapped control oracle | 2742.7 +/- 45.3 | 2523.9 +/- 56.4 |

The mapped controller beats robust by `+425.9` stationary (95% CI
`[+334.5,+517.2]`) and `+260.4` switching (95% CI `[+74.2,+446.6]`), with
`5/5` paired wins in both cases. It also beats every fixed specialist in both
metrics. These are gains of approximately `18.4%` and `11.5%`, respectively.
The promotion gate therefore **passes** for a learned causal router.

The target of inference is now explicitly separated from the control decision:
the probabilistic model estimates persistent physical transition modes while
accounting for per-step actuator noise; the frozen utility map marginalizes
that posterior into control-equivalence classes. Low-confidence decisions must
fall back to the robust policy. This change does not modify the legacy bus path
or the positive RE-SAC regularization sign.

## Learned control-equivalence router result (2026-07-18)

Trainer attempts `t43088/t43089` exposed two launch-time analysis bottlenecks:
rebuilding the fixed-mode JAX scan for every behavior/mode pair and evaluating
432 filter candidates with scalar Python decisions. `t43088` was cancelled
before any optimizer update; `t43089` completed all 1500 estimator updates and
was cancelled only after preserving its atomic final state. Vectorized
finalizer `t43090` loaded that state without retraining and passed the disjoint
validation gate. The frozen filter uses hazard `0.005`, evidence scale `0.25`,
confidence `0.8`, margin `0.02`, minimum history `8`, and hysteresis `0.02`.

Five untouched event streams were evaluated by `t43091-t43095`; CPU aggregate
`t43096` validated their estimator, bundle, source, and map provenance. Only
the compact router NPZ, manifests, and JSON/Markdown audits were synchronized.
No training checkpoint or replay PKL was copied.

| Controller | Stationary mean +/- SD | Switching mean +/- SD |
|---|---:|---:|
| equal-budget robust | 2320.5 +/- 91.1 | 2192.3 +/- 163.7 |
| mapped privileged oracle | 2783.1 +/- 55.2 | 2584.1 +/- 47.3 |
| learned causal router | 2601.9 +/- 90.0 | 2392.1 +/- 45.3 |

The learned router beats robust by `+281.4` stationary (95% CI
`[+157.4,+405.4]`) and `+199.8` switching (95% CI `[+48.1,+351.6]`), with
`5/5` paired wins in both metrics. Its conditional routing accuracy is `99.4%`
stationary and `97.0%` switching; wrong-route rates are only `0.5%` and `2.8%`.
This is a strict positive learned-adaptation result, not an estimator failure.

The preregistered promotion gate nevertheless fails because the router recovers
only `60.8%` of stationary and `51.0%` of switching oracle headroom, below the
`70%` requirement. Per-mode decomposition identifies a controller-utility bug:
the calibration map `[0,2,2,3]` was selected over the four specialists only.
For physics mode 1, calibration already shows robust at `2137.1`, above the
selected specialist 2 at `1962.0`; the final holdout confirms robust at
`2309.3` versus specialist 2 at `1979.5`. The router therefore identifies mode
1 correctly and then confidently chooses a controller worse than robust.

The switching audit also used a 1000-step horizon with 500-step dwell. Although
all four tasks were installed, a run starting in mode 0 can visit only modes 0
and 1 before the horizon ends. It is valid for that binary switch but is not a
complete four-mode router audit. The next protocol must therefore (1) include
robust as an explicit candidate in the calibration-frozen controller map,
(2) distinguish a deliberate robust decision from low-confidence fallback,
and (3) add a full-cycle switching metric with 250-step dwell. The existing
physical-mode estimator can remain frozen; this correction does not justify
retraining it or touching the legacy bus/RE-SAC path.

## Robust-inclusive utility router validation (2026-07-18)

Task `t43138` froze the robust-inclusive utility map `[0,4,2,3]`, where
controller `4` is the equal-budget robust policy. Specialist 1 is dominated by
robust in every calibration row and is therefore excluded from the decision
set. Validation streams `6100/6200` evaluated stationary tasks, the original
500-step slow pair, and five predeclared 250-step four-mode cycles. These
streams are disjoint from estimator training/validation and from the sealed
utility holdouts `7100-7500`.

The original frozen filter (`c80h8`) beats robust on both validation streams in
all three metrics: `+327.1` stationary, `+454.7` slow-pair switching, and
`+291.6` full-cycle switching. It recovers `66.2%`, `87.1%`, and `65.6%` of
the corresponding utility-oracle headroom. Full-cycle action-controller
accuracy is only `86.6%` with `13.4%` wrong routes, however, so the
preregistered validation gate fails and sealed holdouts remain unopened.

Tasks `t43146-t43155` tested lower confidence/minimum-history decisions without
changing the estimator. None passed: full-cycle recovery ranged from `43.3%`
to `58.9%`, action accuracy from `84.9%` to `85.6%`, and wrong-route rate from
`14.4%` to `15.1%`. Tasks `t43156-t43165` and aggregates `t43206-t43209` then
tested faster sticky-filter responses:

| Variant | Stationary recovery | Slow recovery | Full-cycle recovery | Full-cycle accuracy / wrong | Delay | Gate |
|---|---:|---:|---:|---:|---:|---|
| `h005e50c80h4` | 64.9% | 87.9% | 32.0% | 85.5% / 14.5% | 27.0 | fail |
| `h010e50c80h4` | 54.5% | 80.2% | 42.9% | 82.9% / 17.1% | 26.5 | fail |
| `h010e100c80h4` | 46.6% | 52.6% | -5.8% | 74.3% / 25.7% | 36.4 | fail |
| `h020e50c80h4` | 52.6% | 64.1% | 62.5% | 83.1% / 16.9% | 19.0 | fail |

Increasing hazard or emission weight shortens some switch delays but also
amplifies noisy single-transition evidence and degrades stationary routing.
This rules out a scalar threshold/hazard fix. The next targeted mechanism is a
bounded-memory posterior: retain the frozen probabilistic emission model and
utility table, but exponentially forget accumulated log evidence so a long
dwell cannot create effectively unbounded inertia. No five-stream holdout or
policy-seed expansion is authorized until that mechanism passes the same
validation gate.

### Bounded-memory follow-up

Tasks `t43211-t43218` evaluated four exponential posterior-decay values on the
same validation streams; aggregates `t43300-t43303` completed without failed
or retried evaluations.

| Decay | Stationary recovery | Slow recovery | Full-cycle recovery | Full-cycle accuracy / wrong | Gate |
|---:|---:|---:|---:|---:|---|
| 0.900 | -11.0% | -16.6% | -42.6% | 42.3% / 57.7% | fail |
| 0.950 | 33.9% | 66.2% | 37.1% | 73.2% / 26.8% | fail |
| 0.975 | 61.0% | 66.1% | 71.8% | 80.7% / 19.3% | fail |
| 0.990 | 72.1% | 74.4% | 39.5% | 84.1% / 15.9% | fail |

Decay `0.975` yields a significant full-cycle return gain over robust but
fails stationary recovery and routing accuracy. Its stationary mode-0/1/2
action accuracies are approximately `93%/99.7%/98%`, while mode 3 reaches only
`70.5%`. Continuous forgetting therefore removes the weak evidence needed to
identify mode 3 even when the regime is stationary. Decay `0.99` restores
stationary and slow-switch recovery but loses the full-cycle improvement.

The bounded-memory hypothesis is rejected. The next mechanism must preserve
unbounded accumulation during a stable regime and forget only after persistent
likelihood conflict with the currently believed mode. This is a causal
change-point reset, not another dense scalar sweep; the estimator parameters,
variance calibration, utility table, policies, and sealed holdouts remain
frozen.

### One-sided EMA reset follow-up

Tasks `t43305-t43310` and aggregates `t43311-t43313` tested persistent-gap EMA
resets at thresholds `0.25/0.50/1.00` with smoothing `0.25`. All six audits
completed without failure or retry, and all three variants fail:

| Gap threshold | Stationary recovery | Slow recovery | Full-cycle recovery | Full-cycle accuracy / wrong | Gate |
|---:|---:|---:|---:|---:|---|
| 0.25 | 7.1% | 35.7% | 11.4% | 59.0% / 41.0% | fail |
| 0.50 | 38.7% | 46.3% | 38.5% | 74.9% / 25.1% | fail |
| 1.00 | 56.0% | 88.0% | 31.5% | 77.4% / 22.6% | fail |

This reset statistic is structurally biased: it clips mode-supporting evidence
to zero and accumulates only contradictory gaps, so aleatoric noise eventually
causes a false reset even in a stationary regime. The result does not reject
change-point detection generally, but it rejects a nonnegative EMA detector.
The final frozen-emission test will use a drift-corrected CUSUM, where evidence
supporting the current mode reduces the accumulated score. If that detector
also fails validation, filter tuning stops and the next estimator must be
trained explicitly on causal switching sequences.

### Drift-corrected CUSUM result

Tasks `t43314-t43319` and aggregates `t43320-t43322` completed the final
frozen-emission filter test with no failures or retries.

| Threshold / drift | Stationary recovery | Slow recovery | Full-cycle recovery | Full-cycle accuracy / wrong | Gate |
|---:|---:|---:|---:|---:|---|
| 2.0 / 0.25 | 60.2% | 68.2% | 57.8% | 75.8% / 24.2% | fail |
| 4.0 / 0.25 | 60.3% | 89.8% | 87.1% | 83.6% / 16.4% | fail |
| 2.0 / 0.50 | 51.3% | 79.4% | 61.7% | 81.3% / 18.7% | fail |

CUSUM materially improves dynamic control. The `4.0/0.25` variant reaches
`2747.4 +/- 53.8` on the full cycle versus robust at `2360.3 +/- 84.9`, a
paired gain of `+387.1` with validation 95% CI `[+108.1,+666.1]`, and recovers
`87.1%` of oracle headroom. It still fails stationary recovery (`60.3%`) and
full-cycle routing accuracy (`83.6%`), so sealed holdouts remain unopened.

This exhausts post-hoc filtering of the frozen per-transition emission model.
The dynamic result confirms that change detection has value, but the remaining
stationary/dynamic trade-off requires a model that learns temporal evidence
directly. Further hazard, decay, confidence, EMA, or CUSUM threshold sweeps are
prohibited. The next estimator will freeze the current probabilistic emission
network and controller bank, then train a compact causal recurrent filter on
balanced stationary sequences and predeclared 250-step full-cycle sequences.

### Causal sequence-router result

The first sequence-router attempt `t43323` completed all data collection but
failed before update 1 because the NNX graph merge omitted the GRU cell's
retained RNG state. The functional split was corrected to carry non-parameter
state, a real JIT-gradient regression test was added, and retry `t43327`
completed all `2500/2500` updates. Only the compact 51 KB parameter NPZ and
manifest were synchronized.

On independent internal streams `9100/9200`, the frozen decision configuration
reaches `89.3%` stationary action accuracy and `82.0%` full-cycle action
accuracy, with `33.5`-step median switch delay. It therefore fails the hard
`90%` / `10%` gate and no return audit or sealed holdout was opened.

Diagnostic `t43328` separates fallback calibration from estimator capacity.
The best finite temperature/confidence decision reaches `93.7%` stationary but
only `87.4%` full-cycle accuracy. Even an argmax/no-fallback oracle over the
learned posterior is capped at `95.6%` stationary and `88.1%` full-cycle, so
decision tuning cannot pass. Action-time full-cycle physical-mode accuracy is
`86.7/88.5/87.5/90.7%` for modes `0/1/2/3`, respectively, and behavior-policy
accuracy ranges from `87.6%` to `95.1%`; the error is not a single-mode or
single-behavior collapse.

The failure is localized in time after a switch: accuracy is `2.9%` at steps
`0-7`, `22.9%` at `8-15`, `60.2%` at `16-31`, `90.2%` at `32-63`, and above
`93.7%` thereafter. The next controlled test keeps the architecture, emissions,
policies, utility map, and seed partitions fixed, but replaces mostly uniform
chunk sampling with switch-centered curriculum, weights only the first 32
post-switch steps, and selects a checkpoint on the independent internal
validation streams. A dual-timescale estimator is justified only if this
targeted optimization still cannot lift the no-fallback full-cycle ceiling
above `90%` without reducing stationary accuracy below `90%`.

### Switch-centered curriculum result

Task `t43329` collected one 2 MB compressed frozen-evidence dataset, trained
four controlled curricula from the same initialization, selected checkpoints
every 100 updates, and deleted all intermediate evidence and variant parameters
after publishing the compact best model. The result is negative:

| Variant | Best update | Stationary no-fallback accuracy | Full-cycle no-fallback accuracy | Delay |
|---|---:|---:|---:|---:|
| uniform + early selection | 2500 | 93.3% | 87.9% | 26.7 |
| 50% centered, weight 6, context 128 | 2900 | 85.4% | 80.5% | 19.7 |
| 75% centered, weight 8, context 128 | 500 | 80.5% | 77.6% | 20.9 |
| 50% centered, weight 8, context 64 | 2400 | 80.8% | 75.6% | 18.4 |

Switch-centered optimization consistently shortens delay by `6-8` steps but
destroys stable-regime discrimination. The best finite fallback configuration
on the selected uniform model reaches `93.7%` stationary and `87.6%` full-cycle
accuracy, so the internal gate still fails and no return validation or holdout
was run. This rejects the hypothesis that one GRU output can be repaired by
sampling/loss reweighting alone.

The next diagnostic is now justified as a two-timescale oracle ladder. The
uniform model is frozen as the slow expert; a switch-specialized expert is
trained separately and may replace it only in a privileged fixed window after
the true switch. If no fixed window exceeds `90%` full-cycle accuracy while
retaining the slow expert's stationary score, a learned gate has no headroom.
If it does, the subsequent model will learn a causal switch gate without true
mode or switch-time access.

### Dual-timescale oracle result

Task `t43331` froze the uniform router as the slow expert and trained seven
switch-specialized fast variants. The best predeclared fixed-window assembly
uses `fast_s75w8` for 32 steps after the true switch. It retains `93.3%`
stationary action accuracy and improves full-cycle accuracy to `89.8%`, with a
median switch delay of `18.5` steps, but its `10.23%` wrong-route rate misses
the hard gate by `0.23` percentage points.

The stronger privileged per-step selector, which chooses whichever frozen
expert is correct at that action, reaches `96.2%` stationary and `91.19%`
full-cycle accuracy with `8.81%` wrong routes. Thus the two experts are
complementary, but elapsed time since a true switch is not a sufficient gate.
This narrow upper bound justifies one final causal selector trained only from
previous-transition slow/fast posterior summaries. It does not justify return
validation yet; the sealed `7100-7500` streams remain unopened.

### Causal slow/fast gate result

Task `t43333` completed without retry and synchronized only a 4 KB parameter
file plus its manifest. Five independently initialized MLP gates were trained
on `12,916` exclusive-correct rows (`1,669` fast-only and `11,247` slow-only)
using balanced batches. All initializations converge to the same negative
result. The selected seed-0 gate at threshold `0.8` reaches `95.13%`
stationary accuracy but only `87.75%` full-cycle accuracy, `12.25%` wrong
routes, and a `26.9`-step median delay. It selects the fast expert on only
`2.35%` of full-cycle actions and is slightly worse than the frozen slow
expert; lower thresholds select fast more often but reduce validation
accuracy.

This rejects the slow/fast posterior-summary selector. The training classifier
reaches roughly `75%` balanced accuracy, but the causal posterior summaries do
not identify which expert is correct at the current action. Because even the
privileged per-step correctness oracle has only `1.19` percentage points of
headroom above the route threshold, further gate architecture or threshold
tuning is not justified. The dual-timescale route is closed, no return audit
was run, and the original sealed `7100-7500` streams remain unopened.

The negative routing gate must not be confused with absence of control value.
The earlier CUSUM router uses robust fallback when uncertain: on validation it
gains `+387.1` full-cycle return over robust with 95% CI
`[+108.1,+666.1]`, despite only `83.6%` exact-controller accuracy. Its
committed-route conditional accuracy is `93.8%`; most of the exact-route
penalty is deliberate robust fallback, which is counted as wrong even when its
utility loss is small. Any next confirmation therefore has to be a separately
predeclared, fresh-seed return/termination protocol. It cannot reopen or
reinterpret the failed route gate post hoc.

### Fresh CUSUM return confirmation protocol

Before inspecting any new outcome, decision variant `cs4d025c80h8` was frozen
for event seeds `10100/10200/10300/10400/10500`, which are disjoint from all
estimator, utility-calibration, sequence-router, validation, and sealed-holdout
streams. Scheduler tasks `t43340-t43344` evaluate robust, dynamic utility
oracle, and the frozen CUSUM utility router on stationary modes, 500-step slow
switches, and 250-step four-mode cycles. The old `7100-7500` holdouts remain
unopened.

The primary confirmation gate is fixed before these rollouts finish:

- full-cycle CUSUM-minus-robust event-seed paired 95% CI must be positive;
- CUSUM must win at least four of five full-cycle event seeds;
- full-cycle oracle headroom must be positive and CUSUM must recover at least
  `70%` of it;
- stationary return CI lower bound must exceed the `-100` noninferiority
  margin;
- stationary and full-cycle termination rates may exceed robust by at most
  five percentage points; and
- committed full-cycle route accuracy must be at least `90%`.

Exact-controller accuracy is diagnostic only in this new protocol. Robust
fallback is judged by realized return and termination rather than being
automatically counted as a primary failure. This changes the scientific
endpoint prospectively on fresh data; it does not retroactively pass the
failed sequence-router gate.

### Fresh CUSUM confirmation result

Tasks `t43340-t43344` completed without failure or retry, and scheduler task
`t43346` aggregated five provenance-identical groups. The predeclared overall
gate is **FAIL**, but seven of its eight criteria pass; the only failure is the
`70%` full-cycle oracle-headroom target.

| Protocol | Robust | Dynamic oracle | CUSUM router | CUSUM - robust (95% CI) | Wins | Oracle recovery |
|---|---:|---:|---:|---:|---:|---:|
| Stationary | 2299.6 +/- 115.9 | 2829.6 +/- 52.8 | 2649.1 +/- 80.7 | +349.5 [+283.4,+415.5] | 5/5 | 65.9% |
| Slow pair | 2128.7 +/- 267.2 | 2440.2 +/- 335.7 | 2474.0 +/- 281.3 | +345.3 [+215.2,+475.4] | 5/5 | 110.9% |
| Full cycle | 2382.5 +/- 83.9 | 2822.2 +/- 84.7 | 2619.7 +/- 99.4 | +237.1 [+19.4,+454.9] | 5/5 | 53.9% |

There are no stationary or full-cycle termination regressions. Full-cycle
committed-route accuracy is `92.7%`, above the frozen `90%` requirement, while
the robust fallback rate is `18.3%`. The result therefore confirms the primary
scientific claim that utility-aware adaptation improves control under the
structured stochastic benchmark: all three return protocols improve on every
fresh event seed, and the full-cycle paired confidence interval excludes zero.
It does not support the stronger claim that the current router captures at
least `70%` of dynamic-oracle headroom.

The remaining gap is now localized. It is not harmful committed routing or
termination; it is conservative robust fallback during uncertain portions of
the 250-step cycle. The CUSUM threshold is frozen and must not be tuned on
these confirmation seeds. A next algorithmic candidate should replace hard
controller switching with a robust-anchored, posterior-conditioned residual
policy trained on switch-matched rollouts, so partial beliefs can provide
graded adaptation instead of an all-or-nothing fallback. Any such candidate
must use development streams only and reserve a new untouched five-seed set;
the old `7100-7500` holdouts remain unopened.

### Posterior-conditioned residual capacity screen

Before training another policy, a lower-risk capacity screen is frozen on the
existing development streams `6100/6200`. It keeps the CUSUM posterior,
utility table, robust policy, and all specialists fixed. At each step it picks
the specialist with the largest posterior expected utility and executes
`a_robust + alpha * (a_specialist - a_robust)`, clipped to the action bounds.
`alpha` is the candidate cap times the positive expected utility advantage,
normalized by the median positive specialist headroom in the frozen utility
table. Thus an uncertain posterior that does not justify a specialist gives
exactly the robust action; there is no new confidence or CUSUM threshold.

The prospective caps are `0.25/0.50/0.75/1.00`. Promotion requires, on both
development seeds, higher full-cycle return than the hard CUSUM router and the
robust policy; the mean hard-router gain must be at least 50 points.
Stationary mean may fall by at most 50 points and no seed by more than 100,
termination may not increase, and adaptation must be active on at least 5% of
full-cycle steps. Baseline trajectories are replayed and must match the prior
development audit. New seeds `11100-11500` remain sealed unless one cap passes.

### Posterior residual screen result and nonlinear candidate

Scheduler tasks `t43349-t43356` and aggregation task `t43365` completed. All
three replayed baselines match the prior development groups exactly, with
maximum absolute episode-return difference `0.0`. The action-interpolation
screen is **FAIL**:

| Cap | Stationary | Delta hard | Full cycle | Delta hard | Delta robust |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 2348.8 | -295.8 | 2376.4 | -371.0 | +16.1 |
| 0.50 | 2241.2 | -403.5 | 2072.3 | -675.1 | -288.0 |
| 0.75 | 2422.9 | -221.8 | 2248.4 | -498.9 | -111.8 |
| 1.00 | 2710.7 | +66.1 | 2651.9 | -95.5 | +291.6 |

`cap=1.0` improves stationary oracle recovery from the hard router's `60.3%`
to about `73.7%`, but reduces full-cycle recovery from `87.1%` to about
`65.6%`. Termination never increases. This rules out linear interpolation of
independently trained policy actions: their action vectors are not a stable
linear control manifold, especially during posterior transients.

The next candidate is prospectively fixed as one nonlinear residual policy,
not another gate sweep. A two-layer network receives state, the causal
four-mode posterior, and the frozen robust action. Its output is a bounded
action residual multiplied by continuous posterior expected utility support;
when robust has the best posterior utility it is structurally identical to
robust. It trains on balanced 250-step switch-matched trajectories from event
seeds `3100/3200` under robust, dynamic-oracle, and hard-CUSUM behavior.
Initialization is selected only by oracle-action imitation plus an
entropy-weighted robust-anchor loss on `4100/4200`; one on-policy DAgger round
is retained only if that same held-out loss improves. Returns are then tested
once on `6100/6200` with the existing promotion gate. Seeds `11100-11500`
remain sealed.

### Nonlinear posterior residual result

Training task `t43369` selected initialization seed 2 and retained its single
DAgger round because held-out imitation objective improved from `0.197874` to
`0.193106`. The first audit launch exposed only a float32/float64 utility
rounding mismatch in a safety assertion (`9.66e-7` absolute strength
difference); the tolerance was corrected without changing the trained model,
controller bank, posterior, utility table, seeds, or endpoints. Scheduler tasks
`t43380/t43381` then completed the original two development audits, and task
`t43386` aggregated them.

The nonlinear residual gate is **FAIL**:

| Protocol | Robust | Hard CUSUM | Dynamic oracle | Nonlinear residual | Delta hard | Delta robust |
|---|---:|---:|---:|---:|---:|---:|
| Stationary | 2346.4 | 2644.6 | 2840.6 | 2348.0 | -296.7 | +1.5 |
| Slow pair | 2255.3 | 2724.1 | 2777.5 | 2188.2 | -535.8 | -67.0 |
| Full cycle | 2360.3 | 2747.4 | 2804.8 | 2193.5 | -553.8 | -166.7 |

Every comparison against hard CUSUM loses on both development seeds. The
candidate has no termination regression, but perturbs actions on `71.6%` of
full-cycle steps with mean action-delta L2 `1.064`. Its full-cycle wrong-route
rate is `23.7%`, and its selected held-out teacher MSE remains `0.1524`
(per-action RMSE about `0.39`). Stationary mode means further localize the
failure: relative to robust, the candidate changes mode 0/1/2/3 by roughly
`+30/-49/-179/+204`, whereas hard CUSUM gains about
`+766/+33/+472/-78`. The regression is therefore not lack of adaptation use;
it is off-manifold action regression that loses the discrete specialists'
control competence.

This closes both linear and MSE-distilled action residuals. Independently
trained policies do not form a smooth action manifold, and oracle-action MSE
averages incompatible expert targets under uncertain posteriors. More network
capacity, DAgger rounds, anchor weights, or CUSUM thresholds would tune the
failed surrogate rather than fix its objective. Seeds `11100-11500` remain
unopened. A further controller-level experiment, if pursued, must preserve
actual frozen-controller actions and learn from realized return or
counterfactual return advantage; it cannot emit averaged residual actions.

### Posterior-discrete fallback-fill protocol

Before running another learned selector, a zero-training v8 diagnostic is
frozen on the same `6100/6200` development streams. It preserves every
eligible hard-CUSUM decision. Only when hard CUSUM would use its robust
fallback does v8 select the real frozen controller with maximum posterior
expected utility; ties favor robust. It never interpolates actions, never
emits a learned action, and adds no confidence, history, CUSUM, or utility
threshold. The candidate posterior evolves causally under its own executed
controller actions.

The existing promotion gate is unchanged: full-cycle return must exceed hard
CUSUM and robust on both event seeds, mean hard-CUSUM gain must be at least 50,
stationary mean/per-seed losses may not exceed 50/100, termination may not
increase, and non-robust control must be used on at least 5% of full-cycle
steps. Frozen baseline groups are referenced by hash rather than replayed.
Seeds `11100-11500` remain sealed. Failure means hard fallback is protective,
not immediately recoverable headroom; only then may a future learned selector
be justified by return-based counterfactual labels rather than action MSE.

### Posterior-discrete fallback-fill result

Scheduler tasks `t43448/t43449` completed without retry and `t43450`
aggregated the hash-referenced development baselines. The v8 gate is **FAIL**:

| Protocol | Robust | Hard CUSUM | Dynamic oracle | Discrete selector | Delta hard | Delta robust |
|---|---:|---:|---:|---:|---:|---:|
| Stationary | 2346.4 | 2644.6 | 2840.6 | 2645.5 | +0.8 | +299.0 |
| Slow pair | 2255.3 | 2724.1 | 2777.5 | 2685.1 | -39.0 | +429.8 |
| Full cycle | 2360.3 | 2747.4 | 2804.8 | 2397.5 | -349.8 | +37.3 |

Full-cycle hard fallback occurs on `21.1%` of steps and v8 replaces `11.2%`
with a specialist, but wrong-route rate rises to `21.8%`. Across the ten
full-cycle episodes, return delta versus hard CUSUM correlates weakly with
replacement rate (`r=-0.17`) and strongly with wrong-route rate (`r=-0.92`).
Thus fallback is protective uncertainty handling, not free oracle headroom;
removing it exposes posterior ambiguity and can erase nearly all adaptation
gain.

Before training a state-conditioned selector, one final privileged upper bound
is frozen on `6100/6200`: preserve every hard-CUSUM committed decision and,
only during hard fallback, execute the true-mode utility-oracle controller.
Only full-cycle return/termination is needed. This upper bound must beat hard
CUSUM on both seeds, by at least 50 points on average, without increasing
termination. If it fails, no selector limited to fallback replacement has
sufficient demonstrated capacity and that route ends. Confirmation seeds
`11100-11500` remain unopened regardless of this privileged diagnostic.

### Privileged fallback-oracle result

Scheduler tasks `t43460/t43461` and aggregation task `t43462` completed the
predeclared full-cycle capacity audit. The privileged gate is **FAIL**:

| Robust | Hard CUSUM | Dynamic oracle | True-mode fallback oracle | Delta hard |
|---:|---:|---:|---:|---:|
| 2360.3 | 2747.4 | 2804.8 | 2677.3 | -70.1 |

Per-seed fallback-oracle deltas are only `+3.4` on 6100 and `-143.6` on 6200,
with no termination increase. It uses true-mode specialists during `20.3%`
hard-fallback steps (`15.0%` are non-robust specialists) and reduces aggregate
wrong routes to `6.25%`, yet still loses return. This proves that exact mode
identity during fallback is insufficient: brief expert substitutions move the
controller state distribution and posterior trajectory, and switching back to
the hard route can be worse than staying robust. Route accuracy is not a
surrogate for control return.

The fallback-selector route is therefore closed; no return-advantage learner
will be trained and seeds `11100-11500` stay unopened. The best deployable
result remains frozen hard CUSUM `cs4d025c80h8`, whose independent fresh-seed
confirmation improves robust on all five seeds and has a positive full-cycle
paired confidence interval. Recovering more oracle headroom would require a
new jointly trained, persistent option policy with return-level safety
constraints, not another posterior threshold, action residual, or per-step
controller selector. That is a separate algorithm project rather than a
defensible post-confirmation tweak.

### BAPR-v4 persistent-option capacity protocol

BAPR-v4 starts that separate algorithm project without modifying the frozen
v3 confirmation result. It replaces the independent controller bank with one
shared FiLM SAC actor. The robust behavior is the all-zero option; four
one-hot options modulate the same state trunk and output heads. All parameters
are optimized from realized SAC return on switch-matched structured-channel
rollouts. No action imitation, controller interpolation, per-step expert
replacement, or post-confirmation CUSUM tuning is used.

Training alternates one robust-context iteration with three privileged
true-mode option iterations. This keeps robust and option trajectories in the
same replay distribution while giving the low-level policy experience with
the states reached after option changes. The causal estimator is initialized
from the independently validated structured-channel model, then continues to
train on the new policy's transitions. At learned deployment it may commit or
release an option only at 64-step boundaries. The frozen CUSUM parameters are
threshold `4.0`, drift `0.25`, confidence `0.80`, posterior margin `0.02`, and
hysteresis `0.02`.

The capacity test is one HalfCheetah seed-8 controller trained for exactly
`1400` iterations / `5.6M` environment steps. Strict development audits will
compare its robust, privileged oracle-option, and learned persistent-option
contexts against equal-protocol robust, dynamic oracle, and hard CUSUM. The
oracle option must beat hard CUSUM on both development event streams by at
least 50 return on average without a termination increase before learned
option routing can be promoted. If the oracle option fails, the shared
persistent architecture has no demonstrated capacity and no learned-router or
threshold sweep follows. If oracle passes but learned fails, the remaining
failure is localized to causal inference/boundary selection. The sealed
`11100-11500` streams remain unopened.

Scheduler smoke task `t43468` completed all rollout, update, evaluation, and
checkpoint paths at iteration 4 / 512 steps. Formal scheduler task `t43470`
completed on `jtl311linux` at the exact final checkpoint: saved iteration
`1399`, next iteration `1400`, and `5.6M` environment steps. Development
audits `t43635/t43636` and aggregation `t43637` then consumed that checkpoint
through explicit result-file dependencies.

### BAPR-v4 persistent-option result

The privileged-oracle capacity gate is **FAIL**, so this architecture is
closed before any confirmation sweep:

| Event | Source | Stationary | Slow pair | Full cycle | Full - hard CUSUM |
|---:|---|---:|---:|---:|---:|
| 6100 | hard CUSUM | 2673.6 | 2620.1 | 2709.3 | +0.0 |
| 6100 | v4 robust | 1217.6 | 957.8 | 1231.1 | -1478.2 |
| 6100 | v4 oracle option | 2402.7 | 2306.5 | 2337.7 | -371.6 |
| 6100 | v4 learned option | 2172.6 | 2130.0 | 1668.7 | -1040.6 |
| 6200 | hard CUSUM | 2615.7 | 2828.0 | 2785.4 | +0.0 |
| 6200 | v4 robust | 1224.0 | 1004.7 | 1230.2 | -1555.2 |
| 6200 | v4 oracle option | 2326.7 | 2344.9 | 2417.9 | -367.5 |
| 6200 | v4 learned option | 2107.9 | 1878.1 | 1716.9 | -1068.5 |

No method terminated early, and the oracle option used the correct physical
mode on every step. The failure is therefore not an environment, horizon,
termination, or causal-inference artifact. It is controller capacity and
training interference. The shared zero-context branch is about 1100 points
below the equal-protocol robust controller, while even true-mode options are
about 370 points below hard CUSUM on full cycles. Stationary mode 2 is the
largest localization: its v4 oracle return is only `1355-1616`, versus
`3226-3288` for the frozen specialist oracle. The learned option is a
secondary failure: full-cycle physical-mode accuracy is only `55.9-58.1%`,
but improving it cannot rescue an oracle controller that already fails.

The implementation explains the pattern. One FiLM-modulated trunk and output
head receive gradients from every option, the robust context owns only one in
four rollout iterations, and replay trains each transition only under its
stored rollout context. Thus the robust behavior is neither frozen nor
independently represented, and most physically labelled transitions are not
used to train both the robust and matching option objectives. The next
capacity diagnostic must keep persistent switch-matched options but use a
separate robust actor/critic head, hard option-specific actor/critic heads,
and replay relabelling that trains robust plus true-mode contexts from every
transition. It remains a return-trained option policy; it does not revive
action interpolation, residual imitation, per-step specialist swapping, or
CUSUM threshold tuning.

### BAPR-v5 isolated hard-option protocol

V5 keeps the same persistent-option hypothesis but directly tests the two v4
failure mechanisms. It gives robust and each hard option separate actor MLPs
and critic ensembles, initialized to identical control functions. Every replay
transition is relabelled into two optimization examples: zero-context robust
and the transition's privileged true-mode option. Robust and oracle-option
behavior iterations alternate 1:1 so both state distributions remain in the
buffer. This is not a bank of frozen stationary specialists: all option heads
are trained jointly from return on the same switching process, and the learned
router still selects persistent options only at the causal 64-step boundary.

The screen retains seed 8, 1400 iterations / 5.6M environment steps, event
streams `6100/6200`, and the unchanged oracle-first promotion gate. Smoke task
`t44379` completed at iteration 4 / 512 steps. The superseded `t44380` never
entered training: it failed while loading two bootstrap files excluded by
staging. Those exact files were copied by matching SHA-256 into the staged
protocol snapshot before the clean resubmission. No partial checkpoint was
reused.

### BAPR-v5 isolated hard-option result

Formal task `t44385` and strict audits `t44381/t44382` completed at saved
iteration `1399`, next iteration `1400`, and exactly `5.6M` environment steps.
All evaluated episodes reached the strict 1000-step horizon with no early
termination. The privileged option selected the true physical mode on every
step, so the capacity result is independent of estimator accuracy.

| Event | Source | Stationary | Slow pair | Full cycle | Full - hard CUSUM |
|---:|---|---:|---:|---:|---:|
| 6100 | hard CUSUM | 2673.6 | 2620.1 | 2709.3 | +0.0 |
| 6100 | v5 robust | 1948.9 | 1708.1 | 2019.2 | -690.1 |
| 6100 | v5 oracle option | 2551.0 | 2718.2 | 2518.2 | -191.2 |
| 6100 | v5 learned option | 2542.4 | 2498.0 | 2035.0 | -674.4 |
| 6200 | hard CUSUM | 2615.7 | 2828.0 | 2785.4 | +0.0 |
| 6200 | v5 robust | 1943.5 | 1738.3 | 1882.3 | -903.1 |
| 6200 | v5 oracle option | 2518.9 | 2624.1 | 2517.5 | -267.9 |
| 6200 | v5 learned option | 2559.6 | 2389.7 | 2222.0 | -563.5 |

V5 materially repairs v4 but still fails the oracle-first gate. Relative to
v4, full-cycle robust improves by `+788/+652`, privileged oracle by
`+181/+100`, and learned routing by `+366/+505` on the two event streams.
The remaining deficit is structured by mode rather than a uniform environment
or evaluation error. V5 oracle mode 1 reaches `3209.6/3130.4`, about
`970/983` above the old dynamic oracle, and mode 3 is within `118-199` points.
Modes 0 and 2 remain `889-1157` points below the independently trained oracle.
Learned full-cycle physical-mode accuracy is only `61.6/68.6%`, but perfect
oracle routing still loses, so inference remains secondary.

The v5 optimizer does not actually give the five independent heads equivalent
SAC training budgets. Relabelling expands a replay batch of size `B` to `2B`:
the robust head receives `B` examples, while each option receives only about
`B/4`; actor and critic losses then take one unweighted mean over all `2B`
examples. All five heads also share one entropy temperature. With only 250
updates per iteration, each option receives about one quarter of the replay
draw/update budget of a stationary specialist, while the hard-CUSUM controller
bank was trained with four full specialist budgets plus a robust budget. The
late training trace also shows Q-std rising from about `1.14` to `3.58` during
the final 58 iterations.

The next bounded controller test is therefore optimizer-equivalent v6, not a
new router or gate. It retains v5's independent heads and switching rollouts,
but balances robust and four option examples per update, gives every head an
independent SAC temperature, and raises the update count so each option gets
the same replay-draw budget as one stationary specialist. The environment-step
budget remains `5.6M`; this deliberately separates optimization undertraining
from lack of unique per-mode data. A failure would justify a final 4x
environment-data ladder before closing the joint persistent-option route.

Initial aggregate task `t44383` and its automatic retries failed only because
a POSIX environment-prefix command was placed on Windows CPU nodes. The JSON
aggregator has been made JAX-independent and its scheduler spec now uses the
Linux CPU pool; this infrastructure failure did not affect any return above.

### BAPR-v6 optimizer-equivalent option result

Formal task `t45429`, strict audits `t45430/t45431`, and aggregation task
`t45437` completed at saved iteration `1399`, next iteration `1400`, and
exactly `5.6M` environment steps. All strict episodes reached 1000 steps and
privileged routing selected the correct physical mode on every step.

| Event | Source | Stationary | Slow pair | Full cycle | Full - hard CUSUM |
|---:|---|---:|---:|---:|---:|
| 6100 | hard CUSUM | 2673.6 | 2620.1 | 2709.3 | +0.0 |
| 6100 | v6 robust | 2395.9 | 2394.6 | 2471.6 | -237.7 |
| 6100 | v6 oracle option | 2560.2 | 2639.0 | 2517.2 | -192.1 |
| 6100 | v6 learned option | 2525.7 | 2573.4 | 2388.5 | -320.8 |
| 6200 | hard CUSUM | 2615.7 | 2828.0 | 2785.4 | +0.0 |
| 6200 | v6 robust | 2424.3 | 2444.7 | 2424.5 | -360.9 |
| 6200 | v6 oracle option | 2514.5 | 2676.4 | 2581.1 | -204.3 |
| 6200 | v6 learned option | 2517.2 | 2553.3 | 2373.6 | -411.8 |

Balanced examples and independent temperatures repair the robust branch but
do not repair privileged option capacity. Relative to v5, full-cycle robust
improves by `+452/+542`; privileged oracle changes by only `-1/+64`. The v6
stationary oracle mode means are `[2841,2521,2381,2498]` and
`[2856,2458,2387,2357]`. Against the independently trained controller bank,
mode 1 is stronger, but modes 0/2/3 remain lower; mode 2 still trails by about
`840-907` return. Perfect routing therefore cannot rescue this model.

V6 equalizes replay draws, not independent environment information. Across
`5.6M` switching steps each physical option sees only about `1.4M` unique
true-mode transitions, and the shared `1M` replay contains only about `250k`
transitions per mode. A fixed specialist instead receives `5.6M` unique
transitions and a `1M` same-mode replay. The final bounded capacity test is
therefore v7: `5600 x 4000 = 22.4M` switching steps, a `4M` replay, and `250`
updates per iteration. This keeps each option's total replay-draw count equal
to v6 while matching a specialist's unique-data and replay-support budgets.
Failure closes the jointly trained persistent-option route; it will not be
followed by another gate, residual, or estimator sweep.

Scheduler resource declarations were also corrected. New BAPR controller
families start at `2048MB`, and smoke/formal/audit share a `vram_resource_family`
so the formal-shaped smoke peak automatically calibrates queued descendants.
The previous hand-entered `5-9GB` estimates were placement reservations, not
observed requirements, and unnecessarily excluded 8GB GPUs.

### BAPR-v7 unique-data-equivalent option result

Formal task `t47131`, strict audits `t47132/t47133`, and aggregation `t47134`
completed at saved iteration `5599`, next iteration `5600`, and exactly
`22.4M` environment steps. This is the final planned capacity ladder: it
preserves v6's independent heads and balanced optimizer but gives each option
approximately `5.6M` unique same-mode transitions and a `1M` same-mode replay
support, matching an independent hard-CUSUM specialist. All strict episodes
reached 1000 steps and the privileged selector used the true mode exactly.

| Event | Source | Stationary | Slow pair | Full cycle | Full - hard CUSUM |
|---:|---|---:|---:|---:|---:|
| 6100 | hard CUSUM | 2673.6 | 2620.1 | 2709.3 | +0.0 |
| 6100 | v7 robust | 2130.0 | 2217.6 | 2065.0 | -644.4 |
| 6100 | v7 oracle option | 2716.7 | 2894.1 | 2654.2 | -55.1 |
| 6100 | v7 learned option | 2720.0 | 2622.7 | 2209.5 | -499.9 |
| 6200 | hard CUSUM | 2615.7 | 2828.0 | 2785.4 | +0.0 |
| 6200 | v7 robust | 2129.9 | 2126.9 | 2079.3 | -706.1 |
| 6200 | v7 oracle option | 2716.9 | 2724.6 | 2804.8 | +19.4 |
| 6200 | v7 learned option | 2703.5 | 2626.2 | 2385.6 | -399.8 |

The oracle gate fails: it must beat hard CUSUM on both streams and average at
least `+50`, while v7 obtains `-55.1/+19.4` (mean `-17.9`). Equalizing both
optimizer exposure in v6 and unique same-mode data in v7 does not recover the
independent-controller ceiling. The learned policy is lower still. Under the
frozen decision rule, this closes the jointly trained persistent-option route;
the historical raw runs can be discarded after preserving this protocol record.

## Independent training-seed confirmation (v8, frozen 2026-07-20)

The earlier hard-CUSUM result used fresh stochastic event streams but only one
controller-bank training seed. V8 is the confirmatory experiment for that
limitation. It freezes HalfCheetah `structured_channel`, the learned physical-mode
estimator snapshot, utility-aware hard CUSUM `cs4d025c80h8`, strict 1000-step
evaluation, and the independent robust-plus-four-specialist controller bank.
There is no new gate, residual, option architecture, or decision threshold.

Policy-training seeds are `0-4`. For every seed, SAC, ESCP, RE-SAC, the robust
SAC controller, and each of four fixed-mode SAC specialists train in separate
scheduler GPU tasks for 5.6M steps. CPU calibration waits for all seven bundles
for that seed; CPU audits wait for both the calibration table and all bundles.
Thus no evaluation can launch from an intermediate checkpoint. RE-SAC keeps
positive `weight_reg=0.01` and `beta_ood=0.01` with learning rate `1e-5`.

Utility calibration uses only event seeds `2100/2200`. The untouched final event
streams are `11100,11200,11300,11400,11500`, paired across all five methods and
all five training seeds. Primary inference averages event streams within each
policy seed and uses a paired Student-t 95% CI over the five independent policy
seeds. Promotion requires a positive CI against the strongest per-seed
SAC/ESCP/RE-SAC baseline, at least 4/5 policy-seed wins, stationary
noninferiority within 100 return, and termination noninferiority within 0.05.
