# Equal-per-controller adapter upper-bound protocol

## Question

The frozen-base adapter confirmation showed reproducible specialization value
relative to the frozen controller, but not relative to robust continuation.
Each independent adapter saw only 0.7M post-fork transitions while the robust
controller saw 2.8M. This diagnostic asks one narrow question: can independent
fixed-mode residual control beat robust when every adapter receives the full
2.8M post-fork budget?

This is intentionally compute-unmatched. The four-adapter bank consumes 11.2M
post-fork transitions, four times the robust continuation, and cannot be used
as a paper headline or fair baseline comparison.

## Replay boundary

Completed iter-1575 adapter bundles contain policy and optimizer state but
deliberately omit `replay_buffer.npz`. Resuming them would reset replay a second
time and confound additional budget with a new distributional boundary.
Therefore every diagnostic branch starts again from the same validated robust
source at iter 1400 / 5.6M transitions, resets replay and optimizer state once,
then trains continuously for 700 iterations to iter 2100 / 8.4M transitions.
The frozen base actor remains hash-identical throughout.

Frozen settings:

- environment: `HalfCheetah-v2`, `structured_channel`;
- policy-training seeds: `16,24,32,40`;
- modes: `0,1,2,3`, one independent branch per mode;
- residual cap: `0.50`;
- fixed-mode rollout and privileged identity routing;
- 4000 samples and 250 updates per iteration;
- final per-controller budget: 8.4M total, 2.8M post-fork;
- fresh sealed event seeds: `77100-77500`;
- deterministic policy mean, forced 1000-step horizon, dwell 250;
- five stationary episodes per mode and five switching episodes per event.

## Decision gate

The five event streams are averaged within each policy-training seed. The four
independently trained seeds are the inferential units. A positive diagnostic
requires all of:

1. identity routing improves stationary return by at least 5% over robust;
2. identity routing improves switching return by at least 10% over robust;
3. paired two-sided 95% intervals for both gains lie above zero;
4. identity wins both metrics in all four seeds;
5. identity beats each seed's best fixed adapter on both metrics, with four
   wins and paired intervals above zero;
6. switching termination is noninferior by at most 0.05.

A pass authorizes a compute-efficient redesign with one shared all-mode critic
or backbone and lightweight mode-specific residual heads. A failure rejects
independent adapter optimization. It does not trigger learned-router training.

## Scheduler graph

The graph contains 16 GPU training tasks, 20 CPU-only strict audits, and one
CPU aggregate. GPU memory is 2300 MB from the measured 2.09 GiB architecture
peak. Training remains checkpoint-managed and portable across `local`,
`jtl110gpu`, `jtl110gpu2`, `jtl311linux`, and `node007`. No exact GPU or server
slot is pinned. Audits are restricted to `node001-node006` and are released
only after all four adapter bundles and the existing robust bundle for their
seed pass file gates. Only GPU training tasks are explicitly dispatched.

The graph was submitted atomically as `t51411-t51447`: GPU training
`t51411-t51426`, CPU audits `t51427-t51446`, and aggregate `t51447`. Input
staging completed before launch. Logs sampled on `jtl311linux` and
`jtl110gpu` both report `iter=1400`, `steps=5600000`, and empty replay before
continuous training to iter 2100. Twelve producers first launched across local
and the three jtl nodes. The remaining four were then admitted to node007 after
its source bundle finished staging and launched one per GPU as
`t51422-t51425`. Sampled node007 logs also verified the iter-1400 resume and
showed no pthread startup failure. Downstream CPU tasks remain file-blocked.

The first GPU2/3 attempts (`t51424,t51425`) stopped during the bootstrap probe:
their copied actor differed from the source by `5.6177e-6`, while the original
CUDA cross-device tolerance was `5e-6`; critic error was exactly zero. Exact
parameter-block hashes, zero inserted context columns, zero residual
initialization, and frozen-base hashes remain binding. The action forward
tolerance was widened to `1e-5`, still less than one hundred-thousandth of the
normalized action range. Scheduler children `t51465` (mode 2, GPU3) and
`t51466` (mode 1, GPU2) replaced the two failed attempts; both passed bootstrap
and resumed from iter 1400. Together with `t51422`/`t51423`, node007 now runs
one task on each GPU.

## Final result

All 16 training branches reached iter 2100 / 8.4M transitions, all 20 sealed
audits completed, and aggregate task `t51447` completed. Five event streams
were averaged within each independently trained policy seed.

| Seed | Robust stat | Identity stat | Gain | Robust switch | Identity switch | Gain |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 2295.2 | 2551.5 | +11.2% | 2163.8 | 2412.3 | +11.5% |
| 24 | 1993.4 | 1608.7 | -19.3% | 1932.3 | 1593.2 | -17.5% |
| 32 | 2431.3 | 2089.8 | -14.0% | 2395.5 | 1904.8 | -20.5% |
| 40 | 2436.7 | 2386.9 | -2.0% | 2300.8 | 2213.9 | -3.8% |

Across the four independent policy seeds, identity routing minus robust is
`-129.9` stationary (95% CI `[-542.7,+282.9]`, `-5.7%`, one win) and `-167.0`
switching (95% CI `[-615.8,+281.7]`, `-7.6%`, one win). Identity routing also
fails to reliably beat the best single fixed adapter: stationary `+128.7`
with CI `[-49.7,+307.1]`, switching `+61.7` with CI
`[-150.8,+274.2]`. No policy terminated early, so termination does not explain
the differences.

The adapters do improve the frozen iter-1400 base in all four seeds:
stationary `+373.5` (95% CI `[+111.3,+635.6]`, `+20.9%`) and switching
`+289.3` (95% CI `[+71.3,+507.2]`, `+16.6%`). Robust continuation improves the
same frozen base even more: stationary `+503.4` and switching `+456.3`.

## Failure anatomy

The correct-mode controller is stationary-optimal among the four adapters in
12/16 seed-mode rows, so specialization exists. That statistic does not imply
that the specialized controller beats the robust policy:

- seed 16 is the sole successful optimization seed; identity beats robust in
  all four modes by `+6.4%` to `+16.3%`;
- seed 24 loses in all four modes by `-8.6%` to `-25.4%`, while switching is
  only `1.0%` below its own stationary score. This is primarily adapter
  optimization failure, not switch detection or transient control;
- seed 32 loses in all four modes by `-3.0%` to `-30.9%`, then loses another
  `8.9%` from stationary identity to switching identity. Both controller
  quality and switch-state distribution mismatch contribute;
- seed 40 has two mode wins and two losses and is near stationary parity, but
  switching is `7.2%` below its own stationary score.

This audit uses privileged true-mode routing, so no estimator, posterior, gate,
or detection-delay error can explain the primary loss. It also gives each
adapter the same post-fork transition count as the complete robust
continuation, spending four times as much post-fork data for the bank. The
remaining failure is unstable independent controller optimization plus, for
some seeds, incompatibility when separately optimized controllers are switched
on states generated by one another.

## Post-hoc root-cause diagnosis

The preregistered decision remains a failure, but the result rejects this
specific frozen-residual formulation. It does not establish that
`structured_channel` has no adaptation headroom or that independent policies
can never work.

### 1. The frozen base is often still undertrained

Both arms reset replay and optimizer state once at the common fork, so replay
or optimizer continuity is not an arm-specific confound. The important
asymmetry is controller capacity: robust continuation updates its complete
actor, including mean and log standard deviation, while the adapter freezes
that actor at 5.6M transitions and can only add a bounded `0.50` pre-tanh mean
residual.

| Seed | Frozen stat | Robust stat | Identity stat | Robust gain over frozen | Adapter gain over frozen |
|---:|---:|---:|---:|---:|---:|
| 16 | 2192.9 | 2295.2 | 2551.5 | +102.3 | +358.6 |
| 24 | 1437.1 | 1993.4 | 1608.7 | +556.2 | +171.6 |
| 32 | 1753.8 | 2431.3 | 2089.8 | +677.5 | +336.0 |
| 40 | 1759.3 | 2436.7 | 2386.9 | +677.5 | +627.7 |

Identity improves the frozen base in 15/16 stationary seed-mode rows. It loses
mainly because seeds 24 and 32 still need large full-actor improvements after
the fork, which the frozen residual cannot express. Descriptively, frozen-base
score and identity-minus-robust outcome correlate at `r=0.91` across the four
training seeds. Seed 16 is the only base already near its later robust score,
and it is also the only seed on which every adapter wins.

### 2. The two post-fork SAC objectives are not matched

`RegimeSAC` uses the minimum ensemble target. `BAPRv2`, which implements the
adapter update, bootstraps every critic head from its own target head and the
`critic_target_mode` configuration does not select a minimum-target path.
The adapter configuration then disables `bapr_v2_reg_weight`,
`bapr_v2_beta_ood`, and the LCB actor objective. It therefore retains
RE-SAC-style independent bootstrapping while removing the regularization and
conservative actor terms that normally accompany it.

This mismatch is visible in the final 100 updates. Mean adapter Q is
`1.82x`, `1.64x`, `1.99x`, and `3.37x` the corresponding robust Q for seeds
16, 24, 32, and 40, despite no comparable return advantage. This does not by
itself prove overestimation caused the return loss, but it proves that the
experiment is not isolating residual adaptation under a common critic
objective.

The legacy RE-SAC sign is not wrong here. `RESAC` still adds the positive
`weight_reg * norm` term to both target and policy values. That code path is
inactive in this adapter experiment because `bapr_v2_reg_weight=0.0`; the
current failure cannot be attributed to reversing the RE-SAC sign.

### 3. Entropy adaptation is under-actuated

In residual mode, the policy returns the frozen base log standard deviation
and only learns a residual mean. The alpha optimizer nevertheless remains
active. Alpha falls below `1e-15` in seed-24 modes 0/2 and seed-32 mode 0, and
seed-32 mode 2 also becomes very small. Those branches are poor, although
alpha alone does not explain all 16 outcomes. A frozen-variance residual should
either freeze source alpha or learn a bounded mode-conditioned log-standard-
deviation residual.

### 4. Stationary-only experts incur a separate switch penalty

Each adapter trains only in one stationary mode. It never receives states
created immediately after another controller and actuator mask. Relative to
its own stationary identity score, switching loses `5.5%` for seed 16, `1.0%`
for seed 24, `8.9%` for seed 32, and `7.2%` for seed 40. Thus seed 24 is mainly
a controller-optimization failure, while seeds 32 and 40 additionally show
cross-controller state-distribution mismatch.

### 5. Mode identity is useful but not identical to control identity

The stationary row-winning controller maps are `[0,3,2,3]`, `[0,2,2,2]`,
`[0,1,2,3]`, and `[0,3,2,3]` for seeds 16, 24, 32, and 40. A post-hoc
row-wise best-controller oracle would recover only `48.6` stationary points on
average; it would still trail robust by `81.3`. Mapping error therefore
contributes but is not the primary gap.

The environment itself follows the intended stochastic-regime semantics.
Actuator masks remain fixed for a 250-step dwell, all modes use the same
per-step Gaussian execution noise, robot morphology and gravity remain fixed,
and the audit routes the exact mode. There is no observation noise and no
per-step mode resampling. Zero termination, hash-complete checkpoints, and
privileged routing rule out termination, resume corruption, and estimator
delay as explanations.

Evaluation-stream noise is also too small to explain the sign changes.
Stationary identity-minus-robust differences have `42.1` mean within-seed event
SD versus `297.3` SD between training-seed means. The dominant variance is
controller training, not sealed event sampling.

### 6. Minor initialization reproducibility issue

Four mode branches within a training seed were intended to instantiate the
same zero-output residual initialization. This is true for 14/16 branches.
Seed-16 mode 2 and seed-32 mode 3 have different initial residual hidden-layer
hashes after being created under different runtime environments, although
their zero output, frozen base, critic, and target critic all pass exact
bootstrap checks. Both happen to be favorable mode outliers. This cannot
explain the overall negative result, but future experiments should clone one
canonical initialized adapter state rather than instantiate it independently
on heterogeneous nodes.

## Minimal next validation

Do not train an estimator or tune a gate. The next controller experiment should
first remove the diagnosed confounds:

1. add an explicit minimum-target path to BAPR-v2 and match the robust critic
   target exactly;
2. freeze alpha with a frozen log-standard-deviation base, or add a bounded
   conditioned log-standard-deviation residual;
3. start adapters from the completed 8.4M robust controller, or jointly
   continue a shared robust base while routing residual-head gradients by
   mode;
4. train on real 250-step switching rollouts, with mode-balanced replay, so
   each head sees predecessor-controller states;
5. clone one canonical initialized adapter checkpoint to every branch.

Use seeds 24 and 32 first because they distinguish full-actor catch-up from
residual adaptation. Expand to four seeds only if privileged switching beats
the completed robust controller in both seeds without termination. If a
matched-objective, converged-base oracle still cannot exceed robust, then the
remaining diagnosis is genuinely insufficient benchmark headroom and the
environment family, not the estimator, must change.

## Decision

The preregistered gate fails every performance and consistency criterion except
termination noninferiority. Decision:
`reject_independent_adapter_optimization`.

Do not train a learned router for this bank. The positive seed-16 result is a
variance case, not sufficient evidence for an algorithm claim. The next work,
if pursued, is the matched-objective controller validation above rather than
more training of these checkpoints or tuning their router.
