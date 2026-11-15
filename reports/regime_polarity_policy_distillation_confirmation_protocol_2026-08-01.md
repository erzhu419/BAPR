# Frozen policy-distillation confirmation protocol

## Frozen candidate

The candidate is the combined-ten median-teacher student with training seed
`1511`. It was selected only by the supervised validation loss before the
retrospective control audit was read. No model, estimator, controller,
threshold, environment setting, or action-reduction rule may change during
this confirmation.

- student manifest SHA-256:
  `20e180c458b5a57975c9b06d5f4a54796eb5ebe941c62b93a856110d8f328e7a`;
- student parameter SHA-256:
  `cf28fb30dde6b58346319087763f7e51cacf0b603f62cae7ebbf60fc50dc4d25`;
- validation-selection analysis SHA-256:
  `60353c028c7523ae2bab51385b22807bf485b30053cb98804378e02b8415304b`;
- environment: `HalfCheetah-v2`, persistent four-mode `actuator_polarity`;
- causal estimator: frozen expected-action system-ID v4;
- online student inputs: observation and the soft posterior computed only from
  prior observations, commanded actions, rewards, and next observations.

The privileged mode ID, action gain, executed simulator action, and switch
clock remain forbidden online.

## Independent split

The five confirmation event seeds are fixed as
`100019,100043,100069,100103,100151`. They do not occur in the ensemble
diagnostic, student training, DAgger collection, supervised validation, or
retrospective control-audit splits.

Each event task evaluates identical paired stationary and switching streams
for the frozen student, all ten individual robust controllers, the learned
median teacher, and the true-mode oracle median teacher. There is no training,
checkpoint update, model selection, or hyperparameter search in this stage.

## Preregistered decision

The primary strongest-controller comparator is frozen as
`robust_final_seed_719`, which was the strongest individual robust controller
in the completed retrospective analysis. It is not reselected from these five
confirmation seeds. Event seed, rather than episode, is the independent unit
for the two-sided 95% Student-t interval (`df=4`, critical value `2.776445`).

Overall confirmation passes only if every gate passes:

1. the student beats the mean of the ten individual robust controllers on all
   five event seeds and its event-clustered 95% interval is above zero;
2. the student beats frozen robust controller `final_seed_719` on at least
   four of five event seeds and its event-clustered 95% interval is above zero;
3. the student recovers at least 80% of learned-median teacher headroom over
   the robust population;
4. student switching termination rate is zero.

A result that passes only the robust-population gates supports stable
ensemble distillation, but not superiority to the strongest robust controller.
Even an overall pass remains specific to this HalfCheetah polarity benchmark;
it does not by itself establish superiority to SAC, ESCP, or RE-SAC.

## Scheduler graph

The graph contains five 32-core CPU audits restricted to `node001-node006`
and one file-gated CPU aggregate. It uses scheduler submission only. GPU
training, Slurm, auto-adopt, `node007`, `jtl311linux`, local execution, and the
two `jtl110gpu` nodes are excluded. Task IDs are recorded after atomic batch
submission.

The graph was submitted atomically at high priority on 2026-08-01:

- confirmation audits: `t64717-t64721`, one task per new event seed;
- file-gated aggregate: `t64722`.

All five audits launched concurrently on `node004`; the aggregate remained
queued behind all five declared audit manifests.

## Result

All six tasks completed and synced. The five audits each finished in about
29-31 minutes, and aggregate task `t64722` completed normally after all five
manifests became available.

| Comparator | Student delta | Event wins | Event-clustered 95% t interval | Gate |
|---|---:|---:|---:|:---:|
| ten-controller robust population | `+720.5` | `5/5` | `[+659.7,+781.4]` | pass |
| frozen robust `final_seed_719` | `-33.6` | `1/5` | `[-103.8,+36.6]` | fail |

The student switching return is `1899.4`, versus `1178.9` for the robust
population and `1933.0` for frozen robust controller `final_seed_719`.
Learned-teacher headroom recovery is `88.7%`, switching termination is zero,
and both corresponding gates pass. The fixed-controller event-win and
positive-interval gates fail, so the preregistered overall confirmation fails.

The failure is not evidence that the environment lacks adaptation headroom or
that the causal estimator failed. Posterior mode accuracy is `0.9978`, median
switch delay is `8.6` actions, P90 delay is `17.7`, and Brier score is
`0.0039`. The oracle median teacher beats frozen robust `719` by `+111.9` on
`5/5` event seeds, with clustered interval `[+72.0,+151.8]`.

A post-confirmation diagnostic, which does not alter the registered decision,
shows that the learned median teacher also beats robust `719`: delta `+58.5`,
`4/5` event wins, clustered interval `[+5.0,+112.1]`. Robust `719` remains the
strongest of all ten individual robust controllers on the new split, so this
is not an artifact of freezing the wrong comparator. In contrast, the student
trails its own learned teacher by `-92.1` on all `0/5` event seeds, with
clustered interval `[-180.3,-4.0]`.

The result therefore localizes the remaining failure to deployable policy
compression: low supervised validation loss and two DAgger rounds did not
preserve the teacher under closed-loop switching state distributions. The
confirmed claim is limited to stable ensemble-teacher adaptation and a large
gain over the robust-controller population. A single distilled BAPR policy is
not confirmed superior to the strongest robust controller.

These five confirmation seeds are now sealed and cannot be used for tuning.
Any next distillation method must be developed on the earlier development and
validation splits, frozen, and evaluated on another untouched confirmation
split.
