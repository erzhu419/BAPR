# Polarity policy-distillation protocol

## Motivation

The checkpoint-only ensemble screen passed its preregistered gate in both
controller groups. The causal expected-action posterior recovers about 94% of
median-oracle ensemble headroom, and learned median switching return is
`2062.3` for the development controllers and `1970.9` for the final
controllers. Directly averaging independently trained robust actions,
however, is catastrophically weak. All subsequent comparisons therefore use
the distribution of individual robust-controller returns, never the robust
action ensemble.

This is retrospective algorithm development, not confirmation. It tests
whether stable ensemble behavior can be compressed into one deployable policy.

## Frozen teacher and student

- Environment: `HalfCheetah-v2`, persistent four-mode `actuator_polarity`.
- Frozen causal estimator: expected-action system-ID v4; no mode ID, executed
  action, gain vector, or switch clock is available online.
- Teacher reduction: coordinate-wise median deterministic action.
- Teacher groups: development five, final five, and their fixed combined ten.
- Student: one two-layer 256-unit `GaussianPolicy` receiving observation and
  the soft four-mode posterior. Only its deterministic action is audited.
- Student seeds: `1409,1511,1601` for every teacher group.

The student is trained only by supervised action imitation. Initial data cover
oracle-teacher, learned-teacher, and individual-robust state distributions on
event seeds `98001-98003`. Two DAgger rounds collect student-visited states on
`98101-98102` and relabel them with the same frozen median teacher. Validation
uses `98901-98902`; strict control audit uses untouched event seeds
`99001-99003`.

Reward is passed only to the already frozen causal estimator. It is not an
optimization target for the student. No critic, policy gradient, BOCD, LCB,
residual gate, or environment-mode label enters student training or deployment.

## Decision rule

For each student, the primary baseline is the episode-paired mean return of
the individual robust controllers. A student passes only if:

1. it beats that robust population on all three audit event seeds;
2. it recovers at least 80% of learned-median teacher headroom over that
   population;
3. switching termination rate is zero.

A teacher group passes when at least two of three student initializations pass
and the seed selected solely by frozen supervised validation loss also passes.
Only the combined-ten group can promote the method to a new five-seed
confirmation. Development/final subgroup success alone is diagnostic and
cannot be selected post hoc.

The audit also reports performance against the best individual robust
controller, oracle median teacher, per-mode stationary return, and all student
initializations. Runtime ensemble scores are an upper bound, not the proposed
deployment algorithm.

## Scheduler graph

The graph contains nine GPU distillation tasks, 27 file-gated CPU audits, and
one file-gated aggregate. GPU placement is restricted to `jtl110gpu` and
`jtl110gpu2`; CPU evaluation is restricted to `node001-node006`. Each new GPU
task declares 1800 MB rather than an unmeasured 4-8 GB default. Slurm,
auto-adopt, `node007`, `jtl311linux`, and local GPU placement are excluded.

The original graph is `t64574-t64610`. The effective training lineage after
launch recovery is:

- development: `t64574-t64576`;
- final: `t64614-t64616`, clean retries of `t64577-t64579`;
- combined: `t64580`, `t64581`, and `t64618`, where `t64618` is the clean
  retry lineage of `t64582 -> t64617`;
- strict CPU audits: `t64583-t64609`;
- aggregate: `t64610`.

The failed initial lineages are excluded from all result aggregation. They
exposed a scheduler defect rather than an algorithm failure: multi-directory
launch-input staging published its aggregate cache key after the first rsync,
allowing a process to read a mixture of old and newly staged frozen bundles.
The scheduler now publishes that key only after every declared directory has
synced successfully; a later-directory failure publishes no success marker.
The watcher was restarted with this fix, and 68 focused staging/dispatch tests
passed before the clean retries were allowed to proceed.

## Result

All required outputs are complete and synced: nine student models, 27 strict
CPU audits, and the aggregate result. Aggregate task `t64610` finished on
`node004`; the five failed task records belong only to superseded launch
lineages and are not missing experiment arms.

The development, final, and combined teacher groups passed with `3/3`, `2/3`,
and `3/3` student initializations, respectively. The seeds selected only by
frozen supervised validation also passed in every group, so the combined
teacher satisfies the preregistered retrospective promotion rule.

The validation-selected combined student is seed `1511`. Its strict switching
return is `1954.2`, compared with `1182.9` for the episode-paired robust
controller population (`+771.4`; wins `15/15` episodes) and `1921.5` for the
best individual robust controller (`+32.7`; wins `11/15` episodes). Its three
event deltas against the best robust controller are `+19.3`, `+40.4`, and
`+38.3`. Switching termination is zero. Its stationary returns in modes 0-3
are `2009.2`, `2076.0`, `1962.3`, and `2240.9`, so its worst stationary mode
is `1962.3`.

The subgroup result is less uniform. The development-selected seed `1511`
beats its best robust controller by `+681.1`, while the final-selected seed
`1601` is effectively tied with its best robust controller (`-3.0`) despite
beating the final robust population by `+587.2`. Therefore the screen supports
stable compression of the combined median teacher and a large improvement
over the robust-controller population mean. It does not yet establish a
statistically reliable advantage over the strongest robust controller, nor a
final comparison with SAC, ESCP, or RE-SAC.

The next admissible experiment is a genuinely new five-seed confirmation with
combined seed `1511` frozen by validation alone. The current audit returns must
not be used to choose another student or tune the confirmation protocol.
Machine-readable and rendered aggregate outputs are in
`jax_experiments/results_regime_polarity_policy_distillation_analysis_v1/`.
