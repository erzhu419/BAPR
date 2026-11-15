# Frozen mode-head context-ablation protocol

## Question

The frozen `mode_heads/student_seed_1811` confirmation beat robust
`final_seed_719` by `+22.0` on average and won `4/5` events, but its clustered
95% interval crossed zero. This diagnostic asks whether the student's strong
return is caused by dynamic posterior-conditioned behavior or by a generally
stronger robust policy learned during compression.

This is post-confirmation analysis. It cannot select another model, seed,
phase, threshold, or confirmation event, and it cannot reverse the frozen
confirmation failure.

## Frozen artifacts and split

The student manifest, parameters, and development selection analysis retain
the exact SHA-256 records frozen by the v2 confirmation. Five previously unused
diagnostic event seeds are fixed as
`102301,102331,102367,102397,102451`. They are disjoint from every estimator,
training, DAgger, validation, audit, and confirmation split.

## Paired arms

Every event uses identical stationary tasks and switching mode streams for:

- robust `final_seed_719`;
- oracle-median and causal learned-median teachers;
- frozen student with causal learned posterior;
- frozen student with true-mode oracle context;
- frozen student with uniform context;
- frozen student with fixed contexts `0-3`;
- frozen student with cyclically wrong and event-fixed shuffled contexts.

The learned arm remains strictly causal. Only explicit diagnostic upper/lower
bounds receive mode-derived contexts. No parameter is updated.

## Interpretation

The student has genuine dynamic context value only if oracle context beats
uniform and every fixed context with positive event-clustered intervals.
Causal deployment realizes that value only if learned posterior also clears
the same comparisons. The stationary matrix additionally reports how often
the matching mode head is best among fixed heads.

If oracle context fails, the student's strength is robust-policy compression,
not adaptation. If oracle passes but learned fails, the heads specialize but
the causal deployment path does not realize that specialization. Either result
remains explanatory and cannot reopen model selection.

## Execution

Five independent 32-core CPU audits feed one file-gated aggregate. Tasks are
restricted to `node001-node006`, submitted only through scheduler, and use no
Slurm, auto-adopt, training, or GPU resource.

The graph was submitted atomically on 2026-08-02. Audits are `t64890-t64894`
and aggregate is `t64895`. All audits launched on `node004`; the aggregate is
gated on the five immutable audit manifests.

## Result

All five audits and the aggregate completed, and every audit manifest validates
against the frozen student, estimator, teacher ensemble, and event identity.
The result is classified as `causal_posterior_realizes_context_value`.

The frozen student scores `2046.7` with true-mode oracle context and `1916.3`
with its causal learned posterior, versus `1933.9` for preregistered robust
`final_seed_719`. Oracle context beats robust 719 by `+112.8` on `5/5` event
seeds with clustered 95% interval `[+75.7,+149.9]`. The learned posterior
instead trails robust 719 by `-17.6`, wins `2/5`, and has interval
`[-108.2,+73.0]`.

This is genuine context specialization rather than an accidentally stronger
unconditional policy. Oracle and learned contexts each beat uniform and every
fixed context with strictly positive clustered intervals, and the matching
fixed head is stationary-optimal in all `4/4` modes. Uniform, cyclic, shuffled,
and mismatched fixed contexts largely destroy control. Student/teacher parity
also shows that policy compression is no longer the limiting stage: oracle
student/teacher returns are `2046.7/2053.6`, while learned student/teacher
returns are `1916.3/1895.8`.

The remaining loss is concentrated in causal mode acquisition. Learned context
trails oracle context by `-130.4` on `0/5` event seeds, with clustered interval
`[-236.7,-24.0]`, despite `0.9984` mode accuracy and median/P90 switch delays of
`9.4/16.6` actions. Oracle headroom over robust 719 is only about `5.8%`, so the
short post-switch mismatch consumes all of the available adaptation advantage.

This post-confirmation diagnostic explains the mechanism but does not reverse
the failed frozen confirmation. No additional model selection, seed extension,
baseline claim sweep, or GPU training is authorized from these event seeds.
