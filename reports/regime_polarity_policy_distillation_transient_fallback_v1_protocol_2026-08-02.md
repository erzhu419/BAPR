# Causal transient-fallback protocol

## Motivation

The frozen `mode_heads/student_seed_1811` is genuinely context-specialized,
but its causal learned context scores `1916.3` versus `2046.7` with true-mode
context and `1933.9` for robust `final_seed_719`. Its mode accuracy is `0.9984`,
yet median/P90 switch delay is `9.4/16.6` actions. The short post-switch loss
therefore exceeds the entire `+112.8` oracle headroom over robust 719.

This protocol tests whether an immutable robust controller can cover that
transient without changing the environment, estimator, mode heads, teacher,
or any previously sealed result.

## Causal fallback

The adaptive policy remains the hash-frozen
`mode_heads/student_seed_1811`; fallback is the hash-frozen robust
`final_seed_719`. At action time the controller may use only the current causal
posterior and state accumulated from earlier transitions. After each action,
the frozen system-ID model supplies the four one-step mode log likelihoods.

Fallback starts active. It is entered when one-step evidence contradicts the
currently believed mode, the posterior MAP changes, or posterior confidence
drops below `0.60`. It exits only when a single MAP mode has posterior at least
`0.90` and supporting evidence for a registered number of consecutive steps.
Physical mode, actuator gain, executed action, switch clock, and future
transition are unavailable to the gate.

## Two-stage split

Eight fixed configurations cross contradiction thresholds
`0.5,1.0,2.0,4.0` with stability counts `1,3`. Four development events
`103001,103019,103037,103061` select exactly one configuration by mean
switching return minus a fixed termination penalty. Registered order resolves
exact ties.

Five independent events `103301,103331,103367,103399,103451` then evaluate
only that frozen configuration against:

- robust `final_seed_719`;
- the unchanged causal learned student;
- the unchanged true-mode oracle student.

All nine new seeds are disjoint from prior training, DAgger, validation,
development-audit, confirmation, and context-ablation events. Audit events
cannot alter the selected threshold or stability count.

## Continuation gate

Stale/soft-belief student training is authorized only if the selected fallback:

- beats the unchanged learned student with at least `4/5` event wins and a
  positive event-clustered 95% interval;
- beats robust 719 under the same rule;
- recovers at least `50%` of true-oracle headroom over robust 719;
- uses fallback on no more than `20%` of actions;
- has zero switching termination.

Failure stops this branch. Passing authorizes new development-only training
with stale/soft posterior augmentation and the same explicit robust fallback;
it does not reopen either sealed confirmation or authorize baseline claims.

## Execution

The graph contains four CPU development screens, one file-gated selection,
five file-gated independent CPU audits, and one aggregate. There are no GPU
tasks, optimizer updates, Slurm jobs, or auto-adopted processes. Any later GPU
training remains restricted to `jtl311linux`.

The 11-task graph was submitted atomically as `t64913-t64923`: development
screens are `t64913-t64916`, selection is `t64917`, independent audits are
`t64918-t64922`, and aggregate is `t64923`. Only the four screen tasks were
actively dispatched; every later stage is gated by immutable predecessor
manifests.

## Result

All 11 tasks completed and synced. Every frozen student/estimator record,
development-screen manifest, selection manifest, and independent-audit
manifest validates. Development selected `evidence_1p0_k1`: contradiction
threshold `1.0`, posterior entry/exit confidence `0.60/0.90`, and one
supporting transition before leaving fallback.

On independent event seeds `103301,103331,103367,103399,103451`, mean
switching returns are:

| arm | return |
|---|---:|
| robust `final_seed_719` | 1919.2 |
| unchanged causal student | 1929.6 |
| true-mode oracle student | 2032.9 |
| selected causal fallback | 2035.6 |

The selected fallback improves over the unchanged learned student by `+106.0`
with `5/5` event wins and an event-clustered 95% t interval of
`[+67.4,+144.7]`. It improves over robust 719 by `+116.4`, also with `5/5`
wins and interval `[+64.1,+168.7]`. Its `+2.7` difference from the true-mode
oracle is statistically unresolved (`2/5` wins, interval `[-41.0,+46.4]`).
It recovers `102.4%` of oracle headroom while using robust fallback for only
`1.088%` of actions and causing zero termination.

The effect is present in every independent event rather than one outlier:
fallback-minus-learned deltas are `+142.6,+68.0,+83.9,+105.1,+130.6`, and
fallback-minus-robust deltas are `+183.3,+67.8,+103.8,+107.0,+120.0`.
Therefore every preregistered continuation gate passes and
`authorize_belief_augmentation_training=True`.

## Interpretation

This is the first clean result on this benchmark that isolates the remaining
failure to switch-local deployment. The four conditioned heads and causal
posterior are useful away from a switch, but roughly `9-17` stale actions after
a switch erase their modest oracle advantage. Covering only the evidence-poor
transient with the immutable robust controller restores essentially all oracle
performance. No environment change, physical-mode input, future information,
policy update, or estimator update is involved.

The result freezes this fallback configuration as the candidate mechanism. It
does not reopen the earlier failed confirmation or yet support a general
SAC/ESCP/RE-SAC claim: the current audit uses one student initialization. The
next necessary check is a disjoint checkpoint-only audit across all three
existing `mode_heads` student initializations. Stale/soft-belief training is
deferred until that audit establishes that the fallback result is not specific
to student seed `1811`.
