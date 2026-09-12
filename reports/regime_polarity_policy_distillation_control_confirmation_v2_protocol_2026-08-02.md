# Frozen mode-head policy-compression confirmation protocol

## Frozen candidate

The closed-loop policy-compression v2 development sweep selected
`mode_heads/student_seed_1811` solely by the independent control-validation
score. The development aggregate then passed its preregistered promotion gate:
all three mode-head initializations passed and the control-selected student
also passed. No development-audit return is used to change the selected seed.

The following artifacts were frozen before evaluating this confirmation split:

| artifact | size | SHA-256 |
|---|---:|---|
| model manifest | 26975 | `4f3f8cce988e9e546b6009bbaefb3a235f8085c91955fdc7da957c28b744ec75` |
| student parameters | 287370 | `2f54b0a40352a68349b52ee36ede26f0bbe97085e37b595ed9f7c37b400104d8` |
| development selection analysis | 53210 | `a2d9aed89262f687d60042837d698fd5968627d8e120262cc11a989a5f1947d9` |

## Untouched split

The five event seeds are `102019, 102043, 102069, 102103, 102151`. They are
disjoint from estimator fitting, initial distillation, DAgger collection,
supervised validation, control validation, development audit, ensemble
diagnostics, and the previous frozen confirmation. The previous confirmation
seeds `100019,100043,100069,100103,100151` remain sealed and are not reused.

Every event evaluates paired strict-horizon stationary and switching episodes
for the frozen student, the causal learned-median teacher, the true-mode oracle
median teacher, all ten individual robust controllers, and the preregistered
fixed comparator `robust_final_seed_719`. Online student and estimator inputs
remain causal; mode ID, executed action, gain, and switch clock are forbidden.

## Confirmation gate

The candidate passes only if all conditions hold:

1. Student beats the ten-controller robust population on `5/5` events and the
   event-clustered 95% t interval is positive.
2. Student beats fixed robust `719` on at least `4/5` events and the clustered
   interval is positive.
3. Student recovers at least `80%` of learned-teacher headroom over the robust
   population.
4. Switching termination rate is zero.

Failure ends this compression direction; confirmation events must not be used
for another model, seed, phase, or threshold selection. Passing authorizes the
next compute-matched SAC/ESCP/RE-SAC comparison under the identical benchmark.

## Execution

This graph contains five independent 32-core CPU checkpoint audits followed by
one file-gated aggregate. It performs no policy, estimator, or environment
training. Tasks are submitted only through scheduler, restricted to
`node001-node006`, with resume/reroute enabled and no Slurm or auto-adopt path.

The graph was submitted atomically on 2026-08-02. Audits are `t64877-t64881`
and aggregate is `t64882`. All five audits launched on `node004`; the aggregate
remains gated on their five immutable manifests.
