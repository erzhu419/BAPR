# Frozen transient-fallback cross-student protocol

## Question

The development-selected causal fallback `evidence_1p0_k1` passed its first
independent audit with student seed `1811`: it beat both the unchanged learned
student and robust `final_seed_719` on `5/5` events, recovered `102.4%` of
oracle headroom, and used fallback for only `1.088%` of actions. This protocol
tests whether that effect is a general switch-local mechanism or a favorable
student initialization.

## Frozen inputs

No threshold, model, estimator, environment, or dwell schedule is selected in
this audit. The following are hash-frozen before execution:

- fallback `evidence_1p0_k1`: evidence threshold `1.0`, entry/exit confidence
  `0.60/0.90`, one supporting transition before leaving fallback;
- `mode_heads` students with seeds `1709`, `1811`, and `1901`;
- robust controller `final_seed_719`;
- the causal system-identification estimator and combined-ten teacher bundle;
- the complete v1 selection and independent-analysis artifacts.

The three students all passed the earlier closed-loop development audit, but
only seed `1811` was selected there. This protocol evaluates all three and
forbids selecting a winner from the new returns.

## Audit split

Five new event seeds `104301,104331,104367,104399,104451` are disjoint from
all training, DAgger, validation, prior development audit, confirmation,
context-ablation, fallback-screen, and fallback-audit events. For every
student/event pair, the same episode trajectories compare:

- robust `final_seed_719`;
- the student's causal learned posterior;
- the student's true-mode oracle context;
- the frozen causal fallback.

This creates 15 checkpoint-only CPU audits and one file-gated aggregate. There
are no optimizer updates or GPU tasks.

## Gate

Each student seed passes only if fallback:

- beats its unchanged learned arm with at least `4/5` event wins and a positive
  event-clustered 95% t interval;
- beats robust 719 by the same rule;
- recovers at least `50%` of that student's oracle headroom over robust 719;
- uses fallback for no more than `20%` of actions;
- has zero switching termination.

The cross-initialization gate requires `3/3` students to pass. A failed seed
cannot be dropped or replaced. Passing freezes explicit causal fallback as the
deployable mechanism and permits a separate development experiment asking
whether stale/soft-belief training improves it. Failure stops that training
branch and redirects analysis to the failed initialization.

## Execution

The 16-task scheduler graph was submitted atomically as `t64930-t64945`.
Tasks `t64930-t64944` are the 15 paired CPU audits, ordered by student seeds
`1709,1811,1901` and then the five registered event seeds. Task `t64945` is
the aggregate and waits for all 15 immutable audit manifests.

Every task has `vram=0`; producers request 32 CPU cores and are restricted to
`node001-node006`. No GPU node, Slurm job, or auto-adopt path is used. Initial
dispatch placed the seed-1709 block on `node004`; its log passed all frozen
hash checks, loaded the registered 5.6M-step controller checkpoints, and began
the paired evaluation normally. The remaining blocks are scheduler-managed
and may reroute among the same six CPU nodes.

## Result

All tasks `t64930-t64945` completed without retry. All 15 audit manifests and
the aggregate validate against the frozen student, estimator, selection, and
v1-analysis hashes. The strict cross-initialization gate fails with `1/3`
student seeds passing:

| student | fallback - robust 719 | fallback - learned | oracle recovery | fallback use | result |
|---:|---:|---:|---:|---:|:---:|
| 1709 | `+88.4`, 5/5, `[+58.9,+118.0]` | `+66.7`, 5/5, `[+34.4,+98.9]` | 54.6% | 1.16% | pass |
| 1811 | `+78.5`, 5/5, `[+58.7,+98.3]` | `+59.6`, 5/5, `[+12.5,+106.8]` | 46.5% | 1.14% | fail |
| 1901 | `+79.3`, 5/5, `[+51.4,+107.1]` | `+142.9`, 5/5, `[-14.8,+300.7]` | 63.6% | 1.18% | fail |

Seed `1811` fails only the preregistered 50% oracle-headroom threshold. Seed
`1901` fails only the positive clustered interval against its learned arm:
all five event deltas are positive, but event `104451` contains an unusually
large `+365.8` recovery from a learned-policy collapse, widening the interval.
No termination occurs in any arm used by the gate.

The gate must remain failed; thresholds cannot be relaxed after seeing the
audit. The complementary mechanism result is nevertheless clear and should
also be retained: fallback beats robust 719 on all `15/15` student-event
clusters, and each student's event-clustered interval is strictly positive.
Across students, mean fallback/robust returns are `2002.6/1920.6`, a descriptive
gain of `+82.1`, while only about `1.15%` of actions use fallback.

## Decision

This audit does not authorize stale/soft-belief student training. It shows
that explicit causal fallback is a reproducible improvement over the strongest
frozen robust controller, but it does not reproduce the v1 claim of recovering
at least half of oracle headroom for every initialization under every frozen
statistical gate.

The next admissible step is checkpoint-only causal-ceiling diagnosis, not
another optimizer sweep. On new events, compare the same students under
instant oracle context, oracle context delayed by `1,2,5,10` actions after a
hidden switch, and the frozen fallback. If fallback is near the one-step
delayed oracle, most residual loss is the unavoidable first action before the
new mode is observable. A material gap after one-step revelation instead
isolates reducible gate/estimator behavior and would justify a separately
registered transient-training branch.
