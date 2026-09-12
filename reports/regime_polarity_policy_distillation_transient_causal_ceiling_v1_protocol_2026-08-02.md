# Transient delayed-oracle causal-ceiling protocol

## Motivation

The frozen fallback cross-student gate formally failed, but fallback still beat
robust `final_seed_719` for every student initialization and all `15/15`
student-event clusters. The unresolved question is whether its remaining gap
to zero-delay true-mode oracle is reducible after observing a transition or is
mostly the unavoidable cost of acting once before a hidden switch can be
observed.

## Frozen design

This is a checkpoint-only mechanism diagnostic. It freezes all three existing
`mode_heads` students, robust 719, the system-ID estimator, the
`evidence_1p0_k1` fallback, the environment, and the completed cross-student
analysis. It cannot select a student, gate threshold, or environment setting.

Five new event seeds `105301,105331,105367,105399,105451` are disjoint from all
earlier training and evaluation splits. Every student/event pair evaluates the
same mode stream under:

- robust 719 and the unchanged causal learned posterior;
- zero-delay true-mode oracle;
- true-mode oracle delayed by `1,2,5,10` actions after each hidden switch;
- frozen causal fallback.

For delayed oracle, the old true context remains active for exactly the
registered number of actions after a physical-mode switch. Only after those
transitions does the diagnostic reveal the new true context. In particular,
delay 1 represents an idealized causal estimator that cannot predict the
hidden switch before acting, but identifies it perfectly from the first new
transition. These delayed arms use true mode only as explicitly labeled
nondeployable upper bounds.

## Decision rule

For each student, the primary reducible gap is paired `delay1 - fallback` over
five event clusters. A student has a material reducible post-observation gap
only when:

- delay 1 wins at least `4/5` events;
- its event-clustered 95% t interval is strictly positive;
- its mean advantage is at least `2%` of absolute robust-719 return.

Stale/soft-belief transient training is authorized only if at least `2/3`
students have such a gap and every registered arm has zero termination.
Otherwise the explicit fallback is frozen as the deployable mechanism and the
zero-delay oracle gap is reported as noncausal or initialization-unstable
headroom rather than a training target.

## Execution graph

The graph contains 15 independent 32-core CPU audits and one file-gated
aggregate. Every task has `vram=0` and is restricted to `node001-node006`.
There are no optimizer updates, GPU claims, Slurm jobs, or auto-adopted tasks.

The graph was submitted atomically as `t64954-t64969`. Tasks
`t64954-t64968` are ordered by student seeds `1709,1811,1901` and then the
five registered event seeds. Task `t64969` is the aggregate and waits for all
15 immutable audit manifests. Only producer tasks were actively dispatched;
the scheduler controls staging and rerouting among the six allowed CPU nodes.

## Result

All 16 tasks completed without a failed audit. The 15 immutable manifests,
paired mode streams, frozen checkpoint records, and aggregate schema validate;
all registered arms have zero termination.

| student | robust 719 | fallback | delay 1 | delay 2 | delay 5 | delay 10 | zero-delay oracle | fallback recovery of delay-1 headroom |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1709 | 1913.9 | 2032.3 | 2064.0 | 2042.8 | 1969.0 | 1742.5 | 2085.4 | 78.9% |
| 1811 | 1913.9 | 2005.8 | 2041.3 | 2039.1 | 1951.0 | 1690.2 | 2058.4 | 72.2% |
| 1901 | 1913.9 | 2003.1 | 2051.6 | 2035.6 | 1975.2 | 1649.3 | 2053.7 | 64.8% |

Delay 1 retains `87.6%`, `88.2%`, and `98.5%` of the zero-delay oracle
headroom for students `1709`, `1811`, and `1901`. Therefore the unavoidable
first action after an unannounced switch is not the dominant loss. The frozen
fallback nevertheless already recovers `64.8%-78.9%` of the deployable
delay-1 headroom.

The paired `delay1 - fallback` margins are:

- seed `1709`: `+31.7`, `4/5` wins, 95% interval `[-35.3,+98.7]`;
- seed `1811`: `+35.5`, `5/5` wins, 95% interval `[-19.1,+90.1]`;
- seed `1901`: `+48.5`, `5/5` wins, 95% interval `[+8.9,+88.1]`.

Only seed `1901` satisfies the registered win, interval, and 2%-of-robust
margin gates. The required cross-initialization result is therefore `1/3`,
below the frozen `2/3` authorization threshold. Stale/soft-belief transient
training is not authorized.

The latency ladder also bounds the useful response time. Delay 5 remains
better than robust 719 for every student, with mean margins `+55.0`, `+37.0`,
and `+61.3` and strictly positive clustered intervals. Delay 10 falls below
robust by `-171.4`, `-223.8`, and `-264.7`. Adaptation value is real but is
concentrated in the first few post-switch actions; a ten-action response is
too late for this benchmark.

## Decision

Freeze `evidence_1p0_k1` as the deployable mechanism. Across this split it
beats robust 719 by `+118.4`, `+91.9`, and `+89.2`; each comparison wins `5/5`
events and has a strictly positive clustered interval. Do not train another
student, tune another gate, or use the zero-delay oracle as an attainable
target. The remaining work is implementation consolidation and an independent
final comparison after the deployable fallback path is fixed, not another
transient-training sweep.

## Implementation consolidation

The exact audited transition function now lives in
`jax_experiments/common/causal_fallback.py`. The frozen analysis protocol
imports and re-exports those same configuration, state, evidence-margin, and
update objects, rather than maintaining a second implementation. A stateful
`CausalFallbackGate` adds reset and checkpoint round-trip support without
changing the post-transition update semantics. Compilation, behavioral
identity tests, and a rerun of the frozen aggregate all pass.

The selected threshold is calibrated to the frozen expected-action inverse
system-ID estimator. It must not be attached directly to legacy
`BAPRRegime`/`ProbabilisticRegimeContext` likelihoods, whose evidence scale is
different, without a separate calibration protocol. The deployable candidate
is the frozen mode-head student, robust 719, expected-action estimator, and
this common fallback gate as one indivisible stack.
