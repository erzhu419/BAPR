# Frozen final BAPR mechanism audit protocol v2

## Purpose

This checkpoint-only audit asks whether the final five BAPR deployments owe
their return to causal mode adaptation, rather than to unconditional policy
compression or a favorable single initialization. It is a post-confirmation
diagnostic and cannot select another student, estimator, fallback threshold,
or environment configuration.

Version 2 supersedes a cancelled v1 submission. It hashes the full runtime
source closure, including environment construction, Brax physics, model and
checkpoint loaders, policy and critic definitions, estimator code, scheduler
DAG, and this protocol report.

## Frozen inputs

- Final BAPR mode-head students: seeds `2009, 2113, 2213, 2311, 2417`.
- Frozen robust controller: seed `719`.
- Frozen expected-action estimator v4 and causal fallback configuration.
- Five untouched diagnostic event streams: `107503, 107537, 107579, 107621,
  107659`.
- Environment: HalfCheetah persistent actuator-polarity regimes, dwell 250,
  strict 1000-step horizon.

The registration hashes every runtime source, model, estimator, robust
checkpoint, and upstream final-result artifact before submission.

## Arms

Stationary tests compare robust, learned posterior with and without fallback,
true context, zero context, uniform context, and fixed contexts 0-3.
Switching tests add cyclic-wrong, shuffled-wrong, and true context delayed by
1, 5, 10, 25, or 50 actions after every hidden physical switch. All arms use
the same mode sequence and stochastic noise stream within each event seed.

True and delayed-true context arms are diagnostic only. Deployable arms never
receive mode id, action gain, executed action, or switch clock.

## Measurements

- Strict switching return and termination rate.
- Full stationary context-by-mode matrix.
- Learned posterior accuracy, Brier score, and switch-detection delay.
- Fallback action fraction.
- Per-mode reward and post-switch reward in registered transient bins.
- Event-clustered paired 95% intervals over the five untouched event streams.

## Frozen gate

A student has context specialization only if at least three of four stationary
modes are diagonal-optimal and dynamic true context beats both the best fixed
switching envelope and zero context with at least four of five event wins and
a positive clustered interval. A student has deployable causal value only if
learned posterior plus fallback beats robust seed 719 under the same paired
criterion.

The checkpoint family passes only if at least four of five students satisfy
both gates and every evaluated arm has zero termination rate. Incremental
fallback value over learned-only is reported separately and is not silently
folded into the causal-value claim.

## Consequence

Only a pass may release the pre-registered fresh ten-seed deployment
confirmation. A failure freezes this checkpoint family as evidence that the
observed return does not establish a repeatable causal adaptation mechanism.
