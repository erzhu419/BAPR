# V19 specialist training stability preregistration

## Motivation and frozen boundary

V18 passes the standard SAC, recurrent ESCP, and released-B0 RE-SAC gates, but
does not establish an advantage over the equal-policy-count causal SAC5
baseline. Its failure decomposition points primarily to specialist-bank
quality: seed 81039 has accurate routing but weaker controllers than the SAC5
pool in three of four modes.

V19 is a fresh-seed development experiment. It does not reuse or tune against
the v18 holdouts. The actuator-polarity environment, v5 posterior, routing
logic, dwell time, training budget, and evaluation definitions are unchanged.
Only specialist controller initialization is varied.

## Arms and data

Training seeds are 82003, 82021, and 82039. Calibration, stationary holdout,
and switching holdout streams are disjoint and fixed before training.

The three paired arms share the same freshly trained robust SAC source:

- `actor_only_control`: copy the robust actor; initialize critic, target
  critic, alpha, optimizer states, and replay freshly.
- `full_state`: copy robust actor, critic, target critic, and alpha; reset
  optimizer states and replay.
- `critic_warmup`: use actor-only initialization, but update only critic and
  target critic for the first 25,000 fine-tuning updates. Actor and alpha then
  train normally.

Each source is trained for 1,400 iterations and 5.6M transitions. Each
fixed-mode specialist receives 700 additional iterations and 2.8M transitions.

## Decision rule

A candidate must pass the existing robust-relative bank gate on all three
seeds: at least three of four held-out modes improve by 5%, safe switching
improves by 10% on all three switching streams, and termination does not
increase.

Against paired `actor_only_control`, a candidate must have positive mean safe
switching difference, win at least two of three seeds and six of nine switching
events, improve the worst-seed safe gain, retain at least 95% of control's
stationary specialist return in every seed-mode cell, and not reduce the total
number of held-out mode passes.

If both candidates pass, select the higher mean safe switching return. Prefer
`full_state` when its return is within 2% of `critic_warmup`, because it does not
introduce a new update schedule. No result from this three-seed development
screen upgrades the v18 claim; a selected arm must later face a new five-seed
equal-budget confirmation.

## Artifact policy

Scheduler tasks synchronize logs, JSON, policy parameters, and one compact
robust-source controller-state bundle. Replay buffers and full checkpoints stay
on their execution nodes and are used only for scheduler-managed resume.
