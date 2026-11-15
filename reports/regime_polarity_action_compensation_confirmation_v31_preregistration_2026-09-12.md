# V31 canonical action-compensation preregistration

V31 independently confirms the V28 mechanism on five new HalfCheetah policy seeds. It freezes actuator-polarity mode 0 as the sole reference before training; no V31 calibration selects a policy or checkpoint.

## Frozen protocol

- Policy seeds: `87003, 87021, 87039, 87057, 87079`.
- Reference: V21 full-state-final recipe, fixed mode 0, final checkpoint only.
- Estimator: unchanged v5 executed-action posterior, trained on older disjoint policy seeds.
- Evaluation: three new stationary streams and three new balanced switching streams, five 1000-step episodes each.
- Training path: 5.6M robust-source interactions plus 2.8M fixed-mode fine-tuning, 8.4M total per seed.
- Equal-interaction controls: SAC, recurrent ESCP, and RE-SAC each train for 8.4M interactions.
- Synced artifacts: compact evaluation bundles, provenance, logs, and JSON only; no replay buffers or checkpoints.

Mode 0 was fixed from V28 development evidence: its native return exceeded robust SAC on every old policy seed. Those old events are not reused for V31 evaluation.

## Decision

The strong confirmation requires causal compensation to beat equal-budget SAC, ESCP, and RE-SAC with a positive paired mean and 95% interval, at least 4/5 policy-seed wins and 12/15 event wins for each comparator. It must also beat no compensation, recover at least 70% of oracle headroom on 4/5 seeds, retain at least 95% of stationary oracle return on 4/5 seeds, and preserve exact oracle transform equivalence.

A failed gate remains a failed independent confirmation. V31 does not authorize selecting another reference mode, changing the estimator, adding seeds, or tuning on its holdout results.

The claim is limited to HalfCheetah actuator polarity. The exact sign transform does not imply general adaptation to gain loss, changed dynamics, bus uncertainty, or termination safety.
