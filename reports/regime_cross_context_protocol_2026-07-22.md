# Event-grouped RegimeSAC cross-context protocol

## Question

The equal-budget headroom experiment left two explanations entangled:

1. the structured stochastic environments may provide too little useful
   adaptation headroom; or
2. the shared one-hot-conditioned controller may have learned harmful context
   branches or suffered negative transfer, even with true mode labels.

This diagnostic changes no training code or checkpoint. It re-evaluates the
completed robust and oracle RegimeSAC checkpoints.

## Event-grouped design

One scheduler task is one environment, training seed, and sealed event seed.
Inside that single process and runtime, the task evaluates:

- the separately trained robust checkpoint;
- oracle checkpoint with true current-mode one-hot;
- oracle checkpoint with an all-zero vector;
- oracle checkpoint with fixed_0 through fixed_3;
- oracle checkpoint with cyclically wrong context, mapping mode m to
  (m + 1) mod 4.

Grouping all eight evaluations inside one runtime prevents Python, JAX, CPU,
or compiler differences from contaminating paired controller/context
comparisons. The cyclic condition is deterministic and does not add per-step
random context noise.

## Frozen evaluation

- Environments: HalfCheetah-v2, Ant-v2, Walker2d-v2.
- Training seeds: 8, 16, 24, 32, 40.
- Event seeds: 73100, 73200, 73300, 73400, 73500.
- Stationary evaluation: four test modes, five strict 1000-step episodes each.
- Switching evaluation: five strict 1000-step streams, dwell 250 steps.
- Checkpoint budget: iter 1399 complete, 5.6M environment steps and 350k
  updates.
- Inference unit: training seed. Event seeds are averaged within seed.

The 75 CPU jobs are split by environment, training seed, and event seed. They
use scheduler nodes node001-node006, stage both compact controller bundles,
and perform no optimizer update or checkpoint rewrite.

## Protocol sanity checks

- Within each event runtime, stationary true mode m must reproduce fixed_m.
- Within each event runtime, stationary cyclic mode m must reproduce
  fixed_(m+1).
- Task identities and switching sequences must match across all eight cases.
- Every checkpoint must be iter 1399 complete with exactly 5.6M steps.

Any failed identity check invalidates aggregation.

## Interpretation rules

- If zero significantly beats true and approaches the robust checkpoint, the
  learned conditional branch is harmful.
- If zero also remains at least 10% below robust and fewer than 3/4 stationary
  rows are diagonal-optimal, shared conditioned training has negative
  transfer or data-fragmentation failure.
- If true, zero, fixed, and cyclic are nearly equal, the policy ignored context
  or the environment provides little specialization signal.
- Near-universal termination is a protocol failure, not evidence against
  adaptation.

Dynamic oracle headroom is confirmed only if true context:

1. beats robust by at least 10% with a positive paired 95% confidence interval;
2. beats the best fixed context by at least 10% with a positive paired 95%
   confidence interval; and
3. is diagonal-optimal in at least 3/4 stationary modes.

No learned estimator should be trained before an environment passes this
upper-bound test.
