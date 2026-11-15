# Late-base min-target minimal validation

## Question

The equal-per-controller diagnostic failed on seeds 24 and 32. This protocol
tests whether that failure came from an immature 5.6M-step base, independent
ensemble bootstrapping, entropy-temperature collapse, or inconsistent residual
initialization.

## Sealed design

- Environment: `HalfCheetah-v2`, `structured_channel`, four persistent modes.
- Training seeds: 24 and 32 only.
- Source: the completed 8.4M-step robust controller at next iter 2100.
- Adapter budget: 175 iterations / 0.7M transitions per fixed mode.
- Final adapter budget: next iter 2275 / 9.1M transitions.
- Policy: frozen robust actor plus bounded residual, delta 0.5.
- Critic target: ensemble minimum, matching `RegimeSAC`.
- Entropy temperature: copied exactly from the robust source and frozen.
- Initialization: one canonical adapter checkpoint per seed, cloned byte for
  byte to all four mode branches.
- Evaluation: five held-out event streams per training seed; strict stationary
  and switching horizons; identity and all fixed controller maps.

This remains a diagnostic upper bound. The four-controller bank consumes 2.8M
post-fork transitions in aggregate and is not compute-matched to one robust
controller.

## Fail-fast gate

The stationary mechanism passes only if:

1. The copied frozen base matches the source robust controller within 1% on
   both seeds.
2. The identity-routed adapter exceeds the 8.4M robust source by at least 5%
   on both seeds.
3. The correct fixed-mode controller is stationary-optimal in at least three
   of four mode rows for each seed.

Switching return and termination are reported but do not gate this first
stage. A pass authorizes switch-matched distribution training. A failure means
the fixed-mode residual optimization still lacks reproducible headroom, so a
learned estimator should not be trained.

## Task graph

- Canonical preparation: 2 CPU tasks, no rollout or gradient update.
- Fixed-mode training: 8 GPU tasks, measured admission request 2300 MiB.
- Strict audit: 10 CPU tasks.
- Aggregate analysis: 1 CPU task.

GPU nodes are `local`, `jtl110gpu`, `jtl110gpu2`, and `node007`.
`jtl311linux` is explicitly excluded. All stages are file-gated and use the
scheduler checkpoint migration/resume path.

## Status

Seed 24 canonical preparation passed locally:

- source boundary: iter 2100 / 8.4M steps / 525,000 updates;
- copied `log_alpha`: -3.4167187213897705;
- empty replay and reset optimizer states;
- source actor, critic, target critic, and zero-residual equivalence validated.

Submitted through the scheduler:

- canonical seed 32: `t51743` (complete);
- seed 24 fixed-mode training: `t51770` (mode 0) and
  `t51766`-`t51768` (modes 1-3);
- seed 32 fixed-mode training: `t51748`-`t51751`;
- strict audits: `t51752`-`t51761`;
- aggregate analysis: `t51762`.

Seed 24 canonical preparation was already complete locally, so no duplicate
preparation task was submitted. All downstream tasks remain file-gated.

The original seed 24 launches are superseded:

- `t51744` inherited `JAX_PLATFORMS=cpu` from the submission shell. It was
  stopped, and its partial output was quarantined under
  `mode_0_cpu_invalid_t51744`; no downstream task consumes it.
- `t51745`-`t51747` and their first retries `t51763`-`t51765` failed before
  training because a raw checkpoint budget probe did not install the existing
  Flax `VariableState` compatibility patch. The branch runner now installs
  that patch before every raw checkpoint read.
- Every sealed GPU command now sets `JAX_PLATFORMS=cuda` explicitly. The
  replacement logs show CUDA devices and the requested min-target/frozen-alpha
  configuration before training begins.

At the latest status check, all eight valid training branches were running.
The ten audit tasks and aggregate task were still queued behind their expected
output files, so evaluation cannot start on partial branches.

## Final result

All eight valid branches reached iter 2275 / 9.1M transitions / 568,750
updates. All ten strict audits and aggregate `t51762` completed and passed
manifest, budget, and frozen-base validation.

| Seed | Late robust stationary | Identity stationary | Gain | Late robust switching | Identity switching | Gain | Diagonal optimal |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 2009.1 | 2247.0 | +11.8% | 1935.2 | 2101.7 | +8.6% | 4/4 |
| 32 | 2410.9 | 2485.4 | +3.1% | 2384.8 | 2418.1 | +1.4% | 2/4 |

The frozen-base copy error is exactly zero for both seeds and switching
termination remains zero. The mean gain is +7.5% stationary and +5.0%
switching, but the preregistered gate fails because seed 32 does not reach 5%
stationary gain and only two of four correct-mode controllers are optimal.

The per-mode stationary comparison explains the failure:

| Seed | Mode 0 gain | Mode 1 gain | Mode 2 gain | Mode 3 gain |
|---:|---:|---:|---:|---:|
| 24 | +8.6% | +8.9% | +9.5% | +19.2% |
| 32 | +0.4% | -0.9% | +1.9% | +9.6% |

Five held-out event streams agree on the stationary direction, but seed 32's
switching improvement is not stable: its mean difference is +33.4 return with
an event-stream 95% interval of approximately [-31.9, +98.7]. Event streams
are repeated evaluations rather than independent training seeds, so this
interval is descriptive only.

Decision: `reject_latebase_fixed_mode_residual_headroom`. Starting from a
mature base, matching the minimum critic target, freezing inherited alpha, and
using canonical initialization remove the earlier catastrophic losses. They
do not create reproducible privileged-mode headroom across policy seeds. Since
the true-mode oracle itself misses the gate despite consuming extra data, a
learned estimator or switch detector is not authorized under this benchmark.

Machine-readable results are in
`jax_experiments/results_regime_adapter_latebase_min_analysis_v1/analysis.json`.
