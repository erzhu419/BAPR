# RE-SAC Norm-Shift Causality

Frozen development experiment: HalfCheetah spring, unchanged V39 persistent
actuator polarity/noise0.02/dwell250/horizon1000 and strict stationary/switching
event windows. Fresh paired policy seeds110003/110023/110043; no historical pooling.
Each arm trains from scratch for2100 iterations,8.4M interactions,525000 updates.
Retain five width256 critics, LR0.0003, gamma0.99, tau0.005, beta-2, OOD0.01,
BC0.001, independent targets, actor period1, clipping0, no anchor/annealing/EMA.
Only the positive norm shift changes: per-head0.01, shared ensemble-mean0.01,
or weight_reg0. Hidden biases/all kernels are included; output bias is excluded
in every arm. Both target and actor use the same treatment; no sign reversal.

Primary mechanism contrast: shared minus per-head, examining head-offset variance
and final switching return. No-shift minus per-head and shared minus no-shift
separate offset contamination from common reward-shift/scale effects. Report all
three paired seed differences and descriptive95% t intervals; stationary results
remain separate. Final checkpoints only, no actor/checkpoint/seed selection.
These are exploratory mechanism tests, not an original-Torch reproduction or a
new BAPR superiority claim. Do not align update frequency/LayerNorm/clipping here.

Pinned JAX0.4.38/Brax0.10.5/Flax0.10.4/MuJoCo3.1.6. Scheduler-only DAG:
one GPU smoke, nine per-seed GPU trainers, three CPU audits, one CPU aggregate.
No local/Windows training. Retry/migration restores only this treatment's full
checkpoint; never continue a V39 checkpoint. Local synchronization includes only
final actor bundles, metadata and compact diagnostic/audit JSON, not critics,
optimizers, replay or training checkpoints. Existing validated fused sampling is
shared by all arms without reducing interactions or updates.
