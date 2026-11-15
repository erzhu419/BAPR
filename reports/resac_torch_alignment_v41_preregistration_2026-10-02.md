# RE-SAC Torch-Equation Alignment

Fresh development seeds111003/111023/111043. Unchanged HalfCheetah spring
persistent actuator polarity/noise0.02/dwell250/horizon1000 and V39/V40 strict
stationary/switching event windows. Each arm:2100 iterations,8.4M interactions,
525000 critic-update calls. No pooling or checkpoint/actor selection.

Arms: V40 per-head positive fork; Torch-equation-aligned positive shift;
Torch-equation-aligned no shift. Aligned arms use10 critics, first-layer shared
LayerNorm epsilon1e-5, original Torch weight/bias initializations, small actor
heads, alpha initially1 with its original exp-alpha gradient before the critic,
the same actor sample reused for alpha/policy, unbiased ensemble std, actor
period2 and global clipping1. Both retain OOD0.01/BC0.001 and independent targets.
Only weight_reg differs between aligned arms:0.01 versus0; sign remains positive.
Fork retains five critics/actor period1/clipping0/initial alpha0.2. Aligned actor
updates262500 versus fork525000; interaction/update-call budgets match, not FLOPs.

Actual unmodified Torch trainer supplies a small FP32 fixed-weight/batch/noise
trace. GPU smoke repeats parameter/Adam-moment parity on pinned JAX0.4.38 and
trains/restores both aligned arms before releasing nine single-seed trainers.
Three CPU strict audits and aggregate wait for compact models/audits. Scheduler
only, remote GPUs, no local/Windows training. Sync actor bundles and compact
JSON only; the0.32MB parity input is not a training replay/checkpoint. Resume
only this treatment's own full checkpoint. BAPR/old implementations are unchanged.

Report paired switching differences/95% descriptive t intervals and stationary
results separately. Aligned-full versus no-shift isolates norm treatment;
aligned-full versus fork is a bundled implementation-profile contrast.
Limitations: three development seeds, reused windows, FP32 rather than original
GPU AMP, and a different environment/sampling budget from the original paper.
This is an implementation diagnostic, not original-paper reproduction or BAPR confirmation.
