# Runtime-Matched Regularized Comparison

Frozen before submission on 2026-10-01. No BAPR or estimator tuning.
HalfCheetah spring / actuator polarity retains the V32 environment, noise 0.02,
1000-step horizon, 250-step persistent blocks and its three stationary/three
switching event streams. These are reused protocol windows, not new holdouts.
New policy seeds: 109003, 109023, 109043, 109063, 109083; no historical pooling.

All jobs require JAX 0.4.38, Brax 0.10.5, Flax 0.10.4, MuJoCo 3.1.6 and
record versions, actual interpreter, host and completed budgets. GPU sampling
and GPU updates retain the original HalfCheetah training backend; CPU audits
use the same versions. Local training and node007's different GPU profile are
excluded. Checkpoint migration is allowed only within the admitted profiles.

Operator amendment on 2026-10-01: GPU-node whitelisting is removed, including
the node007 exclusion above. Eligible nodes must retain the pinned software
versions and normal resource admission; actual host/GPU metadata remain recorded.
Training recipes, seeds, interaction/update budgets and evaluation are unchanged.

Five final controller arms: unchanged BAPR canonical recipe, SAC, recurrent
ESCP, the historical RE-SAC-0 recipe, and regularized RE-SAC. Each final path
receives 8.4M interactions and 525000 updates. BAPR consists of a 5.6M/350000
robust source followed by a 2.8M/175000 fixed-mode-0 fork, preserving actor,
critics, target critics and alpha while resetting replay/optimizers. Final
checkpoints only; no validation-based actor or checkpoint selection.

Regularized RE-SAC: five critics, width256, learning rate0.0003, fixed LCB
beta=-2, weight_reg=0.01, beta_ood=0.01, behavior cloning0.001, independent
targets, no anchor, adaptive beta or EMA. The positive target/actor weight-norm
shift is retained. RE-SAC-0 separately retains its historical 0.75 target blend,
beta annealing -2 to0, anchor0.001 and EMA evaluation; it is not the full method.
Thus the full/zero comparison changes the full recipe, not only two coefficients.

Primary: BAPR minus regularized RE-SAC switching return, first averaged over
events within each policy seed; two-sided paired95% t interval on five seeds.
SAC, ESCP and RE-SAC-0 contrasts are secondary marginal intervals, with seed
wins reported descriptively. No replacement of the old V32 composite gate.
Stationary, uncompensated, true-mode and robust-source results remain separate.
The historical frozen estimator and its extra data/supervision costs are shared
across BAPR seeds, not charged to memoryless baselines or refitted here.

Scheduler DAG: one GPU smoke,30 per-seed GPU training jobs, five CPU audits,
one CPU aggregate. Reference forks wait for their own source; audits wait for
all six trained artifacts; aggregate waits for five complete audits. Only
evaluation-required compact parameters/metadata and JSON are synchronized;
full checkpoints, optimizer/replay arrays and training logs remain remote.
