# Frozen Publication Extensions

Frozen before submission on 2026-10-05. The existing manuscript algorithm,
negative results and cohort-specific acceptance decisions are unchanged.
No checkpoint selection, filter retuning or replacement of failed seeds.

## Independent Estimator Seeds

Reacher crosses controller seeds 99007, 99027, 99047, 99067, 99087 with
independently trained inverse ensembles 95103, 95121, 95139. These existing
ensembles start from different random initializations, share the prescribed
178,000 training/82,000 validation transitions from a separate provider,
and each receive 4,500 AdamW updates. Reuse the frozen reference/robust
controllers and the primary-model conditioned SAC. Train conditioned SAC
from scratch for each of the other two models and each controller seed:
10 additional training jobs, each 400 iterations, 1.6M interactions and
100,000 updates. Each comparator uses its own matched estimator in both
training and evaluation. All estimator-training and provider costs are
reported separately from controller budgets.

Primary endpoint: paired switching return of causal compensation minus
matched conditioned SAC, reported separately for every estimator seed.
Also report robust, native reference, uncompensated and true-mode controls.
Inference uses five controller-seed means, not 15 estimator/controller
pairs or individual episodes as independent replicates.

HalfCheetah additionally evaluates existing independently trained models
94103, 94121, 94139 on nine frozen policies 88021 through 88179 (the explicit
roster is in the protocol module). Exclude provider 88003. This is a
retrospective calibration sensitivity study: the models use the same
architecture but a different calibration recipe from the manuscript's
two-stage estimator. It is not another nine-seed confirmation or evidence
that the original recipe has been replicated with three model seeds.
Historical model-training runtime was not recorded in those model manifests;
only this audit's evaluation runtime is pinned. Export the existing weights
to portable NPZ arrays without changing or refitting them.

## Non-Diagonal Actuator Coordinates

Reacher uses the four fixed matrices I, [[0,1],[-1,0]], -I and
[[0,-1],[1,0]]. Two modes permute channels, unlike diagonal sign flips.
These known signed-permutation transforms preserve the action box; their
inverse is the transpose. Physical dynamics, reward, noise SD 0.02,
200-step episodes and 50-step persistent modes remain unchanged.

Use new controller seeds 117003, 117023, 117043, 117063, 117083 and new
estimator seeds 118003, 118023, 118043. Provider 116003 is separate. Train
reference, robust and conditioned SAC independently for each controller
seed at 1.6M interactions/100,000 updates. The primary estimator is 118003,
fixed in advance. Each inverse ensemble receives the same 260,000
calibration transitions and 4,500 updates. Recreate those fixed data locally
on each learner node; do not transfer dataset arrays to the local machine.
Count actual repeated data collection in compute accounting, without
pretending regenerated identical rows are independent new information.

Primary endpoint: causal compensation minus information-matched conditioned
SAC using model 118003. Secondary comparisons include robust and all three
model seeds. True-mode compensation must reproduce paired native trajectories;
report every fixed-context control and every failed/nonpositive result.

## Original Torch RE-SAC Comparator

Use the unmodified bundled RESACTrainer, VectorizedQNetwork and rlkit policy,
directly compared with the current local RE-SAC source. Preserve ten critics,
LayerNorm, beta=-2, beta_ood=0.01, beta_bc=0.001, weight_reg=+0.01,
critic/actor ratio 2, gradient clip 1 and automatic entropy tuning.
The no-shift arm changes only weight_reg to zero. It remains an ensemble
algorithm, not SAC. Rewards are unscaled.

Five controller seeds 109003, 109023, 109043, 109063, 109083 are paired with
the existing runtime-matched BAPR reference actors. Each new source arm
receives 8.4M interactions INCLUDING its 10,000 warmup steps, 525,000 critic
calls and 262,500 actor opportunities. Match interaction/critic budgets;
retain and report the original actor period rather than calling actor budgets
equal. Record actual optimizer steps including AMP-skipped updates.

The adapter uses the same HalfCheetah spring physics, persistent modes, noise
and deterministic strict-horizon evaluation as BAPR, with CPU sampling and
Torch CUDA updates. It retains the shared experiment's block collection
convention rather than its original Gym/vector collector. This is a faithful
source-trainer comparator ON THE COMMON BENCHMARK, not a reproduction of
the original Gym-paper protocol. BAPR's frozen estimator training costs and
the different training compute profile remain explicit comparison limits.
Primary endpoint: paired BAPR minus full source RE-SAC switching return.
No adoption decision is inferred from a smoke test.

## Evaluation And Execution

All three studies use paired new event streams 211601, 211617, 211633,
five switching episodes per stream and five stationary episodes per mode
(stationary stream 211701). Full horizons, three observed switches,
finite trajectories and native/oracle equivalence are checked.
Report seed-level paired differences, 95% t intervals and wins, plus mode
accuracy/delay and switch-window returns where applicable. Do not pool
historical cohorts or tune after viewing these outcomes.

Scheduler only: 72 jobs (44 GPU including setup/smoke, 24 CPU audits and
four CPU aggregations). Training is split by seed, uses remote GPUs only,
and waits for its smoke/model artifacts. CPU evaluation waits for complete
compact policy bundles, not historical task IDs. Existing optimizers,
replay, RNG and iteration counters are restored on retry. Synchronize
compact model parameters and JSON only; full training checkpoints remain
on the server except scheduler-controlled recovery/migration.

Execution amendment before source training: GPU2 became unreachable during
runtime installation. Source training/smoke use jtl110gpu only and require
only its ready marker. The optional GPU2 setup does not block those jobs;
it may be admitted after its runtime is ready. Scientific controls, seeds,
budgets and evaluation remain unchanged.

Runtime repair: GPU1's first installation exhausted its disk, leaving Torch
metadata but missing package files. Disposable pip wheel caches were cleared
and torch==2.4.1 was reinstalled without changing dependencies. Setup t135742
and source smoke t135721 passed, including CUDA updates, actor export and
exact checkpoint next-update recovery. The setup script now repairs this
partial-installation case; the registered experimental protocol is unchanged.
