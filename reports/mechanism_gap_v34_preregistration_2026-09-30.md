# Action-compensation mechanism: frozen follow-up experiments

The V32 controller, estimator, evidence and paper results remain unchanged.
This follow-up addresses identification necessity, information fairness and
independent applicability. It does not continue simulator formalization.

## Identification ablations

Use the nine frozen V32 reference policies with seeds 88021, 88039, 88057,
88079, 88103, 88121, 88139, 88157 and 88179. Compare the frozen five-head
inverse model with its instantaneous evidence (no temporal filter) and frozen
head 0 alone. Head 0 is fixed in advance, not selected for performance. Oracle
and uncompensated actions anchor these comparisons. No actor is retrained.

For the direct-classifier comparison, generate a new shared dataset using
reference policy seed 88003, excluded from evaluation. Use 178,000 training
and 82,000 validation transitions, distinct event seeds 203001/203101, cyclic
250-step modes, noise std 0.02, and reset every 1000 steps. Commands mix
oracle-compensated reference actions plus std-0.15 exploration (80%) and
uniform random commands (20%). The inverse target is the expected actuator
signal G_z a, not the realized noisy action. Mode labels are offline training
labels and are unavailable to inference.

Fit a five-head inverse model and a two-hidden-layer direct classifier from
scratch, using identical rows/minibatch indices, 4500 updates/model, batch 256,
AdamW lr 1e-4, weight decay 1e-5 and gradient norm cap 10. Report parameter
counts; this is a practical simple-classifier control, not equal capacity.
Estimator seeds are 94103/94121/94139. Retain the frozen filter settings
(hazard 0.002, evidence scale 1, posterior decay 0.98); no test tuning.
Inverse variance is calibrated on validation residuals with bounds [1e-4,0.25].
The frozen-policy ablations are post-V32 diagnostics, not an independent
confirmation of V32. The original V5 trajectories were not retained. Therefore the fresh paired
models compare representations on new identical data, not exact retraining
of V5. The original V5 remains a separate frozen anchor.

## Information-matched conditioned SAC

Fresh paired policy seeds: 92003,92021,92039,92057,92079. Both arms receive
8.4M controller interactions and 525,000 actor/critic updates, using the common
SAC, two critics, width 256, batch 256, lr 3e-4, final policy only.

The reference arm retains the V32 recipe: 5.6M multimode interactions followed
by 2.8M fixed-mode-0 fine-tuning; preserve networks/targets/alpha but reset
replay and optimizers at the fork. The comparison uses the same new compiled
collector for both arms; it is not pooled into the original V32 confirmation.

The conditioned arm receives (s_t,b_t), where b_t comes from the exact frozen
V5 inverse model and sticky filter. It trains directly on causal inferred
beliefs, never on privileged task IDs or oracle one-hot contexts. Replay stores
(s_t,b_t) and (s_{t+1},b_{t+1}) from the rollout. Actor action precedes the
completed-transition posterior update. Both inference paths reset belief to
uniform at episode reset. No extra residual, gate, BOCD or regularizer.

The fixed pretrained estimator is shared by both arms. Its supervised
training/provider-trajectory cost is additional to each controller budget,
not silently counted as free. Report actor parameters and measured inference
rollout runtime, with first-call compilation separated from subsequent calls.

## Independent applicability screen

Use Brax's native Reacher reset, dynamics, observations and reward, generalized
backend. Do not copy the locomotion reward implementation or change morphology.
Four persistent actuator maps are (+,+),(-,+),(+,-),(-,-); normalized execution
noise std 0.02 is sampled each step. Mode dwell is 50 steps and evaluation
horizon 200, with simulator state/target preserved at each mode switch.
This is a distinct reaching task, but still the same known reversible sign-map
mechanism; it is not evidence for arbitrary parameter or bus adaptation.

Seeds 93003/93021/93039, robust and fixed-reference SAC arms, each 1.6M
interactions and 100,000 updates. Same architecture and optimizer as above.
Compare robust, dynamic oracle, uncompensated, all four fixed compensations,
and native reference under identical initial states/noise streams.
Require oracle/native trajectory equivalence. Before training an estimator,
require every seed to favor oracle over robust, a mean gain exceeding 10% of
abs(mean robust return), a positive mean over each fixed compensation, and
all episodes to contain three valid switches. Failure remains a negative
applicability result; do not tune this environment after accessing outcomes.

## Evaluation and execution

All groups use new event seeds 202601/202617/202633, five paired episodes/event,
rotating mode order (0,1,3,2). The policy seed is the inference unit; report
paired mean, 95% t interval and wins, without turning win counts into a new
universal publication gate. Record switch delays including undetected/censored
switches, full-horizon return, Brier score, mode accuracy, command-mismatch
energy and 1/2/4/8/16/32-step switch-local metrics. Retain all seeds.

Scheduler only. One seed per task, remote GPU training, remote CPU evaluation.
A GPU smoke gates all new training and checks finite updates, causal context
storage, native/oracle equivalence and production replay allocation. Initial
GPU claim is 1500MB, not an unmeasured 4/8GB request. Adjust only if the smoke
shows an actual resource need, not from performance outcomes. Checkpoints,
replay and raw calibration trajectories stay on servers. Sync compact actor
and estimator bundles only when needed by downstream evaluation; final analysis
uses small JSON audits. Aggregation waits on output files, not task IDs, so a
completed retry can satisfy the dependency.

Runtime instrumentation retains the frozen seeds, budgets and sampling path.
Training logs now report completed iterations with a total, cumulative time,
rollout/update time and checkpoint-save time. Simulator reset is executed only
at the existing terminal/timeout boundaries. Scheduler runtime matching
separates fitting, training, smoke and evaluation, including environment/arm;
it accepts the earlier cumulative `elapsed=` logs as well as `Time:` logs.

## Reacher CPU-sampling replacement

Authorized on 2026-09-30: replace all six Reacher runs together, from scratch,
only after admission; do not mix CPU-sampled policies with the old GPU group.
Keep seeds, physics, 4000 samples/250 updates, 400 iterations and evaluation
unchanged. Freeze JAX 0.4.38/Brax 0.10.5 for training and evaluation.
Admission uses independent diagnostic policy seeds 92101/92103/92107 in both
reference and robust modes: equal physics fields; same-state/action single-step
obs/reward/done within atol=rtol=1e-5; paired 95% mean episode-return difference
interval wholly inside +/-1% of abs(GPU mean), with denominator floor 1.
Twenty 200-step episodes per diagnostic case; all six cases must pass.
This is a new matched runtime protocol, not a retroactive bitwise-equivalence
claim or permission to continue an old checkpoint across sampling backends.
Admission `t107840` passed on 2026-09-30; all six diagnostic cases and the GPU
learner smoke succeeded. CPU evaluation pins Flax 0.10.4/MuJoCo 3.1.6 alongside
the frozen JAX/Brax pair, using its own Python 3.10 environment (SciPy 1.15.3).
The existing evaluation environments and old checkpoint directories are unchanged.
