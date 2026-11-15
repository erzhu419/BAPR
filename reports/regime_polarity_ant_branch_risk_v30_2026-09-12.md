# V30 Ant finite-horizon paired branch-risk result

V30 follows `markdown/GPT_diagnosis.md` by replacing the old discounted-risk
interpretation with paired finite-horizon simulator-state branches. It reuses
the frozen V29 reference choices, performs no training, and compares complete
true-mode-compensated continuation with complete robust-SAC continuation under
common future actuator noise.

| Horizon | Candidate term. | Fallback term. | Risk reduction | Rescue | Harm |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0% | 0.0% | +0.0% | 0.0% | 0.0% |
| 8 | 0.0% | 0.0% | +0.0% | 0.0% | 0.0% |
| 16 | 0.0% | 2.1% | -2.1% | 0.0% | 2.1% |
| 32 | 0.0% | 5.4% | -5.4% | 0.0% | 5.4% |
| 64 | 0.0% | 8.6% | -8.6% | 0.0% | 8.6% |
| 128 | 0.0% | 9.2% | -9.2% | 0.0% | 9.2% |
| 250 | 0.3% | 9.9% | -9.5% | 100.0% | 9.9% |

| Seed | Reference | Snapshots | Unique candidate failures | Candidate | Fallback | Difference |
|---:|---:|---:|---:|---:|---:|---:|
| 85003 | 3 | 24 | 2/192 | 1.0% | 7.3% | -6.2 pp |
| 85021 | 3 | 24 | 0/192 | 0.0% | 16.5% | -16.5 pp |
| 85039 | 2 | 24 | 0/192 | 0.0% | 5.7% | -5.7 pp |

The candidate branch is exactly invariant across all four actuator modes:
maximum cumulative-return error is zero and all 1,728 compared termination
traces agree. Every one of the nine source trajectories survived its full
1,000-step horizon, so the preregistered pretermination sampler found no source
failure states. Only two of 576 unique candidate continuations terminated.
Those two were rescued by robust continuation, but there are too few positives
to fit or validate a risk model.

The stronger negative result is that robust continuation is not a safe default
from specialist-visited states. At 250 steps it is worse for all three policy
seeds and all four actuator modes: mode-specific termination is
9.4%/13.0%/10.8%/6.3% versus 0.35% for the invariant candidate. It harms about
9.9% of candidate-surviving paired branches and loses about 582.5 return on
average. Frozen robust parameters therefore do not provide a robust state-
distribution fallback guarantee.

The registered risk-model authorization gate **fails**. The Ant fallback/shield
branch is closed on these development policies; V29 remains a return-headroom
mechanism result and reliability counterexample. This result does not weaken
the positive V28 HalfCheetah causal-compensation result, and it does not justify
reusing the two rare failures to tune a classifier.

Scheduler audits `t93066-t93068` and aggregate `t93069` completed on Linux CPU
nodes. Only compact JSON artifacts were synchronized; no simulator states,
trajectories, replay buffers, or checkpoints were saved or pulled.
