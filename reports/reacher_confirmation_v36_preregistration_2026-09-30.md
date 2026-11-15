# Frozen Reacher five-seed confirmation

Freeze V35 primary estimator 95103, its inverse weights/variance, and filter
controls (0.002,1,0.98). No estimator retraining or selection after outcomes.
Fresh policy seeds: 96003/96021/96039/96057/96079; exclude all V34/V35 policy
seeds and provider 94003. Train reference and robust SAC independently per seed:
400 iterations, 4000 interactions/250 updates per iteration (1.6M/100,000).
Keep the admitted CPU sampling/GPU updates and final-only checkpoints.

Unchanged native generalized Reacher physics, 200-step horizon, persistent
50-step sign maps and per-step actuator noise std 0.02. Runtime remains
JAX/Brax/Flax/MuJoCo 0.4.38/0.10.5/0.10.4/3.1.6. No physics/backend changes.

Switching events 206601/206617/206633, five paired episodes/event; stationary
event base 206701, five episodes/mode. Every route shares the same per-episode
reset/noise keys. Test primary causal, oracle/native, robust, fixed 0-3,
uncompensated, no-filter and head0. Actions use pre-transition beliefs.

Keep V35's decision unchanged: positive paired 95% CI versus robust, positive
mean versus every fixed compensation, and complete horizons/three observed
switches for causal/oracle/robust. Inference unit is the five fresh policy seeds,
never episodes or estimator repeats. Report all seed differences, stationary
returns, headroom recovery, mode accuracy/Brier, missed switches and censored
delays, switch-local rewards, and no-filter/head0 comparisons regardless of sign.

Controller budgets match, but total historical training cost does not: reuse
the disjoint provider's 1.6M interactions/100,000 updates, 260,000 calibration
interactions and 4500 estimator updates as explicit shared additional cost.
Do not pool exploratory scores into confirmation or drop weak seeds.

Scheduler only: 10 GPU training jobs, one GPU smoke, five CPU audits and one
CPU aggregate. Matching GPU nodes: jtl110gpu/jtl110gpu2/jtl311linux; no local.
GPU jobs initially request 1500MB, not 4/8GB. Full resume checkpoints remain
remote; synchronize compact actor exports, required inverse NPZ and JSON only.
CPU jobs use node001-node007 and await their own two completed actor exports.
