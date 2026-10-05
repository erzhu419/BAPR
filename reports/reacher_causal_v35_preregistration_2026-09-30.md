# Reacher causal-compensation follow-up

The admitted CPU-sampling Reacher oracle screen passed on three policy seeds.
Freeze its native generalized physics, 200-step horizon, 50-step persistent
sign maps, per-step noise std 0.02 and existing reference/robust policies.
This is an independent-task mechanism screen, not a new V32 confirmation.

Provider seed 94003 is excluded from evaluation. Train its fixed-reference
SAC with the same 1.6M interactions/100,000 updates as the screen. Collect
178,000 training and 82,000 validation transitions using event seeds
205001/205101, cyclic modes and native reset every 200 steps. Commands mix
oracle compensation plus std-0.15 exploration (80%) and uniform actions (20%).
Targets are expected pre-noise executed actions; the inverse network never
receives the command or mode. All three estimators share these exact saved rows.

Estimator seeds 95103/95121/95139; primary seed 95103, selected before outcomes.
Keep the existing five-head, three-layer, width-256 inverse architecture,
feature scales and validation variance bounds [1e-4,0.25]. Fit 4500 updates,
batch 256, AdamW lr 1e-4, weight decay 1e-5, gradient norm cap 10. Keep the
existing posterior controls (0.002,1,0.98); no test tuning or policy updates.

Evaluate policy seeds 93003/93021/93039 on new event seeds 205601/205617/205633,
five paired 200-step episodes/event. Compare all three causal estimators,
primary no-filter/fixed-head-0 ablations, robust, oracle/native, uncompensated
and four fixed compensations. Also evaluate four stationary modes with five
episodes/mode using event base 205701. The action uses the pre-transition
posterior; the completed transition updates only the next action's posterior.

Primary screen: positive paired 95% CI versus robust, positive mean versus
each fixed compensation, and three observed switches/full horizon in every
primary/oracle/robust episode. Report oracle-headroom recovery, all delays
including undetected switches, mode accuracy, Brier and switch-local metrics.
Policy seed is the inference unit; estimator repeats do not increase n=3.

JAX/Brax/Flax/MuJoCo remain 0.4.38/0.10.5/0.10.4/3.1.6. Scheduler only; GPU
learner jobs exclude local/node007. Collection and three lightweight fits
share jtl311linux to avoid transferring calibration arrays; provider training
can use any matching GPU server. CPU evaluation uses node001-node007.
Dataset/checkpoints stay remote; synchronize only compact models and JSON.
Report provider training and 260,000 calibration interactions as additional
estimator cost, separate from each frozen evaluation controller's budget.

Placement was changed from jtl110gpu2 before data generation: both 110 SSH
routes timed out, while the GPU smoke passed on the matching jtl311linux runtime.
Seeds, data generation, model fitting and evaluation settings are unchanged.
