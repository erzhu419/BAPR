# Native Reward-Transfer Calibration

V43's positive-weight MuJoCo port fails even in native stationary Gym, while its
no-shift counterpart learns. Bus core `sac_ensemble_original_logging.py:update`
always centers each sampled reward batch and scales its sample std to10; the port
does not. Test only this omission, not a bundled reconstruction of the bus recipe.

Fresh development seeds114003/114023/114043, four arms: raw_full, raw_no_shift,
bus_reward_full, bus_reward_no_shift. Bus reward is10*(r-mean(r))/std(r), with
Torch's unbiased sample std; std below1e-6 is replaced by1, exactly as bus core.
Apply inside each critic call to the sampled training batch only. Replay rewards,
rollout physics and reported native Gym evaluation returns remain raw. Positive
weight_reg0.01 versus0; all other V43 source YAML controls, width256, alpha update,
CUDA AMP, collectors, runtime, interleaving and budgets stay unchanged.
Each independent task starts fresh, with1.01M interactions/1M critic calls/500k
actor opportunities. Final outcome is mean of eight1000-step deterministic native
episodes. No best-checkpoint selection; three policy seeds, not24 independent runs.

Primary diagnostic: paired reward-by-shift interaction plus absolute reward gain
for both weights. Also report positive-minus-zero under each reward treatment,
per-seed returns, descriptive95% paired t intervals, Q/norm/disagreement/alpha,
raw/transformed reward mean/std and actual AMP optimizer steps. A smaller gap
caused only by degrading no-shift is not restoration of positive-weight control.

Reuse the frozen V43 source and tested sampler/checkpoint implementation. Four-arm
GPU smoke verifies same-arm next-transition/four-update continuation and timing;
production admission requires estimated isolated training under20h. New protocol,
arm/seed and reward treatment identify checkpoints; do not resume V43 checkpoints.
Scheduler only, remote110 GPU nodes, measured-memory-based600MB initial VRAM,
four CPU threads per trainer; CPU aggregate waits for all12 complete JSON files.
Synchronize JSON only, with full replay and checkpoint retained remotely.
Limitations: development calibration in stationary Gym, not a bus reproduction,
not a BAPR confirmation; alpha/architecture/precision differences remain frozen.
