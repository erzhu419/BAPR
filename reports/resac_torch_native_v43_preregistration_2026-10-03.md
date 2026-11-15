# Original Torch / Native Gym Calibration

V42's higher update density improves no-shift returns but does not restore the
positive-weight recipe. The source experiment runs stationary Gym HalfCheetah-v2,
not gravity switching or actuator polarity. This is a calibration, not a BAPR result.

Freeze copied, unmodified RE-SAC Torch trainer/networks and DSAC rlkit collectors,
replay buffer and Gym factory. Native MuJoCo2.1, Gym0.10.9, Torch2.4.1/cu121,
NumPy1.24.4; source CUDA AMP remains enabled. No physics/mode/noise changes.
Fresh seeds113003/113023/113043; full weight_reg0.01 and no_shift0, all other
source YAML values unchanged. Four exploration envs,10000 random warmup steps,
1000 epochs x1000 interactions/1000 critic calls:1.01M steps,1M critic calls,
500k actor opportunities. Record actual optimizer steps, including AMP skips.
Collect one four-env step then perform four updates, including source prefetch.
Source periodic deterministic evaluation requests5 paths and actually returns8
at1000 steps. Primary outcome is final native-evaluation mean, paired full-minus-zero
across three seeds with descriptive95% t interval; report stationary physics,
Q/bonus/alpha/loss/time samples and actual optimizer counts. No checkpoint selection.

Operational deviations: use existing synchronous VectorEnv for evaluation instead
of its subprocess transport, with independent per-env RNG and unchanged collectors;
seed the legacy Gym spaces generator because source Box.seed is a no-op. Training
eval is native source-loop evaluation, not a separate GPU evaluation batch.
GPU smoke checks both arms and next-transition/four-update checkpoint continuation;
submit production only if projected native training is under20h per isolated task.
Checkpoint models, Adam/alpha/scaler, replay/prefetch, collector/physics/RNG every25
epochs and after warmup. Resume only the same arm/seed. Scheduler-only, remote110
GPU nodes; CPU-only aggregation. Sync compact JSON only, keep models/replay remote.

Limitations: three development seeds, stationary source protocol and source-loop
final windows, not a nonstationary paper reproduction or comparison with V32.
Native success would motivate a matched-world bridge; failure would localize the
issue to this original trainer/config/runtime before another JAX transfer change.
