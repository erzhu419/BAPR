# Executed-action inverse system-ID screen

## Question

The v1 forward likelihood and v2 affine evidence calibration both fail under
closed-loop deployment even though true-mode oracle control has large,
five-seed-confirmed headroom. Their mode evidence is policy- and
trajectory-dependent, and mode 0 is systematically aliased with incorrect
heads.

This v3 positive-control asks whether a more policy-invariant physical
quantity can identify the persistent actuator regime: infer the action
actually applied to physics from consecutive observations, then compare it
with each candidate polarity transform of the commanded action.

## Split and information boundary

- Model-fit controller seeds: `8,16`.
- Filter-selection controller seed: `24`.
- Untouched frozen-control seeds: `101,211,307,419,523`.
- Fit events: `92001,92002`.
- Selection events: `93001,93002`.
- Sealed test events: `94001,94002,94003`.
- Modes persist for 250 actions; per-step Gaussian execution noise remains
  `std=0.02`.

During exploratory model fitting only, the simulator's `executed_action` is a
supervised inverse-dynamics target. The online estimator and all sealed audits
receive only `(observation, commanded_action, next_observation)`. They do not
receive mode ID, gain vector selected by the environment, executed action,
future transitions, reward labels, or a switch clock.

The four candidate gain patterns are part of the synthetic benchmark
definition. At inference, the model first predicts executed action without
seeing commanded action. A mixture likelihood then compares this prediction
with each candidate transformed commanded action.

## Model and causal filter

The inverse model is a five-head independent MLP ensemble over
`(s_t, s_{t+1}, delta_s)`. Each head uses a separately bootstrapped minibatch.
Empirical inverse residual variance is frozen from the fit split. Ensemble
dispersion and residual variance remain separate epistemic and aleatoric
quantities.

A sticky HMM accumulates candidate action likelihood causally. Hazard,
evidence scale, and posterior decay are selected on seed 24 only. The action
at time `t` uses the posterior obtained from transitions strictly before
`t`.

## Frozen gates

The same gates used for v1 and v2 apply:

- mode accuracy at least `0.85`;
- median and P90 switch delay at most `25` and `50` actions;
- Brier score at most `0.25`;
- at least 50% oracle-headroom recovery;
- at least 4/5 policy-seed wins over robust;
- termination gap no more than five percentage points.

Both inference and frozen-control gates must pass before any
posterior-conditioned SAC training. Failure rejects threshold tuning and
requires either a multi-transition inverse model with explicit excitation or
an environment whose latent regime is more directly observable.

This remains a supervised simulator positive control, not a final generic
BAPR estimator.

## Scheduler graph

The scheduler-only graph was submitted on 2026-07-28:

- `t58371`: inverse-model training; allowed GPU nodes are only
  `local`, `jtl110gpu`, and `node007`, with a measured-small 2.5 GB VRAM
  claim;
- `t58372-t58376`: five independent CPU audits, file-gated on the complete
  model manifest and parameter archive;
- `t58377`: CPU aggregation, file-gated on all five immutable audit
  manifests.

`jtl110gpu2` is excluded after its recent disconnect and `jtl311linux` remains
excluded from BAPR work. The graph uses neither Slurm nor auto-adopt.

## Final result

All seven tasks completed. The model, five audit manifests, and every sealed
event file pass hash and schema validation. The remote aggregate output was
not synchronized by the scheduler, so the same pure-JSON analyzer was rerun
locally after validation.

The inverse model's RMSE is `0.248` on its fit trajectories and `0.478` on
the seed-24 validation trajectories. Despite that controller-seed shift, the
structured candidate-action likelihood is sharply discriminative. The
validation split passes with `0.9996` switching mode accuracy and selects
`hazard=0.002`, `evidence_scale=1.0`, and `posterior_decay=0.98`.

The untouched five-seed result passes every preregistered gate:

| Arm | Switching return | Delta vs robust | Seed wins | Oracle headroom recovery |
|---|---:|---:|---:|---:|
| robust | 1023.4 | - | - | - |
| true oracle | 2327.7 | +1304.3 | 5/5 | 100% |
| inverse soft | 2190.5 | +1167.0 | 5/5 | 89.3% |
| inverse MAP | 2135.1 | +1111.6 | 5/5 | 84.5% |

For the soft estimator, the paired switching improvement is `+1167.0` with
95% CI `[+688.3,+1645.7]`. Every policy seed wins; per-seed oracle-headroom
recovery ranges from `82.3%` to `93.6%`. Mode accuracy is `0.9992`, Brier
score `0.0013`, median delay `5`, and P90 delay `11`. Termination remains
zero. Stationary mean is `2303.5`, effectively matching oracle `2320.3`, and
its mean worst-mode return (`2005.7`) is also oracle-level (`2003.0`).

This resolves the mechanism question. The benchmark has adaptation headroom,
the conditioned policy can use it, and causal inference can recover it. The
failed v1/v2 results came from an unconstrained forward-likelihood
representation that encoded policy/trajectory distribution. Constraining
mode evidence through a shared physical quantity, executed actuator command,
removes that aliasing.

Direct posterior-conditioned training is now authorized by the frozen gates,
but the current model is still a simulator-supervised positive control. The
next frozen ablation removes `executed_action` labels: its target is only the
known training-mode transform of commanded action, i.e.
`clip(gain(mode) * commanded_action)`. Per-step execution noise remains
unobserved. Passing that ablation would show that the result needs privileged
training mode, which is already standard in the oracle policy curriculum, but
not privileged realized simulator noise.
