# RE-SAC / ESCP numerical-semantics audit

This protocol replaces the invalid v1 `paper-fidelity` label. It is a two-seed implementation diagnostic at the sealed v1 SAC budget, not an 8M-step confirmatory reproduction.

## Provenance correction

- The earlier v1 smoke used positive weight_reg=0.01 and pure independent targets, so it was not the completed MuJoCo B0 artifact.
- The released MuJoCo B0 launcher disables weight_reg and beta_ood and uses an independent/min blend, EMA evaluation, and anchoring.
- The positive bus regularization sign remains unchanged; this MuJoCo protocol sets its coefficient to zero.
- The corrected ESCP arm is still a state-only JAX approximation, not the original recurrent (state,last_action) environment probe.

## Legacy ESCP failure

- HalfCheetah: first non-finite update at iteration 13, global update 13346 (scan index 346).
- Ant: first non-finite update at iteration 0, global update 881 (scan index 881).

## HalfCheetah-v2

| Method | Stationary OOD | Switching |
|---|---:|---:|
| SAC | 2416.0 +/- 59.0 | 2227.4 +/- 53.2 |
| ESCP | 2102.7 +/- 104.0 | 1732.5 +/- 140.4 |
| RESAC | 2175.4 +/- 114.0 | 2126.3 +/- 154.4 |

Paired differences against the sealed SAC reference:
- `escp_minus_sac`: stationary -313.2 (0/2 wins), switching -495.0 (0/2 wins).
- `resac_minus_sac`: stationary -240.5 (0/2 wins), switching -101.1 (0/2 wins).

Numerical semantics pass: **True**

## Ant-v2

| Method | Stationary OOD | Switching |
|---|---:|---:|
| SAC | 1928.1 +/- 475.8 | 2195.9 +/- 303.3 |
| ESCP | 2333.2 +/- 530.8 | 2791.8 +/- 270.3 |
| RESAC | 2334.4 +/- 349.6 | 2309.5 +/- 769.8 |

Paired differences against the sealed SAC reference:
- `escp_minus_sac`: stationary +405.2 (1/2 wins), switching +595.8 (2/2 wins).
- `resac_minus_sac`: stationary +406.4 (1/2 wins), switching +113.5 (1/2 wins).

Numerical semantics pass: **True**

## Decision boundary

Global numerical pass: **True**.

A numerical pass only makes the corrected rows interpretable. It does not establish that ESCP or RE-SAC beats SAC, and it does not make the state-only ESCP approximation architecture-faithful. A fresh 8M-step, five-seed run is warranted only after this diagnostic is finite and its Q/actor scales are plausible.

## Fresh released-B0 confirmation

The registered graph `t72958-t72998` is complete: 20/20 independent 8M-step
training bundles, 20/20 strict CPU audits, and the final aggregate validate.
Every controller completed 2,000 iterations, 8M environment steps, and 499,500
updates. Evaluation uses three independently generated held-out streams per
training seed, deterministic-mean actions, strict 1,000-step stationary
horizons, and 500-step switching sequences.

| Environment | Method | Stationary OOD | Worst quartile | Switching |
|---|---|---:|---:|---:|
| HalfCheetah-v2 | SAC | 1216.8 +/- 620.8 | -27.3 +/- 223.7 | 225.0 +/- 134.5 |
| HalfCheetah-v2 | RE-SAC | 925.4 +/- 330.3 | 250.7 +/- 418.8 | 348.5 +/- 323.9 |
| Ant-v2 | SAC | 656.7 +/- 48.8 | 177.8 +/- 46.2 | 302.7 +/- 153.2 |
| Ant-v2 | RE-SAC | 722.1 +/- 30.3 | 278.6 +/- 79.5 | 511.1 +/- 195.5 |

On HalfCheetah, paired RE-SAC minus SAC is `-291.4 +/- 519.1` for stationary
mean (1/5 wins), `+278.1 +/- 295.5` for worst quartile (4/5 wins), and
`+123.6 +/- 312.8` for switching (3/5 wins). On Ant, the corresponding deltas
are `+65.4 +/- 78.0` (4/5 wins), `+100.8 +/- 81.3` (5/5 wins), and
`+208.4 +/- 78.0` (5/5 wins).

The completion exposed a checkpoint-compatibility defect rather than a CPU
scheduler defect. A `jtl110gpu2` outage interrupted four RE-SAC producers;
their retries loaded `train_state.pkl` but failed on the first actor update
because the saved Flax `anchor_params` metadata did not match the current
policy pytree. The loader now remaps compatible anchor leaves onto the current
treedef. Replacement tasks `t74441-t74444` resumed from the retained iterations,
completed the exact budget, published their bundles, and automatically released
the four dependent audits and final aggregate.

The scientific boundary is narrower than a blanket RE-SAC superiority claim.
RE-SAC clearly improves Ant tail robustness and switching and improves
HalfCheetah tail robustness, but it lowers HalfCheetah stationary held-out mean
and the five-seed uncertainty is substantial. The released legacy stream remains
a provenance cross-check, not an interchangeable evaluation result. ESCP stays
excluded from this confirmation until its recurrent history encoder is
reproduced.
