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
