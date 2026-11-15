# RE-SAC paper-fidelity JAX smoke

This is a two-policy-seed diagnostic. It aligns the continuous gravity generator and the 10k random plus 1000 x (1k rollout, 1k update) budget. It does not claim an exact reproduction of the original PyTorch ESCP architecture.

## HalfCheetah-v2

| Method | Stationary OOD | Switching |
|---|---:|---:|
| SAC | 2416.0 +/- 59.0 | 2227.4 +/- 53.2 |
| ESCP | nan +/- nan | nan +/- nan |
| RESAC | 750.4 +/- 46.2 | 781.6 +/- 324.2 |

Paper ordering recovered: **False**

- `escp_minus_sac`: stationary +nan, switching +nan.
- `resac_minus_sac`: stationary -1665.6, switching -1445.9.

## Ant-v2

| Method | Stationary OOD | Switching |
|---|---:|---:|
| SAC | 1928.1 +/- 475.8 | 2195.9 +/- 303.3 |
| ESCP | nan +/- nan | nan +/- nan |
| RESAC | -798.5 +/- 360.1 | -1167.9 +/- 84.2 |

Paper ordering recovered: **False**

- `escp_minus_sac`: stationary +nan, switching +nan.
- `resac_minus_sac`: stationary -2726.5, switching -3363.8.

Global smoke pass: **False**

A failed smoke blocks using the current JAX RE-SAC result as a scientific baseline. A pass authorizes a fresh five-seed run; these two seeds are not pooled into that confirmation.
