# BAPR-v87 Validation and Promotion Gate

## Audit

- Complete evaluation rows: **36/36**.
- All stationary results use the untouched validation task stream; the reserved stream remains unopened.
- Robust, oracle, and learned policies are evaluated from the same final checkpoint and deterministic RNG schedule.
- Promotion is decided per predeclared variant; metrics are not mixed across variants.

## Same-checkpoint ladder

| variant | env | robust OOD / switch | oracle gain OOD / switch | learned gain OOD / switch | learned corr / MAE | AUC / delay | adapt strength | Ant/test term robust->learned | safe target std |
|---|---|---|---|---|---|---|---:|---|---:|
| v87a_switch_deploy | Ant | 919.6 / 1463.6 | -18.7% / -19.8% | -15.2% / -25.5% | 0.964 / 0.259 | 0.666 / 0.0 | 0.874 | 0.758->0.850 | 0.000 |
| v87a_switch_deploy | HalfCheetah | 1990.0 / 1711.8 | -3.1% / -16.3% | -1.3% / -7.7% | 0.961 / 0.291 | 0.577 / 0.0 | 0.845 | n/a | 0.000 |
| v87a_switch_deploy | Hopper | 632.9 / 1669.6 | 18.8% / -21.5% | 18.2% / -2.1% | 0.931 / 0.344 | 0.477 / 6.0 | 0.941 | n/a | 0.000 |
| v87a_switch_deploy | Walker2d | 345.2 / 2371.7 | 37.5% / -43.0% | 56.4% / -51.5% | 0.819 / 0.528 | 0.444 / 18.0 | 0.914 | n/a | 0.000 |
| v87b_paired_safe | Ant | 667.1 / 1491.5 | 71.4% / 21.2% | 66.6% / 15.2% | 0.978 / 0.167 | 0.453 / 5.0 | 0.621 | 0.858->0.600 | 0.371 |
| v87b_paired_safe | HalfCheetah | 2653.7 / 2377.7 | 0.2% / 0.5% | -1.9% / -2.7% | 0.974 / 0.238 | 0.667 / 0.0 | 0.161 | n/a | 0.100 |
| v87b_paired_safe | Hopper | 303.1 / 1686.2 | 22.3% / 59.8% | -17.7% / 33.4% | 0.944 / 0.322 | 0.632 / 16.0 | 0.317 | n/a | 0.208 |
| v87b_paired_safe | Walker2d | 380.5 / 2497.0 | 4.1% / -88.8% | 22.4% / -53.9% | 0.827 / 0.489 | 0.419 / 16.0 | 0.405 | n/a | 0.132 |
| v87c_paired_strict | Ant | 940.7 / 1450.5 | 12.4% / 30.6% | 12.8% / 33.5% | 0.980 / 0.181 | 0.560 / 2.0 | 0.357 | 0.758->0.792 | 0.318 |
| v87c_paired_strict | HalfCheetah | 2434.0 / 2106.8 | -5.7% / -2.7% | -5.4% / -10.2% | 0.961 / 0.228 | 0.560 / 0.0 | 0.447 | n/a | 0.070 |
| v87c_paired_strict | Hopper | 504.0 / 4494.7 | -5.1% / -41.1% | -20.7% / -44.1% | 0.938 / 0.333 | 0.542 / 10.0 | 0.341 | n/a | 0.207 |
| v87c_paired_strict | Walker2d | 390.4 / 1830.1 | -46.0% / -68.4% | 25.3% / -39.8% | 0.769 / 0.553 | 0.574 / 20.5 | 0.248 | n/a | 0.203 |

## Fixed gate

| variant | oracle 3/4 | Ant+HC +10% | recover 70% | Hopper+Walker 95% | AUC/delay | safe targets | Ant risk | protocol | promote |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v87a_switch_deploy | FAIL (0/4) | FAIL | PASS (97.2%) | FAIL | FAIL | FAIL | FAIL | PASS | FAIL |
| v87b_paired_safe | FAIL (2/4) | FAIL | FAIL (-761.3%) | FAIL | FAIL | PASS | PASS | PASS | FAIL |
| v87c_paired_strict | FAIL (1/4) | FAIL | PASS (103.1%) | FAIL | FAIL | PASS | FAIL | PASS | FAIL |

## Verdict

**No V87 variant passed:** do not run five seeds. Freeze the algorithm-search result and pivot to the systematic analysis paper route, using bus and V87b Ant as validated positive cases, historical HalfCheetah only after protocol-matched revalidation, and robust-policy or termination failures as negative cases.
