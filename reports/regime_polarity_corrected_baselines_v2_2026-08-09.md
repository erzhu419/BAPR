# Polarity BAPR corrected-baseline comparison

This replaces the state-only ESCP approximation and the invalid legacy RE-SAC arm. BAPR and SAC are immutable SHA-256-locked audit results; recurrent ESCP and released-B0 RE-SAC are newly trained.

| method | switching | stationary | switching termination |
|---|---:|---:|---:|
| bapr | 2015.3 +/- 32.1 | 2118.7 +/- 14.5 | 0.000 |
| sac | 1222.5 +/- 772.4 | 1257.6 +/- 766.4 | 0.000 |
| escp_recurrent | 1633.8 +/- 407.4 | 1657.4 +/- 538.2 | 0.000 |
| resac_b0 | 1568.4 +/- 316.5 | 1640.0 +/- 344.5 | 0.000 |

## Mode diagnostics

| method | mode 0 | mode 1 | mode 2 | mode 3 |
|---|---:|---:|---:|---:|
| bapr | 2027.8 | 2099.0 | 2017.3 | 2330.6 |
| sac | 1307.2 | 1279.9 | 1240.0 | 1203.1 |
| escp_recurrent | 1700.2 | 1678.3 | 1538.9 | 1712.2 |
| resac_b0 | 1559.8 | 1766.7 | 1436.8 | 1796.6 |

## Registered decision

- Recurrent ESCP minus SAC switching: +411.2 (3/5 seed-slot wins)
- Released-B0 RE-SAC minus SAC switching: +345.9 (3/5 seed-slot wins)
- BAPR minus strongest corrected baseline: +93.5
- Conservative 95% interval: [-298.0, +485.0]
- Registered seed-slot wins: 4/5
- Stationary retention: 107.1%
- Primary pass: `False`

deployment-performance comparison on the frozen HalfCheetah actuator-polarity protocol; BAPR retains its frozen multi-controller training history, while all single-controller baselines use the same 5.6M-step/350k-update budget
