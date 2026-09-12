# Independent source-controller headroom

This two-seed development screen trains robust SAC, ESCP, and four fully independent fixed-mode SAC specialists. The privileged dynamic oracle selects the matching specialist from the true physical mode. No learned estimator is trained in this stage.

## Seed 4021

| Arm | Stationary mean | Worst mode | Switching |
|---|---:|---:|---:|
| `robust_sac` | 1717.6 | 1377.5 | 1634.8 |
| `escp` | 1613.8 | 1093.2 | 1309.0 |
| `specialist_0` | 327.7 | -565.1 | 1129.0 |
| `specialist_1` | 484.4 | -398.1 | 560.5 |
| `specialist_2` | 16.3 | -502.7 | -511.8 |
| `specialist_3` | 591.4 | -489.2 | -396.3 |
| `dynamic_oracle` | 2622.4 | 1476.5 | 2762.6 |

- Dynamic gains over strongest comparator: stationary +52.7%, worst mode +7.2%, switching +69.0%.
- Diagonal specialist optima: 4/4.
- Seed gate: **False**.

## Seed 4049

| Arm | Stationary mean | Worst mode | Switching |
|---|---:|---:|---:|
| `robust_sac` | 1644.1 | 1543.2 | 1640.8 |
| `escp` | 1884.7 | 1621.2 | 1812.0 |
| `specialist_0` | 697.9 | -474.8 | 1268.8 |
| `specialist_1` | 287.3 | -598.4 | 422.9 |
| `specialist_2` | 501.6 | -433.7 | -209.8 |
| `specialist_3` | 202.1 | -586.9 | -437.3 |
| `dynamic_oracle` | 2812.6 | 2269.7 | 3103.9 |

- Dynamic gains over strongest comparator: stationary +49.2%, worst mode +40.0%, switching +71.3%.
- Diagonal specialist optima: 4/4.
- Seed gate: **True**.

Headroom gate pass: **False**

Decision: `stop estimator training; source-controller upper bound failed`.

Only a pass on both fresh development seeds authorizes another BAPR controller/estimator training stage.
