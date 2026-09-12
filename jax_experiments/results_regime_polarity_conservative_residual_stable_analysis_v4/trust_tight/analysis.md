# Anchored residual development result

Decision: **FAIL** for promotion to an untouched five-seed confirmation.

| arm | mean delta | relative gain | wins | max term gap |
|---|---:|---:|---:|---:|
| anchored_base | -12.1 | 5.6% | 1/3 | 0.000 |
| oracle_residual | -5.6 | 6.5% | 1/3 | 0.000 |
| oracle_safe | 14.4 | 7.7% | 1/3 | 0.000 |
| learned_raw | -4.6 | 6.3% | 1/3 | 0.000 |
| learned_safe | 25.9 | 9.2% | 1/3 | 0.000 |

| seed | mode mask | robust | base | oracle | learned safe |
|---:|---|---:|---:|---:|---:|
| 1103 | 0000 | 1468.2 | 1410.7 | 1404.6 | 1410.7 |
| 1213 | 0100 | 600.5 | 772.3 | 788.1 | 813.5 |
| 1301 | 1000 | 1902.8 | 1752.3 | 1762.1 | 1824.9 |

Failed gates:
- anchored base did not preserve paired robust control
- zero-initialized oracle residual lacked stable headroom
- calibrated learned fallback lacked stable gain
- calibration did not enable a residual mode for every seed

This is a three-seed development screen. A pass permits a new untouched five-seed run; it is not itself a paper claim.

## Conservative training diagnostics

| seed | actor changed | accept rate | min LCB | mean shortfall |
|---:|---|---:|---:|---:|
| 1103 | True | 0.973 | -0.0120 | 0.0060 |
| 1213 | True | 1.000 | -0.0021 | 0.0034 |
| 1301 | True | 0.972 | -0.0099 | 0.0045 |
