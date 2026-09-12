# Anchored residual development result

Decision: **FAIL** for promotion to an untouched five-seed confirmation.

| arm | mean delta | relative gain | wins | max term gap |
|---|---:|---:|---:|---:|
| anchored_base | -12.1 | 5.6% | 1/3 | 0.000 |
| oracle_residual | 29.8 | 6.3% | 2/3 | 0.000 |
| oracle_safe | 22.9 | 7.4% | 1/3 | 0.000 |
| learned_raw | 38.3 | 6.7% | 2/3 | 0.000 |
| learned_safe | 9.2 | 6.7% | 1/3 | 0.000 |

| seed | mode mask | robust | base | oracle | learned safe |
|---:|---|---:|---:|---:|---:|
| 1103 | 0000 | 1468.2 | 1410.7 | 1376.0 | 1410.7 |
| 1213 | 0000 | 600.5 | 772.3 | 737.9 | 772.3 |
| 1301 | 1010 | 1902.8 | 1752.3 | 1947.2 | 1816.2 |

Failed gates:
- anchored base did not preserve paired robust control
- calibrated learned fallback lacked stable gain
- calibration did not enable a residual mode for every seed

This is a three-seed development screen. A pass permits a new untouched five-seed run; it is not itself a paper claim.

## Conservative training diagnostics

| seed | actor changed | accept rate | min LCB | mean shortfall |
|---:|---|---:|---:|---:|
| 1103 | True | 0.968 | -0.0165 | 0.0050 |
| 1213 | True | 1.000 | -0.0033 | 0.0070 |
| 1301 | True | 0.959 | -0.0056 | 0.0084 |
