# Anchored residual development result

Decision: **FAIL** for promotion to an untouched five-seed confirmation.

| arm | mean delta | relative gain | wins | max term gap |
|---|---:|---:|---:|---:|
| anchored_base | -12.1 | 5.6% | 1/3 | 0.000 |
| oracle_residual | -12.1 | 5.6% | 1/3 | 0.000 |
| oracle_safe | -12.1 | 5.6% | 1/3 | 0.000 |
| learned_raw | -12.1 | 5.6% | 1/3 | 0.000 |
| learned_safe | -12.1 | 5.6% | 1/3 | 0.000 |

| seed | mode mask | robust | base | oracle | learned safe |
|---:|---|---:|---:|---:|---:|
| 1103 | 0000 | 1468.2 | 1410.7 | 1410.7 | 1410.7 |
| 1213 | 0000 | 600.5 | 772.3 | 772.3 | 772.3 |
| 1301 | 0000 | 1902.8 | 1752.3 | 1752.3 | 1752.3 |

Failed gates:
- anchored base did not preserve paired robust control
- zero-initialized oracle residual lacked stable headroom
- calibrated learned fallback lacked stable gain
- calibration did not enable a residual mode for every seed

This is a three-seed development screen. A pass permits a new untouched five-seed run; it is not itself a paper claim.

## Conservative training diagnostics

| seed | actor changed | accept rate | min LCB | mean shortfall |
|---:|---|---:|---:|---:|
| 1103 | False | 0.000 | 0.0000 | 0.0069 |
| 1213 | False | 0.000 | 0.0000 | 0.0069 |
| 1301 | False | 0.000 | 0.0000 | 0.0069 |
