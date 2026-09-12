# Anchored residual development result

Decision: **FAIL** for promotion to an untouched five-seed confirmation.

| arm | mean delta | relative gain | wins | max term gap |
|---|---:|---:|---:|---:|
| anchored_base | 0.1 | 6.0% | 1/3 | 0.000 |
| oracle_residual | 31.8 | 8.3% | 1/3 | 0.000 |
| oracle_safe | 36.4 | 7.9% | 2/3 | 0.000 |
| learned_raw | 9.1 | 6.2% | 2/3 | 0.000 |
| learned_safe | 27.8 | 7.5% | 1/3 | 0.000 |

| seed | mode mask | robust | base | oracle | learned safe |
|---:|---|---:|---:|---:|---:|
| 1103 | 0000 | 1465.7 | 1402.9 | 1381.1 | 1402.9 |
| 1213 | 0000 | 610.6 | 781.5 | 801.1 | 781.5 |
| 1301 | 1010 | 1910.5 | 1802.7 | 1900.1 | 1885.7 |

Failed gates:
- anchored base did not preserve paired robust control
- zero-initialized oracle residual lacked stable headroom
- calibrated learned fallback lacked stable gain
- calibration did not enable a residual mode for every seed

This is a three-seed development screen. A pass permits a new untouched five-seed run; it is not itself a paper claim.
