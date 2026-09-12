# Anchored residual development result

Decision: **FAIL** for promotion to an untouched five-seed confirmation.

| arm | mean delta | relative gain | wins | max term gap |
|---|---:|---:|---:|---:|
| anchored_base | -361.3 | -38.6% | 1/3 | 0.000 |
| oracle_residual | -760.5 | -72.4% | 1/3 | 0.000 |
| oracle_safe | -347.0 | -37.8% | 1/3 | 0.000 |
| learned_raw | -755.1 | -71.8% | 1/3 | 0.000 |
| learned_safe | -349.2 | -37.9% | 1/3 | 0.000 |

| seed | mode mask | robust | base | oracle | learned safe |
|---:|---|---:|---:|---:|---:|
| 1103 | 0000 | 1402.9 | 848.2 | -76.9 | 848.2 |
| 1213 | 0000 | 774.1 | 139.0 | -137.6 | 139.0 |
| 1301 | 0111 | 1823.0 | 1929.0 | 1933.1 | 1965.4 |

Failed gates:
- anchored base did not preserve paired robust control
- zero-initialized oracle residual lacked stable headroom
- calibrated learned fallback lacked stable gain
- frozen estimator failed under the new controller
- calibration disabled every residual mode

This is a three-seed development screen. A pass permits a new untouched five-seed run; it is not itself a paper claim.
