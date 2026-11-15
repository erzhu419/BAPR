# Stationary controller-capacity diagnostic for failed v6 banks

This checkpoint-only diagnostic cross-evaluates the robust controller and all four fixed-mode specialists from failed v6 policy banks on three fresh stationary event streams. It does not refit the estimator, router, or controller.

A bank has sufficient stationary adaptation capacity only when the diagonal specialist beats robust in at least 3/4 modes, the mean diagonal oracle gain is at least 10%, and diagonal termination is zero.

| Seed | Stationary robust | Stationary oracle | Oracle gain | Diagonal wins | v6 switching gain | Classification |
|---:|---:|---:|---:|---:|---:|:---|
| 5003 | 3144.5 | 2528.9 | -19.6% | 0/4 | -19.2% | controller_capacity_failure |
| 5021 | 2134.1 | 2500.5 | 17.2% | 2/4 | 8.3% | controller_capacity_failure |
| 5077 | 2273.8 | 2753.3 | 21.1% | 3/4 | 4.2% | switch_transient_failure |

## Per-mode diagonal comparison

| Seed | Mode | Robust | Specialist | Gain | Terminated |
|---:|---:|---:|---:|---:|---:|
| 5003 | 0 | 3837.3 | 3300.5 | -14.0% | 0.0% |
| 5003 | 1 | 2810.2 | 2327.2 | -17.2% | 0.0% |
| 5003 | 2 | 3078.3 | 2065.8 | -32.9% | 0.0% |
| 5003 | 3 | 2852.4 | 2421.9 | -15.1% | 0.0% |
| 5021 | 0 | 1415.5 | 2384.5 | 68.5% | 0.0% |
| 5021 | 1 | 3148.9 | 2481.7 | -21.2% | 0.0% |
| 5021 | 2 | 996.0 | 2710.8 | 172.2% | 0.0% |
| 5021 | 3 | 2976.1 | 2425.0 | -18.5% | 0.0% |
| 5077 | 0 | 1654.9 | 1915.8 | 15.8% | 0.0% |
| 5077 | 1 | 2314.3 | 2016.2 | -12.9% | 0.0% |
| 5077 | 2 | 2878.5 | 4169.2 | 44.8% | 0.0% |
| 5077 | 3 | 2247.4 | 2912.0 | 29.6% | 0.0% |

Controller-capacity failures: `[5003, 5021]`.
Switch-transient failures: `[5077]`.

Conclusion: failed v6 banks include intrinsic specialist-controller capacity failures; estimator or router tuning cannot recover that headroom.

Decision: freeze each robust actor and train bounded mode options with an explicit per-mode no-regression objective before revisiting routing.
