# Switch-weighted expected-action estimator development result

Policies, utility maps, environment, posterior filter, and MAP routing are frozen; only inverse-evidence training changed.

| Seed | Robust | Oracle | V5 MAP | V16 MAP | Gain | Recovery | d(V5) | Switch acc d | Strict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 71003 | 2836.0 | 3505.9 | 3354.9 | 3315.3 | 16.9% | 71.5% | -39.7 | -3.8% | yes |
| 71021 | 2559.3 | 3934.8 | 3742.8 | 3796.0 | 48.3% | 89.9% | +53.1 | -5.2% | yes |
| 71039 | 1094.5 | 2732.3 | 2497.3 | 2518.6 | 130.1% | 87.0% | +21.3 | -11.7% | yes |
| 71057 | 2424.0 | 3735.1 | 3447.6 | 3577.2 | 47.6% | 88.0% | +129.6 | -5.4% | yes |
| 71079 | 1677.3 | 2863.6 | 2559.6 | 2664.9 | 58.9% | 83.2% | +105.2 | -5.6% | yes |

Headroom: **5/5**; delay-4: **5/5**; strict v16: **5/5**.
V16 versus frozen v5: **4/5** seed wins, held-out policy wins **2/2**, paired mean **+53.9**, 95% CI [-29.8, +137.7].
Diagnosis: **switch_weighted_estimator_supported_in_development**.
Estimator development pass: **True**.
Fresh-policy-bank confirmation authorized: **True**.
