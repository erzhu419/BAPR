# Frozen conflict-fallback router holdout confirmation

The v14 confirm-3 router and all controller, estimator, and utility-map parameters were frozen before these events.

| Seed | Robust | Oracle | Delay-4 | Plain MAP | Confirm-3 | Gain | Recovery | Strict | Beats MAP |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 71003 | 3159.7 | 3626.6 | 3480.8 | 3600.8 | 3456.4 | 9.4% | 63.5% | no | no |
| 71021 | 2454.3 | 3857.0 | 3841.7 | 3398.7 | 3737.5 | 52.3% | 91.5% | yes | yes |
| 71039 | 1159.8 | 2717.1 | 2761.4 | 2546.4 | 2595.5 | 123.8% | 92.2% | yes | yes |
| 71057 | 2377.6 | 3758.2 | 3700.1 | 3719.8 | 3666.7 | 54.2% | 93.4% | yes | no |
| 71079 | 1826.2 | 2623.8 | 2585.3 | 2580.6 | 2773.7 | 51.9% | 118.8% | yes | yes |

Headroom seeds: **5/5**; delay-4 passes: **4/5**; strict router passes: **4/5**.
Confirm-3 versus plain MAP: **3/5** seed wins, paired mean **+76.7**, 95% CI [-162.7, +316.1].
Diagnosis: **frozen_conflict_fallback_has_no_incremental_map_value**.
Holdout confirmation pass: **False**.
Fresh-policy-bank confirmation authorized: **False**.
