# V18 frozen-v5 final equal-policy-budget comparison

The v17 policy bank and v5 posterior were frozen before these event streams and baseline runs.

| Seed | Robust | BAPR oracle | BAPR v5 | ESCP | RE-SAC | SAC5 causal | Recovery | Stationary retention |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 81003 | 1356.4 | 2590.7 | 2629.2 | 977.7 | 1445.3 | 2108.9 | 103.1% | 135.2% |
| 81021 | 1773.5 | 2956.3 | 2874.7 | 1532.2 | 1657.9 | 2637.5 | 93.1% | 116.5% |
| 81039 | 1555.8 | 2287.2 | 2052.8 | 1651.1 | 1960.9 | 2752.4 | 68.0% | 91.0% |
| 81057 | 1501.7 | 3070.9 | 2945.8 | 1711.5 | 1284.9 | 2244.4 | 92.0% | 146.2% |
| 81079 | 2270.5 | 3645.7 | 3513.0 | 1691.0 | 1883.0 | 2290.2 | 90.4% | 149.0% |

| Comparator | Paired difference | 95% CI | Seed wins | Event wins | Pass |
|---|---:|---:|---:|---:|:---:|
| robust_sac | +1111.5 | [+659.0, +1564.0] | 5/5 | 15/15 | yes |
| escp_recurrent | +1290.4 | [+608.0, +1972.8] | 5/5 | 15/15 | yes |
| resac_b0 | +1156.7 | [+367.6, +1945.8] | 5/5 | 14/15 | yes |
| sac5_v5_posterior_map | +396.4 | [-485.3, +1278.2] | 4/5 | 11/15 | no |

Oracle recovery: **4/5**; stationary retention: **4/5**.
Strong final algorithm claim: **False**.
Limited adaptation claim: **True**.
Diagnosis: **adaptation_supported_but_not_equal_policy_budget_advantage**.
