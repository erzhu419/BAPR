# V21 full-state-final equal-policy-budget confirmation

The full-state-final specialist recipe and v5 posterior were frozen before these new policy seeds, event streams, and baseline runs.

| Seed | Robust | BAPR oracle | BAPR v5 | ESCP | RE-SAC | SAC5 causal | Recovery | Stationary retention |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 84003 | 1220.2 | 2950.7 | 2767.8 | 1080.9 | 1432.4 | 2201.4 | 89.4% | 137.8% |
| 84021 | 1785.2 | 3646.5 | 3734.5 | 1981.4 | 1743.4 | 2388.4 | 104.7% | 154.7% |
| 84039 | 1927.2 | 3342.9 | 3236.2 | 972.1 | 2137.7 | 1978.4 | 92.5% | 155.7% |
| 84057 | 959.0 | 3080.8 | 2882.2 | 2021.4 | 1868.1 | 2660.1 | 90.6% | 126.1% |
| 84079 | 1352.5 | 3853.8 | 3365.5 | 758.5 | 1273.4 | 2761.0 | 80.5% | 157.1% |

| Comparator | Paired difference | 95% CI | Seed wins | Event wins | Pass |
|---|---:|---:|---:|---:|:---:|
| robust_sac | +1748.5 | [+1368.7, +2128.2] | 5/5 | 15/15 | yes |
| escp_recurrent | +1834.4 | [+1011.4, +2657.4] | 5/5 | 15/15 | yes |
| resac_b0 | +1506.3 | [+880.6, +2132.0] | 5/5 | 15/15 | yes |
| sac5_v5_posterior_map | +799.4 | [+199.3, +1399.5] | 5/5 | 15/15 | yes |

Oracle recovery: **5/5**; stationary retention: **5/5**.
Strong final algorithm claim: **True**.
Limited adaptation claim: **False**.
Diagnosis: **full_state_final_bapr_supported_at_equal_policy_budget**.
