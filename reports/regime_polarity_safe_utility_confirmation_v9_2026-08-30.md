# Independent posterior-MAP safe-utility confirmation

The per-mode robust/specialist map is selected on three stationary calibration streams. All switching returns use three disjoint holdout streams. ESCP and RE-SAC are trained from scratch with the same seed, environment, horizon, and per-controller interaction budget.

| Seed | Map | Robust SAC | Safe oracle | MAP utility | ESCP | RE-SAC | MAP gain | Recovery | Gate |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 61003 | `[S0,S1,S2,S3]` | 1032.1 | 2886.0 | 2625.2 | 1106.4 | 1641.0 | 154.4% | 85.9% | True |
| 61021 | `[S0,R,S2,S3]` | 2294.2 | 2535.3 | 2320.1 | 1134.1 | 1751.1 | 1.1% | 10.8% | False |
| 61039 | `[R,R,R,R]` | 2350.3 | 2350.3 | 2350.3 | 1747.5 | 1406.9 | 0.0% | n/a | True |
| 61057 | `[R,R,R,R]` | 2714.3 | 2714.3 | 2714.3 | 1860.7 | 1612.5 | 0.0% | n/a | True |
| 61079 | `[R,S1,S2,S3]` | 1361.7 | 1766.1 | 1665.3 | 1410.8 | 1009.1 | 22.3% | 75.1% | True |

Composition gate: **False**. Paired baseline confirmation: **False**.

| Comparator | MAP mean | Comparator mean | Difference | 95% paired CI | Seed wins | Event wins | Confirmed |
|:---|---:|---:|---:|:---:|:---:|:---:|:---:|
| robust_sac | 2335.0 | 1950.5 | 384.5 | [-469.3, 1238.4] | 3/5 | 8/15 | False |
| escp_recurrent | 2335.0 | 1451.9 | 883.2 | [271.6, 1494.8] | 5/5 | 15/15 | True |
| resac_b0 | 2335.0 | 1484.1 | 850.9 | [568.7, 1133.1] | 5/5 | 15/15 | True |

## Cost accounting

Each BAPR bank trains five policies: 28,000,000 environment steps and 1,750,000 updates per seed. Each SAC/ESCP/RE-SAC comparator trains one policy: 5,600,000 steps and 350,000 updates. The policy-training ratio is therefore 5x; frozen estimator pretraining is additional and excluded from that ratio.

Conclusion: the v8 safe-utility result does not transfer to every independent policy bank under the frozen gate.

Decision: stop expanding this policy-bank candidate and retain it only as a diagnostic upper-bound construction.
