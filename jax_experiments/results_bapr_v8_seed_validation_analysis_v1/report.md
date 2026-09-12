# BAPR-v8 independent-training-seed validation

Promotion gate: **FAIL**

The independent statistical unit is the policy training seed. Each seed is evaluated on the same five paired, previously sealed event streams.

| Controller | Stationary | Slow switching | Full cycle | Full-cycle termination |
|---|---:|---:|---:|---:|
| sac | 1674.8 +/- 213.8 | 1682.8 +/- 185.1 | 1689.6 +/- 208.1 | 0.000 +/- 0.000 |
| escp | 2039.5 +/- 528.9 | 1987.2 +/- 484.1 | 2007.1 +/- 523.9 | 0.000 +/- 0.000 |
| resac | -188.6 +/- 34.9 | -189.6 +/- 35.7 | -188.7 +/- 35.0 | 0.000 +/- 0.000 |
| bapr | 1474.1 +/- 367.8 | 1441.9 +/- 352.5 | 1400.4 +/- 432.5 | 0.000 +/- 0.000 |
| oracle | 2267.3 +/- 352.7 | 2226.5 +/- 453.3 | 2168.1 +/- 413.9 | 0.000 +/- 0.000 |

| Paired full-cycle contrast | Mean | 95% CI | Wins |
|---|---:|---:|---:|
| BAPR - sac | -289.2 | [-742.7, +164.3] | 2/5 |
| BAPR - escp | -606.7 | [-1519.2, +305.8] | 1/5 |
| BAPR - resac | +1589.2 | [+1033.6, +2144.7] | 5/5 |

Against the strongest baseline selected independently within each training seed, BAPR is -686.5 with 95% CI [-1430.4, +57.3] and wins 1/5 seeds.

Gate details: `positive_full_cycle_ci_vs_strongest=false, at_least_four_of_five_full_cycle_wins=false, stationary_noninferiority_100=false, termination_noninferiority_0p05=true, pass=false`
