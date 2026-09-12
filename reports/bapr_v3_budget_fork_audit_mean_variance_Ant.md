# BAPR-v3 shared-fork audit: mean_variance / Ant

Validated one exact producer pair and **30/30** controller outputs across event seeds `1100,1200,1300,1400,1500`.

Pair numerical gate: **FAIL**.

## mean_variance / Ant

| Controller | Stationary mean ± SD | Switching mean ± SD |
|---|---:|---:|
| equal-budget robust | 1570.2 ± 182.0 | 2714.8 ± 63.9 |
| dynamic oracle | 1415.9 ± 105.0 | 2410.1 ± 214.7 |
| fixed context 0 | 1430.3 ± 213.5 | 2440.5 ± 70.4 |
| fixed context 1 | 1955.9 ± 228.7 | 2463.0 ± 85.8 |
| fixed context 2 | 1146.6 ± 161.8 | 2236.3 ± 194.2 |
| fixed context 3 | 1198.1 ± 267.9 | 2024.1 ± 369.4 |

| Paired comparison | Stationary difference (95% CI) | Wins | Switching difference (95% CI) | Wins | Mean gate |
|---|---:|---:|---:|---:|---:|
| dynamic oracle - equal-budget robust | -154.4 [-426.6, +117.9] | 2/5 | -304.6 [-509.0, -100.3] | 0/5 | FAIL |
| dynamic oracle - fixed context 0 | -14.4 [-193.2, +164.3] | 3/5 | -30.4 [-269.3, +208.5] | 2/5 | FAIL |
| dynamic oracle - fixed context 1 | -540.0 [-801.7, -278.3] | 0/5 | -52.9 [-302.5, +196.8] | 3/5 | FAIL |
| dynamic oracle - fixed context 2 | +269.3 [-60.2, +598.8] | 4/5 | +173.8 [-287.1, +634.8] | 4/5 | PASS |
| dynamic oracle - fixed context 3 | +217.8 [-141.6, +577.2] | 4/5 | +386.0 [-96.3, +868.2] | 4/5 | PASS |

| Physics mode | Fixed context 0 | Fixed context 1 | Fixed context 2 | Fixed context 3 | Best | Diagonal? |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1241.5 | 2019.9 | 835.6 | 1013.7 | 1 | no |
| 1 | 781.6 | 1574.9 | 708.1 | 830.0 | 1 | yes |
| 2 | 1684.0 | 1991.0 | 972.0 | 1073.6 | 1 | no |
| 3 | 2014.0 | 2237.7 | 2070.6 | 1875.1 | 1 | no |

- Dynamic oracle beats robust and every fixed context in stationary paired mean: **FAIL**
- Dynamic oracle beats robust and every fixed context in switching paired mean: **FAIL**
- Stationary fixed-context diagonal optima: **1/4 (FAIL)**
- Preregistered family/environment gate: **FAIL**


The event seeds are paired evaluation streams for one training seed (`seed=0`), not independent policy seeds.
