# BAPR-v3 shared-fork audit: deterministic_mean / Ant

Validated one exact producer pair and **30/30** controller outputs across event seeds `1100,1200,1300,1400,1500`.

Pair numerical gate: **FAIL**.

## deterministic_mean / Ant

| Controller | Stationary mean ± SD | Switching mean ± SD |
|---|---:|---:|
| equal-budget robust | 1985.2 ± 305.5 | 2823.9 ± 118.7 |
| dynamic oracle | 1718.4 ± 194.2 | 2439.6 ± 137.9 |
| fixed context 0 | 1619.5 ± 269.5 | 2383.8 ± 90.1 |
| fixed context 1 | 1972.1 ± 43.4 | 2256.4 ± 63.7 |
| fixed context 2 | 1573.2 ± 70.4 | 2329.6 ± 102.3 |
| fixed context 3 | 1178.4 ± 115.6 | 2009.1 ± 387.5 |

| Paired comparison | Stationary difference (95% CI) | Wins | Switching difference (95% CI) | Wins | Mean gate |
|---|---:|---:|---:|---:|---:|
| dynamic oracle - equal-budget robust | -266.8 [-824.0, +290.4] | 1/5 | -384.3 [-641.4, -127.3] | 0/5 | FAIL |
| dynamic oracle - fixed context 0 | +98.9 [-43.2, +241.1] | 4/5 | +55.8 [-182.1, +293.8] | 4/5 | PASS |
| dynamic oracle - fixed context 1 | -253.7 [-522.3, +15.0] | 1/5 | +183.2 [+56.0, +310.5] | 5/5 | FAIL |
| dynamic oracle - fixed context 2 | +145.2 [-152.0, +442.4] | 4/5 | +110.1 [-147.6, +367.7] | 4/5 | PASS |
| dynamic oracle - fixed context 3 | +540.0 [+378.2, +701.9] | 5/5 | +430.6 [-140.6, +1001.7] | 4/5 | PASS |

| Physics mode | Fixed context 0 | Fixed context 1 | Fixed context 2 | Fixed context 3 | Best | Diagonal? |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1762.4 | 2005.0 | 1540.2 | 1191.9 | 1 | no |
| 1 | 587.5 | 1501.8 | 528.4 | 386.0 | 1 | yes |
| 2 | 1817.6 | 1987.9 | 1790.2 | 1316.4 | 1 | no |
| 3 | 2310.3 | 2393.8 | 2434.0 | 1819.3 | 2 | no |

- Dynamic oracle beats robust and every fixed context in stationary paired mean: **FAIL**
- Dynamic oracle beats robust and every fixed context in switching paired mean: **FAIL**
- Stationary fixed-context diagonal optima: **1/4 (FAIL)**
- Preregistered family/environment gate: **FAIL**


The event seeds are paired evaluation streams for one training seed (`seed=0`), not independent policy seeds.
