# Fresh-policy-bank estimator confirmation result

All robust sources and actor-only specialists use new policy seeds; v5 and v16 estimator parameters remain frozen.

| Seed | Modes | Robust | Oracle | V5 MAP | V16 MAP | d(V5) | Switch acc d | Wrong-action d | Strict | Mechanism |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 81003 | 4/4 | 1314.0 | 2623.2 | 2569.0 | 2573.3 | +4.3 | -6.5% | 0.3% | yes | no |
| 81021 | 4/4 | 1845.5 | 2859.8 | 2524.6 | 2510.7 | -13.9 | -9.2% | 0.4% | no | no |
| 81039 | 3/4 | 1527.2 | 2056.6 | 2117.2 | 2100.5 | -16.7 | -8.5% | 0.3% | yes | no |
| 81057 | 4/4 | 1514.7 | 3072.1 | 2993.7 | 2885.8 | -107.9 | -11.9% | 0.4% | yes | no |
| 81079 | 4/4 | 2247.6 | 3707.8 | 3476.8 | 3528.4 | +51.6 | -5.8% | 0.2% | yes | no |

Policy banks: **5/5**; headroom: **5/5**; delay-4: **5/5**; strict v16: **4/5**.
V16 versus v5: **2/5** seed wins, paired mean **-16.5**, 95% CI [-88.4, +55.4].
Mechanism gate: **0/5** seeds.
Diagnosis: **v16_fails_fresh_bank_seed_win_gate**.
Estimator confirmation pass: **False**.
