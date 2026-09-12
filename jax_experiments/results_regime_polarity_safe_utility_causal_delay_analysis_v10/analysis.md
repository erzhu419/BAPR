# Safe-utility causal-delay diagnostic

All controllers, utility maps, and estimator parameters are frozen from v9. No model is trained in this diagnostic.

| Seed | Robust | Zero-delay safe oracle | Posterior MAP | Oracle gain | MAP recovery | Stale d1/d2/d4/d8 | Robust handoff d1/d2/d4/d8 |
|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 61003 | 897.9 | 3085.5 | 2848.6 | 243.6% | 89.2% | 96.4%/98.6%/94.7%/81.7% | 95.4%/91.0%/93.6%/95.3% |
| 61021 | 2236.1 | 2564.7 | 2374.2 | 14.7% | 42.0% | 99.1%/53.6%/76.6%/30.4% | 94.9%/64.1%/85.2%/72.7% |
| 61039 | 2367.5 | 2367.5 | 2367.5 | 0.0% | n/a | n/a/n/a/n/a/n/a | n/a/n/a/n/a/n/a |
| 61057 | 2739.2 | 2739.2 | 2739.2 | 0.0% | n/a | n/a/n/a/n/a/n/a | n/a/n/a/n/a/n/a |
| 61079 | 1312.8 | 1704.2 | 1983.3 | 29.8% | 171.3% | 156.0%/109.5%/121.2%/75.4% | 116.7%/120.6%/110.3%/165.0% |

Headroom seeds: **3/5**. Delay-4 retention seeds: **3/5**. Posterior recovery seeds: **2/5**.

Diagnosis: **policy_bank_headroom_not_reproducible**.
Estimator retraining authorized: **False**.

A stale-delay arm keeps the previous mapped controller after each regime onset. A robust-handoff arm uses robust SAC for the same privileged delay, then switches to the correct mapped controller.
