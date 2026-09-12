# Frozen-estimator transfer result

The v12 policy banks, robust-inclusive utility maps, and v5 estimator are frozen. This audit performs no training.

| Seed | Robust | Safe oracle | Delay-4 oracle | Frozen-v5 MAP | Oracle gain | Delay recovery | MAP gain | MAP recovery | MAP accuracy | Pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 71003 | 2853.8 | 3608.9 | 3298.0 | 3344.2 | 26.5% | 58.8% | 17.2% | 64.9% | 98.6% | no |
| 71021 | 2415.7 | 3895.9 | 3852.3 | 3235.0 | 61.3% | 97.1% | 33.9% | 55.4% | 98.7% | no |
| 71039 | 1104.2 | 2872.4 | 2880.9 | 2752.9 | 160.1% | 100.5% | 149.3% | 93.2% | 98.9% | yes |
| 71057 | 2400.7 | 3796.5 | 3741.9 | 3687.4 | 58.1% | 96.1% | 53.6% | 92.2% | 98.9% | yes |
| 71079 | 1695.3 | 2430.5 | 2648.6 | 2543.1 | 43.4% | 129.7% | 50.0% | 115.3% | 98.8% | yes |

Headroom seeds: **5/5**. Delay-4 passes: **4/5**. Frozen-estimator passes: **3/5**.

Diagnosis: **frozen_v5_estimator_transfer_is_limiting**.
Frozen estimator confirmed: **False**.
Estimator retraining authorized: **True**.
