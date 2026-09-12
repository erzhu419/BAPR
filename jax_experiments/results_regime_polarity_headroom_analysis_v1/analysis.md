# Equal-budget regime-control headroom result

The inferential unit is the independent training seed (`n=3`). 3 sealed event seeds are averaged inside each training seed. The oracle receives the true current persistent mode; the robust arm receives an all-zero vector through the same architecture.

| Env | Robust switch | Oracle switch | Switch delta 95% CI | Rel. | Robust worst | Oracle worst | Worst delta 95% CI | Rel. | Mode wins | Term gap | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| HalfCheetah | 738.8 +/- 160.1 | 2633.9 +/- 750.3 | +1895.1 [-359.2, +4149.5] | 256.5% | 138.2 +/- 416.5 | 1348.2 +/- 855.6 | +1210.0 [-386.6, +2806.7] | 875.6% | 4/4 | +0.0 pp | no |
| Ant | 1244.8 +/- 346.7 | 2304.5 +/- 931.1 | +1059.7 [-695.6, +2815.0] | 85.1% | 663.9 +/- 62.8 | 1602.2 +/- 1262.8 | +938.2 [-2118.9, +3995.4] | 141.3% | 4/4 | +0.0 pp | no |
| Hopper | 2671.3 +/- 72.2 | 2750.4 +/- 147.1 | +79.1 [-443.7, +602.0] | 3.0% | 168.1 +/- 30.1 | 192.1 +/- 63.1 | +24.0 [-204.7, +252.7] | 14.3% | 3/4 | +0.0 pp | no |
| Walker2d | 2287.5 +/- 81.4 | 2302.6 +/- 98.1 | +15.2 [-280.1, +310.5] | 0.7% | 176.6 +/- 24.8 | 200.9 +/- 20.1 | +24.3 [-30.1, +78.7] | 13.8% | 2/4 | +0.0 pp | no |

## Preregistered decision

Passing environments: **0/4**. Learned-estimator gate: **FAIL**.

An environment passes only when switching and worst-mode gains are both at least 15%, both paired 95% intervals are above zero, at least 3/4 stationary modes improve, and switching termination does not increase by more than 5 percentage points.

Do not train another estimator on this benchmark: privileged mode information did not establish sufficient controller headroom.
