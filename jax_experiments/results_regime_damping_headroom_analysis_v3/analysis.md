# Equal-budget regime-control headroom result

The inferential unit is the independent training seed (`n=3`). 3 sealed event seeds are averaged inside each training seed. The oracle receives the true current persistent mode; the robust arm receives an all-zero vector through the same architecture.

| Env | Robust switch | Oracle switch | Switch delta 95% CI | Rel. | Robust worst | Oracle worst | Worst delta 95% CI | Rel. | Mode wins | Term gap | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| HalfCheetah | 2523.7 +/- 992.1 | 2923.2 +/- 1409.3 | +399.5 [-3858.0, +4656.9] | 15.8% | 2511.9 +/- 962.9 | 2887.0 +/- 1394.6 | +375.1 [-3794.5, +4544.8] | 14.9% | 4/4 | +0.0 pp | no |
| Ant | 3340.7 +/- 452.7 | 3089.2 +/- 512.7 | -251.5 [-2588.4, +2085.4] | -7.5% | 3156.1 +/- 303.3 | 2955.9 +/- 392.5 | -200.2 [-1779.7, +1379.3] | -6.3% | 0/4 | +0.0 pp | no |
| Hopper | 2873.3 +/- 130.3 | 2990.6 +/- 66.3 | +117.4 [-65.2, +300.0] | 4.1% | 246.1 +/- 33.1 | 277.4 +/- 11.1 | +31.3 [-25.7, +88.3] | 12.7% | 4/4 | +0.0 pp | no |
| Walker2d | 2253.1 +/- 47.9 | 2332.2 +/- 94.9 | +79.1 [-266.4, +424.7] | 3.5% | 242.3 +/- 52.3 | 155.8 +/- 61.3 | -86.4 [-252.9, +80.1] | -35.7% | 0/4 | +0.0 pp | no |

## Preregistered decision

Passing environments: **0/4**. Learned-estimator gate: **FAIL**.

An environment passes only when switching and worst-mode gains are both at least 10%, both paired 95% intervals are above zero, at least 3/4 stationary modes improve, and switching termination does not increase by more than 5 percentage points.

Do not train another estimator on this benchmark: privileged mode information did not establish sufficient controller headroom.
