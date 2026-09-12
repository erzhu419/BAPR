# Equal-budget regime-control headroom result

The inferential unit is the independent training seed (`n=5`). Five sealed event seeds are averaged inside each training seed. The oracle receives the true current persistent mode; the robust arm receives an all-zero vector through the same architecture.

| Env | Robust switch | Oracle switch | Switch delta 95% CI | Rel. | Robust worst | Oracle worst | Worst delta 95% CI | Rel. | Mode wins | Term gap | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| HalfCheetah | 1945.0 +/- 326.3 | 1317.7 +/- 324.4 | -627.4 [-1157.5, -97.3] | -32.3% | 1553.9 +/- 451.5 | 832.2 +/- 402.8 | -721.7 [-1688.2, +244.7] | -46.4% | 0/4 | +0.0 pp | no |
| Ant | 2930.8 +/- 173.7 | 3291.4 +/- 238.3 | +360.6 [+22.4, +698.8] | 12.3% | 2716.3 +/- 242.4 | 2906.6 +/- 176.2 | +190.3 [-135.7, +516.4] | 7.0% | 4/4 | +0.0 pp | no |
| Walker2d | 2177.4 +/- 73.0 | 2200.6 +/- 18.3 | +23.2 [-74.8, +121.3] | 1.1% | 192.7 +/- 54.2 | 172.2 +/- 38.7 | -20.5 [-114.5, +73.5] | -10.6% | 1/4 | +0.0 pp | no |

## Preregistered decision

Passing environments: **0/3**. Learned-estimator gate: **FAIL**.

An environment passes only when switching and worst-mode gains are both at least 10%, both paired 95% intervals are above zero, at least 3/4 stationary modes improve, and switching termination does not increase by more than 5 percentage points.

Do not train another estimator on this benchmark: privileged mode information did not establish sufficient controller headroom.
