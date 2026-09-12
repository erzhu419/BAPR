# Equal-budget regime-control headroom result

The inferential unit is the independent training seed (`n=5`). 3 sealed event seeds are averaged inside each training seed. The oracle receives the true current persistent mode; the robust arm receives an all-zero vector through the same architecture.

| Env | Robust switch | Oracle switch | Switch delta 95% CI | Rel. | Robust worst | Oracle worst | Worst delta 95% CI | Rel. | Mode wins | Term gap | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| HalfCheetah | 1033.8 +/- 318.8 | 2308.0 +/- 475.4 | +1274.2 [+750.2, +1798.1] | 123.3% | 682.2 +/- 289.9 | 2012.3 +/- 486.4 | +1330.1 [+932.7, +1727.5] | 195.0% | 4/4 | +0.0 pp | yes |

## Preregistered decision

Passing environments: **1/1**. Learned-estimator gate: **PASS**.

An environment passes only when switching and worst-mode gains are both at least 10%, both paired 95% intervals are above zero, at least 3/4 stationary modes improve, and switching termination does not increase by more than 5 percentage points.

Proceed to a causal learned mode estimator trained against this oracle interface.
