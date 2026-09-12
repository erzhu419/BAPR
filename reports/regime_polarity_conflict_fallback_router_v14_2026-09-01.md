# Evidence-conflict fallback router development result

All controllers, utility maps, and estimator parameters are frozen. Candidate routers use no mode ID or switch clock.

| Candidate | Seed passes | Mean return | Mean recovery | Fallback actions |
|:---|---:|---:|---:|---:|
| conflict_fallback_confirm1_safe_utility | 3/5 | 3180.7 | 81.7% | 1.0% |
| conflict_fallback_confirm2_safe_utility | 3/5 | 3178.1 | 80.2% | 1.7% |
| conflict_fallback_confirm3_safe_utility | 4/5 | 3257.1 | 90.9% | 2.9% |

Selected arm: **conflict_fallback_confirm3_safe_utility**.
Headroom seeds: **4/5**; delay-4 passes: **4/5**.
Diagnosis: **causal_conflict_fallback_router_supported_in_development**.
Router development pass: **True**.
Estimator retraining needed: **False**.
