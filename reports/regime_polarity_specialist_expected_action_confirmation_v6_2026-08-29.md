# Fresh policy-bank confirmation for frozen v5

This graph freezes the v5 expected-action estimator, its selected filter, and the confirm-3 router. Only five new robust/specialist policy banks are trained. The controller seeds are `5003, 5021, 5039, 5051, 5077` and the event seeds are `155301, 155317, 155333`.

A source seed passes only when confirm-3 beats robust by at least 10%, recovers at least 70% of dynamic-oracle headroom, reaches return 2200, wins all three event streams, and has zero termination. The primary confirmation requires all five source seeds to pass.

| Seed | Robust | Oracle | Frozen v5 | Gain | Recovery | Accuracy | Event wins | Pass |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 5003 | 3182.9 | 2796.1 | 2572.7 | -19.2% | n/a | 98.33% | 0/3 | False |
| 5021 | 2184.5 | 2454.4 | 2365.0 | 8.3% | 66.9% | 98.20% | 3/3 | False |
| 5039 | 2304.3 | 2951.0 | 2782.9 | 20.8% | 74.0% | 98.23% | 3/3 | True |
| 5051 | 1830.0 | 3958.2 | 3607.6 | 97.1% | 83.5% | 98.29% | 3/3 | True |
| 5077 | 1886.1 | 2030.4 | 1965.4 | 4.2% | 55.0% | 98.15% | 3/3 | False |

Mean frozen-v5 return: **2658.7**; robust: **2277.5**; oracle: **2838.0**.

Mean relative gain: **22.2%**; mean oracle recovery on banks with positive oracle headroom: **69.8%** (4/5 banks); seed wins: **4/5**; event wins: **12/15**.

Primary pass: **False**

Diagnosis: v5 is directionally positive but does not meet the strict per-bank confirmation gate.

Decision: report heterogeneous controller-bank transfer and inspect only the failed banks before authorizing new training.
