# Robust-inclusive specialist utility on fresh policy banks

Each per-mode map is frozen on three stationary calibration streams. A matching specialist is enabled only when it beats robust by at least 5%, wins all three calibration streams, and has zero termination. All reported switching results use three untouched event streams.

Banks with at least 10% safe-oracle headroom must gain at least 10%, recover at least 70% of that headroom, and win all holdout streams. Banks without such headroom must remain within 5% of robust on every holdout stream.

| Seed | Map | Robust | Safe oracle | MAP utility | Confirm-3 utility | Confirm gain | Confirm recovery | Headroom | MAP pass | Confirm pass |
|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| 5003 | `[R,R,R,R]` | 2980.6 | 2980.6 | 2980.6 | 2980.6 | 0.0% | n/a | False | True | True |
| 5021 | `[S0,R,S2,R]` | 2164.6 | 2675.8 | 2601.5 | 2616.4 | 20.9% | 88.4% | True | True | True |
| 5039 | `[S0,S1,R,S3]` | 2277.8 | 2750.8 | 2789.5 | 2686.3 | 17.9% | 86.4% | True | True | True |
| 5051 | `[S0,S1,S2,S3]` | 1822.5 | 3948.8 | 3638.2 | 3747.5 | 105.6% | 90.5% | True | True | True |
| 5077 | `[S0,R,S2,S3]` | 1879.6 | 2321.0 | 2201.4 | 2176.4 | 15.8% | 67.2% | True | True | False |

Mean confirm-3 return: **2841.4**; robust: **2225.0**; safe oracle: **2935.4**.

Mean gain: **32.0%**; seed wins: **4/5**; event wins: **12/15**; mean headroom recovery: **83.1%**.

Mean MAP return: **2842.2**; mean gain: **31.9%**; mean recovery on headroom banks: **88.0%**; seed wins: **4/5**; event wins: **12/15**.

MAP composition pass: **True**; confirm-3 primary pass: **False**.

Conclusion: the simpler posterior-MAP safe-utility router passes every fresh bank; confirm-3 is an unnecessary source of transient loss.

Decision: freeze posterior-MAP safe utility as the candidate and run a fully new policy-bank confirmation with explicit training-cost accounting.
