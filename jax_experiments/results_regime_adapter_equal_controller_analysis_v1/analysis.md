# Equal-per-controller adapter upper bound

**Diagnostic only.** Each of four fixed-mode adapters receives the same 2.8M post-fork transitions as the single robust continuation. The bank therefore uses four times the post-fork data and is not a paper comparison.

| Seed | Robust stat | Identity stat | Gain | Robust switch | Identity switch | Gain | Identity-best fixed switch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 2295.2 | 2551.5 | +11.2% | 2163.8 | 2412.3 | +11.5% | +92.6 |
| 24 | 1993.4 | 1608.7 | -19.3% | 1932.3 | 1593.2 | -17.5% | -48.3 |
| 32 | 2431.3 | 2089.8 | -14.0% | 2395.5 | 1904.8 | -20.5% | +265.7 |
| 40 | 2436.7 | 2386.9 | -2.0% | 2300.8 | 2213.9 | -3.8% | -63.2 |

## Paired inference

- Identity minus robust stationary: -129.9, 95% CI [-542.7, +282.9], relative -5.7%, wins 1/4.
- Identity minus robust switching: -167.0, 95% CI [-615.8, +281.7], relative -7.6%, wins 1/4.
- Identity minus best fixed stationary: +128.7, 95% CI [-49.7, +307.1].
- Identity minus best fixed switching: +61.7, 95% CI [-150.8, +274.2].
- Correct-mode controller is stationary-optimal in 12/16 rows.
- Decision gate: FAIL.

This deliberately spends four times the robust post-fork data. A pass only establishes controller specialization headroom and authorizes a compute-efficient shared-backbone redesign; it is not a publishable BAPR comparison.

Decision: `reject_independent_adapter_optimization`.
