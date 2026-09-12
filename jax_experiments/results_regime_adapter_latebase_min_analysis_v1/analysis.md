# Late-base min-target adapter diagnostic

**Diagnostic only.** The two previously failing seeds start from the completed 8.4M-step robust controller. Four canonically initialized fixed-mode residuals each receive 0.7M additional transitions.

| Seed | Robust stat | Identity stat | Gain | Robust switch | Identity switch | Gain | Diagonal optimal | Base-copy error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 2009.1 | 2247.0 | +11.8% | 1935.2 | 2101.7 | +8.6% | 4/4 | 0.00% |
| 32 | 2410.9 | 2485.4 | +3.1% | 2384.8 | 2418.1 | +1.4% | 2/4 | 0.00% |

- Mean stationary relative gain: +7.5%.
- Mean switching relative gain: +5.0%.
- Decision gate: FAIL.

A pass only shows that a mature robust controller has usable fixed-mode residual headroom under matched critic targets and a fixed entropy temperature. It authorizes switch-distribution training; it is not a publishable BAPR result.

Decision: `reject_latebase_fixed_mode_residual_headroom`.
