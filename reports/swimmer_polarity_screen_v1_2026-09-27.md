# Swimmer polarity screen: completed negative result

All ten scheduler tasks t100673-t100682 completed, including six training runs,
three strict audits, and the aggregate. All six final bundles record iteration
2100 (next iteration), 8.4M interactions, and 525,000 updates. Each audited route
has 15 episodes per policy seed, spanning three event streams.

| Policy seed | Robust SAC | True-mode compensation | Paired difference | Headroom gate |
|---|---:|---:|---:|---|
| 91003 | 65.66 | 51.19 | -14.48 | Fail |
| 91021 | 42.34 | 49.72 | +7.38 | Pass |
| 91039 | 57.12 | 49.98 | -7.14 | Fail |
| Mean | 55.04 | 50.30 | -4.75 (-8.6%) | Fail |

All episodes reach the full 1,000-step horizon. True-mode compensation and
native reference-mode execution have exactly equal paired episode returns
(maximum difference 0.0). The uncompensated canonical mean is 6.07. Thus the
transform works, but this canonical-controller recipe does not outperform
robust control even with privileged mode information. The registered mean
headroom and 2/3-seed requirements both fail. This is a valid negative screen,
not an incomplete run or an evaluation-budget mismatch.

## Paper decision

Retain the frozen V32 single-reference policy plus causal action compensation
as the primary result: ten independent HalfCheetah policy seeds, positive
paired mean-effect intervals against SAC, ESCP, and RE-SAC. V33 did not establish
a consistent recipe improvement. Keep Swimmer as a separate negative upper-bound
screen; do not train its estimator or retune its modes. No new jobs were submitted.

Sources: `reports/swimmer_polarity_screen_v1_preregistration_2026-09-26.md`,
`jax_experiments/results_swimmer_polarity_screen_v1/analysis/analysis.json`, and
the three `audits/seed_*/audit.json` records under the same result directory.
The compact outputs were already synchronized; no checkpoints were pulled.
