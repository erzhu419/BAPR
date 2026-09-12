# Frozen transient-fallback cross-student audit

Frozen config: `evidence_1p0_k1`.

| student seed | robust 719 | learned | oracle | fallback | fallback - robust | fallback - learned | fallback use | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1709 | 1920.6 | 1942.4 | 2082.5 | 2009.0 | +88.4 (5/5) | +66.7 (5/5) | 1.16% | True |
| 1811 | 1920.6 | 1939.4 | 2089.4 | 1999.1 | +78.5 (5/5) | +59.6 (5/5) | 1.14% | False |
| 1901 | 1920.6 | 1856.9 | 2045.2 | 1999.8 | +79.3 (5/5) | +142.9 (5/5) | 1.18% | False |

## Decision

- Student passes: `1/3`.
- Cross-initialization gate: `False`.

The fallback effect is initialization-dependent. Do not train a new belief-augmented student from this branch; retain the passing seeds as diagnostics and analyze the failed initialization.
