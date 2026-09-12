# Causal transient-fallback audit

Development-selected config: `evidence_1p0_k1`.

| arm | switching return |
|---|---:|
| `robust_final_seed_719` | 1919.2 |
| `student_learned` | 1929.6 |
| `student_oracle` | 2032.9 |
| `evidence_1p0_k1` | 2035.6 |

| comparison | delta | event wins | clustered 95% interval |
|---|---:|---:|---:|
| `fallback_minus_learned` | +106.0 | 5/5 | [+67.4, +144.7] |
| `fallback_minus_robust719` | +116.4 | 5/5 | [+64.1, +168.7] |
| `fallback_minus_oracle` | +2.7 | 2/5 | [-41.0, +46.4] |
| `learned_minus_robust719` | +10.4 | 3/5 | [-14.8, +35.5] |
| `oracle_minus_robust719` | +113.7 | 5/5 | [+94.4, +133.0] |

## Decision

- Oracle headroom recovery: `102.4%`.
- Fallback action fraction: `1.1%`.
- Zero termination: `True`.
- Beats learned posterior: `True`.
- Beats robust 719: `True`.
- Authorize stale/soft-belief training: `True`.

The frozen causal fallback clears its independent gate. A new development-only student may now be trained with stale and soft belief augmentation while retaining the same explicit robust fallback; these audit events remain sealed.
