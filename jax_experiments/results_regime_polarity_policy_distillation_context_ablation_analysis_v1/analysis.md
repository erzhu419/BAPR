# Frozen mode-head context-ablation diagnostic

This is a post-confirmation mechanism diagnostic. It cannot select another model or reopen the failed confirmation.

| arm | switching return |
|---|---:|
| `robust_final_seed_719` | 1933.9 |
| `teacher_oracle_median` | 2053.6 |
| `teacher_learned_median` | 1895.8 |
| `student_learned` | 1916.3 |
| `student_oracle` | 2046.7 |
| `student_uniform` | -14.1 |
| `student_fixed_0` | 815.5 |
| `student_fixed_1` | 847.1 |
| `student_fixed_2` | -268.1 |
| `student_fixed_3` | -328.8 |
| `student_cyclic` | -197.3 |
| `student_shuffled` | -365.7 |

## Paired context comparisons

| comparison | delta | event wins | clustered 95% interval |
|---|---:|---:|---:|
| `learned_minus_robust719` | -17.6 | 2/5 | [-108.2, +73.0] |
| `oracle_minus_robust719` | +112.8 | 5/5 | [+75.7, +149.9] |
| `learned_minus_oracle` | -130.4 | 0/5 | [-236.7, -24.0] |
| `learned_minus_uniform` | +1930.4 | 5/5 | [+1841.2, +2019.6] |
| `oracle_minus_uniform` | +2060.8 | 5/5 | [+2039.1, +2082.5] |
| `learned_minus_cyclic` | +2113.6 | 5/5 | [+2015.8, +2211.4] |
| `learned_minus_shuffled` | +2282.0 | 5/5 | [+2185.3, +2378.8] |
| `learned_minus_fixed_0` | +1100.8 | 5/5 | [+1005.8, +1195.7] |
| `learned_minus_fixed_1` | +1069.3 | 5/5 | [+968.5, +1170.0] |
| `learned_minus_fixed_2` | +2184.4 | 5/5 | [+2069.3, +2299.5] |
| `learned_minus_fixed_3` | +2245.1 | 5/5 | [+2150.0, +2340.3] |
| `oracle_minus_fixed_0` | +1231.2 | 5/5 | [+1196.5, +1265.8] |
| `oracle_minus_fixed_1` | +1199.6 | 5/5 | [+1064.3, +1335.0] |
| `oracle_minus_fixed_2` | +2314.8 | 5/5 | [+2295.8, +2333.8] |
| `oracle_minus_fixed_3` | +2375.5 | 5/5 | [+2355.6, +2395.4] |

## Mechanism

- Classification: `causal_posterior_realizes_context_value`.
- Oracle context beats uniform and every fixed context with positive intervals: `True`.
- Learned posterior beats uniform and every fixed context with positive intervals: `True`.
- Matching fixed head is stationary-optimal in `4/4` modes.
- Learned posterior accuracy: `0.9984`; median switch delay: `9.4` steps.

The student is genuinely context-dependent, but the prior confirmation still failed superiority to robust 719. Report the mechanism result without reopening model selection.
