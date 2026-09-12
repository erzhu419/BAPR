# Final BAPR mechanism audit

This is a frozen, checkpoint-only post-confirmation diagnostic. It cannot select another model, estimator, fallback threshold, or environment configuration.

| student | robust | learned | fallback | true | context | causal | fallback value | delay 5 | delay 10 |
|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
| 2009 | 1913.1 | 1914.9 | 1956.4 | 2040.5 | True | True | True | False | False |
| 2113 | 1913.1 | 1915.6 | 2057.2 | 2059.3 | True | True | True | True | False |
| 2213 | 1913.1 | 1967.8 | 2048.2 | 2084.5 | True | True | True | True | False |
| 2311 | 1913.1 | 1941.8 | 2051.6 | 2079.4 | True | True | True | True | False |
| 2417 | 1913.1 | 1885.2 | 1987.1 | 2028.8 | True | True | True | False | False |

## Frozen decision

- Context-specialized students: `5/5`.
- Students with causal fallback value over robust 719: `5/5`.
- Students with incremental fallback value over learned-only: `5/5`.
- Overall mechanism pass: `True`.

The final result is supported by repeatable dynamic context specialization and a causal deployable path, rather than by a single unconditional compressed policy. The fallback ablation is reported separately and is not required to be beneficial for every initialization.
