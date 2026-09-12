# Transient delayed-oracle causal-ceiling diagnostic

| student | robust | fallback | delay 1 | delay 2 | delay 5 | delay 10 | oracle | fallback causal recovery | delay1 - fallback | reducible |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1709 | 1913.9 | 2032.3 | 2064.0 | 2042.8 | 1969.0 | 1742.5 | 2085.4 | 78.9% | +31.7 (4/5) | False |
| 1811 | 1913.9 | 2005.8 | 2041.3 | 2039.1 | 1951.0 | 1690.2 | 2058.4 | 72.2% | +35.5 (5/5) | False |
| 1901 | 1913.9 | 2003.1 | 2051.6 | 2035.6 | 1975.2 | 1649.3 | 2053.7 | 64.8% | +48.5 (5/5) | True |

## Decision

- Students with material reducible gaps: `1/3`.
- Authorize transient training: `False`.

The frozen fallback is already close to the one-transition causal ceiling, or the remaining gap is not stable across initializations. Do not train another student; freeze fallback as the deployable mechanism and report the zero-delay oracle as noncausal headroom.
