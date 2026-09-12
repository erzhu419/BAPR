# Polarity policy-ensemble diagnostic

This is a retrospective, checkpoint-only controller-variance diagnostic. It is not a new confirmation experiment.

## Development controller group

| arm | switching mean | delta vs matched robust | relative | event wins | action disagreement |
|---|---:|---:|---:|---:|---:|
| `oracle_mean` | 1765.4 | +1538.0 | +676.5% | 3/3 | 0.488 |
| `learned_mean` | 1645.2 | +1417.8 | +623.6% | 3/3 | 0.489 |
| `oracle_median` | 2175.5 | +1767.6 | +433.4% | 3/3 | 0.507 |
| `learned_median` | 2062.3 | +1654.4 | +405.6% | 3/3 | 0.508 |

| training seed | robust | oracle | delta |
|---:|---:|---:|---:|
| 101 | 735.2 | 2268.7 | +1533.5 |
| 211 | 871.0 | 2725.3 | +1854.3 |
| 307 | 1386.7 | 2369.4 | +982.6 |
| 419 | 1400.3 | 2648.1 | +1247.8 |
| 523 | 795.8 | 1596.6 | +800.8 |

Individual oracle wins: `5/5`.

Individual robust switching baseline: mean `1037.8`, best `1400.3`, worst `735.2`.

## Final controller group

| arm | switching mean | delta vs matched robust | relative | event wins | action disagreement |
|---|---:|---:|---:|---:|---:|
| `oracle_mean` | 1999.4 | +2033.6 | +2033.6% | 3/3 | 0.431 |
| `learned_mean` | 1897.7 | +1932.0 | +1932.0% | 3/3 | 0.433 |
| `oracle_median` | 2087.6 | +2180.7 | +2180.7% | 3/3 | 0.448 |
| `learned_median` | 1970.9 | +2064.0 | +2064.0% | 3/3 | 0.454 |

| training seed | robust | oracle | delta |
|---:|---:|---:|---:|
| 607 | 955.7 | 2833.6 | +1877.8 |
| 719 | 1913.9 | 1976.8 | +62.9 |
| 823 | 1172.1 | 2744.4 | +1572.2 |
| 929 | 1000.7 | 2378.4 | +1377.7 |
| 1031 | 1649.3 | 1462.9 | -186.4 |

Individual oracle wins: `4/5`.

Individual robust switching baseline: mean `1338.4`, best `1913.9`, worst `955.7`.

## Decision

Both action reductions pass the preregistered aggregate gate, but the matched robust action ensembles collapse and are not a credible strong baseline. Freeze the stronger diagnostic teacher (median), distill one posterior-conditioned student, and compare that student against the individual robust-controller distribution on new event seeds.

No GPU training or untouched-seed confirmation is launched by this diagnostic.
