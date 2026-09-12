# Closed-loop policy-compression v2 development result

This is development-only. The five sealed confirmation events were not reused. Individual robust controllers, including the fixed `robust_final_seed_719`, are the valid baselines.

## wide_dagger

| student seed | student | robust population | population delta | fixed-719 delta | teacher gap | recovery | event wins vs 719 | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1709 | 1942.0 | 1174.3 | +767.7 | +33.5 | +14.7 | 102.0% | 2/3 | yes |
| 1811 | 1788.0 | 1174.3 | +613.7 | -120.4 | -139.3 | 81.5% | 1/3 | no |
| 1901 | 1958.4 | 1174.3 | +784.1 | +50.0 | +31.2 | 104.1% | 3/3 | yes |

Control-validation-selected seed: `1811`. Student passes: `2/3`. Variant pass: `no`.

## mode_heads

| student seed | student | robust population | population delta | fixed-719 delta | teacher gap | recovery | event wins vs 719 | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1709 | 1952.5 | 1174.3 | +778.2 | +44.1 | +25.2 | 103.4% | 3/3 | yes |
| 1811 | 1932.7 | 1174.3 | +758.4 | +24.3 | +5.5 | 100.7% | 2/3 | yes |
| 1901 | 1932.3 | 1174.3 | +758.0 | +23.8 | +5.0 | 100.7% | 3/3 | yes |

Control-validation-selected seed: `1811`. Student passes: `3/3`. Variant pass: `yes`.

## mode_heads_return

| student seed | student | robust population | population delta | fixed-719 delta | teacher gap | recovery | event wins vs 719 | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1709 | 1918.0 | 1174.3 | +743.7 | +9.6 | -9.2 | 98.8% | 2/3 | yes |
| 1811 | 1897.7 | 1174.3 | +723.4 | -10.7 | -29.5 | 96.1% | 1/3 | no |
| 1901 | 1867.4 | 1174.3 | +693.1 | -41.0 | -59.9 | 92.1% | 2/3 | no |

Control-validation-selected seed: `1901`. Student passes: `1/3`. Variant pass: `no`.

## Decision

Freeze mode_heads with its control-validation-selected student and register a new untouched five-event confirmation. Do not reuse the sealed 100019-100151 confirmation events or reselect from this development audit.
