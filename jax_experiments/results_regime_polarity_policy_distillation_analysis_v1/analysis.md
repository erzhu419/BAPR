# Polarity policy-distillation screen

This is retrospective algorithm development. Individual robust controllers, not their invalid action ensemble, are the baseline.

## Development teacher

| student seed | student | robust population | delta | best robust delta | teacher recovery | event wins | pass |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1409 | 2096.8 | 1034.4 | +1062.3 | +686.9 | 100.3% | 3/3 | yes |
| 1511 | 2091.0 | 1034.4 | +1056.6 | +681.1 | 99.7% | 3/3 | yes |
| 1601 | 2048.2 | 1034.4 | +1013.8 | +638.3 | 95.7% | 3/3 | yes |

Validation-selected seed: `1511`. Student passes: `3/3`. Group pass: `yes`.

## Final teacher

| student seed | student | robust population | delta | best robust delta | teacher recovery | event wins | pass |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1409 | 2021.3 | 1331.3 | +690.0 | +99.8 | 98.7% | 3/3 | yes |
| 1511 | 1776.3 | 1331.3 | +445.0 | -145.2 | 63.7% | 3/3 | no |
| 1601 | 1918.5 | 1331.3 | +587.2 | -3.0 | 84.0% | 3/3 | yes |

Validation-selected seed: `1601`. Student passes: `2/3`. Group pass: `yes`.

## Combined teacher

| student seed | student | robust population | delta | best robust delta | teacher recovery | event wins | pass |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1409 | 1944.7 | 1182.9 | +761.9 | +23.2 | 116.4% | 3/3 | yes |
| 1511 | 1954.2 | 1182.9 | +771.4 | +32.7 | 117.8% | 3/3 | yes |
| 1601 | 1902.2 | 1182.9 | +719.4 | -19.3 | 109.9% | 3/3 | yes |

Validation-selected seed: `1511`. Student passes: `3/3`. Group pass: `yes`.

## Decision

The combined median teacher can be compressed into one stable causal student. Freeze the validation-selected combined student and run a genuinely new five-seed confirmation; do not select a student from audit return.
