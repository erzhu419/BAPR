# Fresh ten-seed BAPR deployment confirmation

The old five model seeds are excluded. All four methods use the same five untouched event streams and strict deterministic evaluation.

| method | switching mean +/- sd | stationary mean +/- sd | switch termination |
|---|---:|---:|---:|
| bapr | 1981.5 +/- 28.5 | 2100.0 +/- 24.4 | 0.000 |
| sac | 1021.1 +/- 639.8 | 960.5 +/- 649.0 | 0.000 |
| escp_recurrent | 1849.1 +/- 383.1 | 2022.4 +/- 457.2 | 0.000 |
| resac_b0 | 1633.1 +/- 550.8 | 1685.5 +/- 460.5 | 0.000 |

## Registered comparisons

| baseline | delta | relative | wins | Holm p | simultaneous lower | stationary retention | pass |
|---|---:|---:|---:|---:|---:|---:|:---:|
| sac | +960.4 | +94.1% | 9/10 | 0.00165 | +449.0 | 218.6% | True |
| escp_recurrent | +132.4 | +7.2% | 7/10 | 0.1513 | -171.4 | 103.8% | False |
| resac_b0 | +348.4 | +21.3% | 8/10 | 0.07218 | -80.9 | 124.6% | False |

## Decision

- Overall registered pass: `False`.
- BAPR minus per-seed strongest envelope (diagnostic only): -83.6, 5/10 wins.
- Scope: deployment performance of the frozen BAPR teacher-estimator-student pipeline on the HalfCheetah actuator-polarity protocol; not sample efficiency, end-to-end training-budget parity, or universal nonstationary-RL superiority.
