# Frozen policy-distillation confirmation

The combined median student seed `1511` was frozen before these five event seeds were evaluated.

| event seed | student | robust population | fixed robust 719 | delta vs population | delta vs fixed |
|---:|---:|---:|---:|---:|---:|
| 100019 | 1892.9 | 1174.7 | 1947.5 | +718.2 | -54.6 |
| 100043 | 1953.9 | 1176.8 | 1912.3 | +777.1 | +41.6 |
| 100069 | 1934.1 | 1187.3 | 1940.8 | +746.9 | -6.6 |
| 100103 | 1899.8 | 1184.4 | 1937.7 | +715.4 | -37.9 |
| 100151 | 1816.4 | 1171.2 | 1927.0 | +645.2 | -110.6 |

## Aggregate

- Student switching return: `1899.4`.
- Robust-population return: `1178.9`; delta `+720.5`, event wins `5/5`, cluster 95% t interval `[+659.7, +781.4]`.
- Frozen strongest robust return: `1933.0`; delta `-33.6`, event wins `1/5`, cluster 95% t interval `[-103.8, +36.6]`.
- Learned-teacher headroom recovery: `88.7%`.
- Student switching termination rate: `0.000`.

## Decision

Overall confirmation: `fail`.

The frozen student confirms a robust-population improvement but not superiority to the preregistered strongest robust controller. Retain this as an ensemble-distillation result, not a stronger BAPR control claim.
