# Frozen policy-distillation confirmation

The `mode_heads` student seed `1811` was frozen before these five event seeds were evaluated.

| event seed | student | robust population | fixed robust 719 | delta vs population | delta vs fixed |
|---:|---:|---:|---:|---:|---:|
| 102019 | 1884.2 | 1188.2 | 1940.6 | +696.0 | -56.4 |
| 102043 | 1987.5 | 1167.7 | 1983.2 | +819.8 | +4.3 |
| 102069 | 1996.5 | 1187.5 | 1939.4 | +809.1 | +57.1 |
| 102103 | 1977.8 | 1191.3 | 1904.6 | +786.5 | +73.1 |
| 102151 | 1969.2 | 1150.5 | 1937.6 | +818.8 | +31.7 |

## Aggregate

- Student switching return: `1963.1`.
- Robust-population return: `1177.0`; delta `+786.0`, event wins `5/5`, cluster 95% t interval `[+721.4, +850.7]`.
- Frozen strongest robust return: `1941.1`; delta `+22.0`, event wins `4/5`, cluster 95% t interval `[-41.3, +85.3]`.
- Learned-teacher headroom recovery: `116.9%`.
- Student switching termination rate: `0.000`.

## Decision

Overall confirmation: `fail`.

The frozen student confirms a robust-population improvement but not superiority to the preregistered strongest robust controller. Retain this as an ensemble-distillation result, not a stronger BAPR control claim.
