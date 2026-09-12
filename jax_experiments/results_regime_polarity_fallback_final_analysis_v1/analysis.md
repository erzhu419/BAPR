# Causal-fallback BAPR final comparison

| method | switching | stationary | switching termination |
|---|---:|---:|---:|
| bapr | 2015.3 +/- 32.1 | 2118.7 +/- 14.5 | 0.000 |
| sac | 1222.5 +/- 772.4 | 1257.6 +/- 766.4 | 0.000 |
| escp | 1858.5 +/- 710.6 | 1912.5 +/- 723.1 | 0.000 |
| resac | -375.9 +/- 69.2 | -364.2 +/- 54.8 | 0.000 |

## Frozen decision

- BAPR minus strongest baseline: -118.9
- Conservative 95% interval: [-817.3,+579.5]
- Registered seed-slot wins: 2/5
- Stationary retention: 95.5%
- Termination gap: +0.000
- Primary pass: `False`
- RE-SAC reproduction warning: `True`

deployment-performance comparison; BAPR uses a frozen ten-controller teacher and is not sample-efficiency matched to the single-controller baselines
