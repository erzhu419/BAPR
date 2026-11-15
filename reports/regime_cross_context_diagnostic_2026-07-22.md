# Event-grouped RegimeSAC cross-context diagnostic

Each event task evaluates the robust checkpoint and all seven oracle-checkpoint contexts in one process. This removes node/runtime differences from paired context and controller comparisons. Independent training seeds (n=5) remain the inferential units.

| Env | Robust | True | Zero | Best fixed | True-robust (95% CI) | True-zero (95% CI) | Diagonal | Fixed spread | Diagnosis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HalfCheetah | 1877.7 | 1317.7 | 391.3 | fixed_0: 1176.7 | -560.1 [-1162.1,+41.9] (-29.8%) | +926.3 [+234.7,+1618.0] | 3/4 | 142.8% | no_reliable_dynamic_headroom |
| Ant | 2838.5 | 3291.4 | 2132.9 | fixed_1: 2524.0 | +452.8 [+357.8,+547.9] (+16.0%) | +1158.5 [+430.7,+1886.2] | 4/4 | 79.3% | dynamic_oracle_headroom_confirmed |
| Walker2d | 2177.4 | 2204.4 | 2125.5 | fixed_2: 2192.6 | +27.0 [-77.4,+131.4] (+1.2%) | +78.9 [+32.9,+124.8] | 2/4 | 17.3% | protocol_pathological_termination |

## HalfCheetah stationary context matrix

| Physics | Robust | True | Zero | Fixed 0 | Fixed 1 | Fixed 2 | Fixed 3 | Cyclic |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1577.1 | 978.1 | 249.7 | 978.1 | 462.6 | 753.9 | 528.3 | 462.6 |
| 1 | 2122.1 | 1668.0 | 546.2 | 1382.9 | 1668.0 | 1044.1 | 1504.9 | 1044.1 |
| 2 | 1775.7 | 908.4 | 311.2 | 967.1 | 597.7 | 908.4 | 605.1 | 605.1 |
| 3 | 2082.9 | 1406.3 | 414.2 | 780.6 | 1196.0 | 743.2 | 1406.3 | 780.6 |

## Ant stationary context matrix

| Physics | Robust | True | Zero | Fixed 0 | Fixed 1 | Fixed 2 | Fixed 3 | Cyclic |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2679.8 | 2960.7 | 1572.0 | 2960.7 | 1518.2 | 1933.3 | 1960.9 | 1518.2 |
| 1 | 3021.1 | 3604.2 | 2730.0 | 1637.8 | 3604.2 | 2506.3 | 2590.6 | 2506.3 |
| 2 | 2895.8 | 3211.8 | 1784.1 | 2211.8 | 2261.4 | 3211.8 | 1902.4 | 1902.4 |
| 3 | 2905.9 | 3706.5 | 2414.0 | 2356.1 | 2421.9 | 1688.4 | 3706.5 | 2356.1 |

## Walker2d stationary context matrix

| Physics | Robust | True | Zero | Fixed 0 | Fixed 1 | Fixed 2 | Fixed 3 | Cyclic |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 239.3 | 199.5 | 239.6 | 199.5 | 210.3 | 256.7 | 211.5 | 210.3 |
| 1 | 193.0 | 207.4 | 229.6 | 202.0 | 207.4 | 177.2 | 184.6 | 177.2 |
| 2 | 245.1 | 214.6 | 231.1 | 188.6 | 172.0 | 214.6 | 213.6 | 213.6 |
| 3 | 211.7 | 193.8 | 239.0 | 199.1 | 224.2 | 191.9 | 193.8 | 199.1 |

## Mechanistic interpretation

- HalfCheetah: true context beats zero by +926.3 [+234.7,+1618.0] and the best fixed context by +141.0 [+53.9,+228.1] (+12.0%), but trails the separately trained robust policy by -560.1 [-1162.1,+41.9]. The mode signal and dynamic specialization are real; the failure is shared conditional-controller negative transfer or base-policy degradation, not estimator ambiguity.
- Ant: true context beats robust by +452.8 [+357.8,+547.9] (+16.0%) and the best fixed context by +767.3 [+536.9,+997.7]. This is a valid adaptation-positive environment.
- Walker2d: robust and true switching termination are both 100%. Controller comparisons are not interpretable until the environment passes a survival gate.


## Decision

Strict dynamic-oracle headroom confirmed in **1/3** environments: Ant.

A learned estimator remains blocked unless true dynamic context beats both the robust checkpoint and every fixed context by at least 10%, with positive paired confidence intervals, and at least 3/4 stationary rows are diagonal-optimal. Near-universal termination is classified as a protocol failure rather than algorithm evidence.
