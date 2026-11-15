# Polarity context-causality and delay audit

This is a checkpoint-only paired audit on the existing actuator-polarity controllers. Each training seed is evaluated on three identical event streams under true, zero, fixed, cyclic, shuffled, and delayed contexts. The three existing training seeds are exploratory units; they are not extended post hoc.

| Env | Robust | True | Per-seed fixed envelope | True-robust | True-fixed envelope | Diagonal | Delay 10 retained | Delay 25 retained | Candidate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HalfCheetah | 738.8 | 2633.9 | 1394.5 | +1895.1 [-359.2, +4149.5] (+256.5%) | +1239.4 [+848.1, +1630.8] | 4/4 | 87.9% | 64.0% | PASS |
| Ant | 1244.8 | 2304.5 | 356.5 | +1059.7 [-695.6, +2815.0] (+85.1%) | +1948.0 [+1046.2, +2849.8] | 4/4 | 59.8% | 15.2% | FAIL |

## Delayed-oracle ladder

| Env | Delay | Return | Delta vs robust | Headroom retained | Beats robust all 3 seeds |
|---|---:|---:|---:|---:|---|
| HalfCheetah | 1 | 2631.9 | +1893.1 [-468.7, +4255.0] | 99.2% | True |
| HalfCheetah | 5 | 2538.2 | +1799.5 [-413.8, +4012.7] | 94.6% | True |
| HalfCheetah | 10 | 2437.8 | +1699.0 [-625.8, +4023.8] | 87.9% | True |
| HalfCheetah | 25 | 1900.2 | +1161.5 [+169.6, +2153.3] | 64.0% | True |
| HalfCheetah | 50 | 1638.0 | +899.2 [+250.4, +1548.0] | 50.2% | True |
| Ant | 1 | 2273.2 | +1028.4 [-665.7, +2722.4] | 97.3% | True |
| Ant | 5 | 1894.2 | +649.4 [-164.0, +1462.8] | 66.2% | True |
| Ant | 10 | 1897.5 | +652.7 [-433.0, +1738.4] | 59.8% | True |
| Ant | 25 | 1588.9 | +344.0 [-892.5, +1580.6] | 15.2% | False |
| Ant | 50 | 1375.8 | +131.0 [-1024.1, +1286.0] | -10.9% | False |

## Decision

The continuation gate requires all of the following within an environment: true context beats the robust controller on every exploratory seed with at least 10% mean gain; true context beats the per-seed best fixed-context envelope on every seed; at least 3/4 stationary rows are diagonal-optimal; delay 10 and delay 25 retain at least 70% and 50% of instantaneous-oracle headroom while beating robust on every seed; and termination is not pathological.

Fresh five-seed confirmation candidates: HalfCheetah.

A learned estimator remains blocked at this stage. Passing this audit authorizes a new preregistered robust-versus-oracle five-seed confirmation, not estimator training.
