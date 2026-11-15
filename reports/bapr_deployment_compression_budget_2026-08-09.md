# BAPR deployment/compression budget audit

## Frozen empirical result

| method | switching mean | std over 5 seeds |
|---|---:|---:|
| bapr | 2015.3 | 32.1 |
| escp_recurrent | 1633.8 | 407.4 |
| resac_b0 | 1568.4 | 316.5 |
| sac | 1222.5 | 772.4 |

BAPR improves over recurrent ESCP by 23.4% and over RE-SAC B0 by 28.5%. Against the registered per-seed strongest corrected comparator, the gain is only 4.9% with 4/5 wins and interval [-298.04649490734727, 485.01184234377496]. Primary superiority gate: **False**.

## Construction budget

| component | environment interactions | gradient updates |
|---|---:|---:|
| 20-controller teacher bank | 112,000,000 | 7,000,000 |
| 6 estimator-source controllers | 33,600,000 | 2,100,000 |
| estimator data and fitting | 144,000 | 3,000 |
| 5 final students | 1,315,000 | 300,000 |
| BAPR construction total | 147,059,000 | 9,403,000 |
| final single-controller baselines | 84,000,000 | 5,250,000 |
| construction plus final baselines | 231,059,000 | 14,653,000 |

One additional student initialization uses 263,000 interactions and 60,000 supervised updates after the teacher bank and estimator exist. This is a deployment replication cost, not an end-to-end sample-efficiency number.

## Deployment footprint

- Robust fallback actor: 74,508 parameters.
- Causal estimator: 732,190 parameters.
- Mode-head student: 76,568 parameters.
- Complete deployed BAPR stack: 883,266 parameters, 3.37 MiB raw float32.
- Twenty actor-only teachers: 5.68 MiB; actor-only compression is 1.69x.
- Versus one SAC actor, BAPR deploys 11.85x as many float32 parameters. Full training checkpoints are not a fair deployment-storage comparator.

## Claim boundary

The supported framing is deployment-time compression and causal adaptation on the frozen HalfCheetah actuator-polarity benchmark. The present evidence does not support end-to-end sample efficiency, universal MuJoCo superiority, or superiority over a per-seed oracle choice of corrected baselines. Mechanism v2 and the untouched persistent-damping headroom screen remain required before expanding the claim.
