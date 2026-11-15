# V18 equal-budget mechanism diagnostic

## Result boundary

The frozen BAPR arm passes every preregistered comparison against robust SAC,
recurrent ESCP, and released-B0 RE-SAC. It does not pass the equal-policy-budget
comparison against causal SAC5. BAPR's mean switching return is `2803.1`, versus
`2406.7` for SAC5, but the paired difference `+396.4` has a 95% interval of
`[-485.3,+1278.2]`.

## Oracle and routing decomposition

The causal difference can be decomposed as

`BAPR causal - SAC5 causal = (BAPR oracle - SAC5 oracle) + relative routing effect`.

| Seed | Causal difference | Oracle bank difference | BAPR causal-oracle | SAC5 causal-oracle | Relative routing effect |
|---:|---:|---:|---:|---:|---:|
| 81003 | +520.3 | +712.2 | +38.5 | +230.4 | -191.9 |
| 81021 | +237.2 | -65.7 | -81.6 | -384.5 | +302.9 |
| 81039 | -699.6 | -611.9 | -234.4 | -146.7 | -87.7 |
| 81057 | +701.4 | +849.4 | -125.2 | +22.8 | -148.0 |
| 81079 | +1222.9 | +1297.8 | -132.6 | -57.6 | -75.0 |
| Mean | +396.4 | +436.4 | -107.1 | -67.2 | -39.9 |

The mean advantage comes from the specialist policy bank, not from a routing
advantage over SAC5, which uses the same frozen posterior. The failed strong
claim is therefore primarily a controller-bank reproducibility problem.

## Seed 81039 failure

The seed-81039 BAPR specialists beat their matched robust policy in every
calibration mode, so specialization itself did not collapse. However, the SAC5
pool contains substantially stronger controllers in three of four modes.

| Mode | BAPR matched specialist | Best SAC5 controller | SAC5 return | BAPR - SAC5 |
|---:|---:|---|---:|---:|
| 0 | 1438.9 | sac_replica_1 | 3522.1 | -2083.2 |
| 1 | 3119.9 | sac_replica_3 | 2674.0 | +445.9 |
| 2 | 2383.2 | sac_replica_2 | 3215.8 | -832.6 |
| 3 | 2069.0 | sac_replica_3 | 2459.7 | -390.7 |

The reversal is replicated on all three switching holdouts: BAPR minus SAC5 is
`-481.7`, `-763.4`, and `-853.6`. BAPR routing accuracy remains
`98.80-98.86%`, but causal return is `159-322` below its own oracle per event.
Thus the primary failure is inferior controller coverage, with switch-local
routing regret as a secondary cost. It is not persistent mode ambiguity or a
single adverse event stream.

## Next experiment boundary

Do not tune against the v18 holdout or relax its gate. If a stronger algorithm
claim is required, use new development seeds to test a training-only change
that reduces policy-bank variance while retaining the frozen v5 posterior and
environment. The candidate should target robust specialist quality across all
modes, not another estimator or gate variant. A fresh confirmation must again
compare against SAC5 at equal total policy-training budget.
