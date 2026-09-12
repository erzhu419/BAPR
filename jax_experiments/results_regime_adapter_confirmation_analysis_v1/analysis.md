# Regime-adapter multi-seed confirmation

The primary inference excludes development seed 8. Each row first averages five paired event streams within one independently trained policy seed.

| Seed | Role | Robust stat | Identity stat | Gain | Robust switch | Identity switch | Gain | Identity-best fixed switch |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | development | 2200.0 | 2559.3 | +16.3% | 2225.5 | 2580.3 | +15.9% | +107.1 |
| 16 | holdout | 2302.4 | 2468.7 | +7.2% | 2266.7 | 2313.4 | +2.1% | +46.5 |
| 24 | holdout | 1960.8 | 1607.9 | -18.0% | 1949.9 | 1616.0 | -17.1% | +25.1 |
| 32 | holdout | 2425.3 | 1984.8 | -18.2% | 2323.0 | 1838.8 | -20.8% | +312.5 |
| 40 | holdout | 2431.6 | 2284.6 | -6.0% | 2286.3 | 2147.7 | -6.1% | -70.9 |

## Untouched-seed inference

- Stationary: -193.5, 95% CI [-622.5, +235.5], relative -8.5%, wins 1/4.
- Switching: -227.5, 95% CI [-595.3, +140.3], relative -10.3%, wins 1/4.
- Identity minus per-seed best fixed switching: +78.3, 95% CI [-183.0, +339.7], wins 3/4.
- Promotion gate: FAIL.

## Mechanism diagnosis

- Identity versus frozen base, stationary: +299.8, 95% CI [+63.6, +535.9], wins 4/4.
- Identity versus frozen base, switching: +236.1, 95% CI [+95.8, +376.4], wins 4/4.
- Robust continuation versus frozen base, stationary: +493.3; switching: +463.6.
- Correct-mode adapter is stationary-optimal in 13/16 holdout mode rows.
- Adapter bank recovers 60.8% of the shared robust stationary gain and 50.9% of its switching gain.

Adapters improve the frozen pre-fork controller and usually specialize correctly, but four separately optimized heads recover less value than one shared robust continuation. The bottleneck is controller sample/optimization efficiency, not absent mode-conditioned control value.

Decision: `stop_before_learned_router_and_reassess_controller_variance`.

A pass authorizes training a causal router on separate training and calibration streams. It is not yet a learned-adaptation result.
