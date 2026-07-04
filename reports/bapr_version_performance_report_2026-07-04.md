# BAPR version and performance report, 2026-07-04

This report summarizes the BAPR redesign experiments currently available in
`jax_experiments/results_paper`. It is written for follow-up analysis by another
AI or reviewer. The main warning is that not every run with a clean exit code is
equally trustworthy: several node007 runs used an older staged code path and
should not be used as final evidence for the latest controller logic.

## Executive summary

- `redesign_gate_v45_v36_regemerg02_latch70_recentfloor03_cap015` remains the
  best completed multi-environment redesign run: 4 MuJoCo environments, 3 seeds.
- `v73c_ant_latch_reg0_recent12` was the best completed Ant-only v73 variant
  before the new weight-reg/actor-throttle tests.
- `v75r_c_wreg01_ant_reg0_recent12_actor025` is the strongest new Ant seed0
  result: `tail20 = 3001.6`, `best = 5592.3`, clean latest-controller logging.
- `weight_reg=0.01` is promising only when paired with `recent12` and
  `actor_update_multiplier=0.25`. The `actor_update_multiplier=0.0` variants
  show critic/pathology symptoms and lower final performance.
- Do not use the node007 `v74c2`, `v74d2`, `v75r_a`, or `v75r_b` runs as final
  latest-controller evidence. They have `launcher_exit.status` return code 0 and
  complete metric arrays, but their logged `BAPR config` lacks the latest
  controller fields and scheduler marked them failed.

## Version/configuration notes

| Version | Scope | Key configuration | Purpose | Validity |
|---|---|---|---|---|
| `v45` | 4 envs x 3 seeds | `actor_objective=mean`, `weight_reg=0.003`, `beta_ood=0.003`, recent floor `0.03`, reg latch/emergency | Best completed redesign baseline | Valid completed reference |
| `v72a_controller_latch` | 4 envs x seed0 | Recovery controller latch, no actor throttle | Stabilize collapse recovery | Valid, but weaker than v73/v75 on Ant |
| `v73a` | Ant seed0 | `reg_multiplier=0`, recent08 | Test no post-collapse reg | Valid |
| `v73b` | Ant seed0 | `controller_reg_multiplier=0.1`, recent08 | Test adding reg after collapse | Valid; hurts tail performance |
| `v73c` | Ant seed0 | `reg_multiplier=0`, recent12 | Stronger recent replay under latch | Valid; best v73 |
| `v73d` | Ant seed0 | `controller_reg_multiplier=0.1`, recent12 | Reg after collapse with stronger recent replay | Valid; hurts tail performance |
| `v74a` | Ant seed0 | `weight_reg=0.003`, recent08, `actor_update_multiplier=0.25` | Actor throttle test | Valid |
| `v74b` | Ant seed0 | `weight_reg=0.003`, recent08, `actor_update_multiplier=0.0` | Freeze actor after controller | Valid; critic blows up |
| `v74c2` | Ant seed0 | Intended: recent12, actor `0.25` | Stronger recent + throttle | Not final: node007 stale-code run |
| `v74d2` | Ant seed0 | Intended: recent12, actor `0.0` | Stronger recent + actor freeze | Not final: node007 stale-code run |
| `v75r_a` | Ant seed0 | Intended: `weight_reg=0.01`, recent08, actor `0.25` | RE-SAC-style reg test | Not final: node007 stale-code run |
| `v75r_b` | Ant seed0 | Intended: `weight_reg=0.01`, recent08, actor `0.0` | RE-SAC-style reg + freeze | Not final: node007 stale-code run |
| `v75r_c` | Ant seed0 | `weight_reg=0.01`, recent12, actor `0.25` | Best current candidate | Valid |
| `v75r_d` | Ant seed0 | `weight_reg=0.01`, recent12, actor `0.0` | Actor freeze with `0.01` reg | Valid but unstable/low |

## Completed multi-environment v45 reference

Metrics are mean of the last 20 eval points (`tail20`) across seeds.

| Environment | v45 tail20 mean +- std | Seeds | Seed tail20 values |
|---|---:|---|---|
| HalfCheetah | `18124.2 +- 1189.6` | 0,1,2 | 17672.1, 17226.9, 19473.5 |
| Ant | `1776.5 +- 623.5` | 0,1,2 | 2421.5, 1177.1, 1730.8 |
| Hopper | `1909.0 +- 858.6` | 0,1,2 | 2844.8, 1157.5, 1724.6 |
| Walker2d | `1607.5 +- 1161.4` | 0,1,2 | 2889.4, 625.5, 1307.6 |

Interpretation: v45 is the strongest completed redesign reference, but it is not
enough for a strong "BAPR beats SAC/ESCP/RE-SAC across benchmarks" claim. It is
strong on HalfCheetah, mixed on Ant, and high-variance on Hopper/Walker2d.

## Recent Ant seed0 results

| Label | Clean latest-controller run? | n eval | tail20 | std20 | last | best |
|---|---:|---:|---:|---:|---:|---:|
| `v73a` | yes | 160 | 1259.2 | 503.7 | 1468.8 | 5087.0 |
| `v73b` | yes | 160 | 72.1 | 1457.9 | 279.3 | 6487.0 |
| `v73c` | yes | 160 | 1467.2 | 737.5 | 1508.2 | 10170.5 |
| `v73d` | yes | 160 | 106.0 | 1475.2 | 1704.9 | 8991.2 |
| `v74a` | yes | 160 | 1313.5 | 534.8 | 749.3 | 4798.3 |
| `v74b` | yes | 160 | 545.8 | 151.6 | 450.6 | 4513.1 |
| `v74c2` | no, node007 stale-code | 160 | -3245.6 | 1511.9 | -3798.2 | 5633.8 |
| `v74d2` | no, node007 stale-code | 160 | -3612.6 | 2565.2 | -1224.1 | 5326.6 |
| `v75r_a` | no, node007 stale-code | 160 | -3946.9 | 2785.4 | -1923.4 | 8668.9 |
| `v75r_b` | no, node007 stale-code | 160 | -4791.9 | 2056.6 | -2293.9 | 4884.8 |
| `v75r_c` | yes | 160 | 3001.6 | 1184.4 | 3149.1 | 5592.3 |
| `v75r_d` | yes | 160 | 1353.3 | 410.0 | 754.9 | 4567.8 |

## Mechanistic observations

1. Post-collapse regularization is harmful in v73.
   - `v73a/v73c` (`controller_reg_multiplier=0`) clearly beat `v73b/v73d`
     (`controller_reg_multiplier=0.1`) in final tail performance.
   - The earlier `reg01` names in v73 refer to controller reg multiplier `0.1`,
     not to base `weight_reg=0.01`.

2. `weight_reg=0.01` is not automatically good.
   - Valid `v75r_c` is the best current Ant candidate.
   - Valid `v75r_d` is much weaker, suggesting actor update throttle `0.25`
     matters.
   - The best candidate is specifically:
     `weight_reg=0.01`, recent multiplier `3.0`, recent add frac `0.03`,
     actor update multiplier `0.25`.

3. Actor freezing is risky.
   - `v74b` and `v75r_d` have very large negative `q_mean` in the final window
     (around `-1.55e11`), consistent with critic/pathology rather than healthy
     convergence.
   - Prefer actor update scaling (`0.25`) over full actor freeze (`0.0`).

4. node007 staging must be treated as suspect.
   - `v74c2`, `v74d2`, `v75r_a`, and `v75r_b` logged environment variables for
     the intended controller, but their printed `BAPR config` lacks the latest
     controller fields (`controller_mode`, `controller_actor_update_multiplier`,
     etc.).
   - Those runs also have scheduler status `failed` despite launcher return code
     `0`. The metric arrays are complete, but they are not comparable to local
     or jtl latest-controller runs.

## Baseline comparison caveat

Against older paper baselines, v45 is strong on HalfCheetah, but not enough for a
strong main algorithm claim:

- HalfCheetah: v45 is higher than SAC/ESCP/RE-SAC in available `paper_full_v2`
  and `paper_full_v3gfix` comparisons, though significance against RE-SAC is
  weak because baseline variance is large.
- Ant: v45 loses badly to SAC in both available full baseline sets and does not
  cleanly beat ESCP/RE-SAC.
- Hopper/Walker2d: v45 is above `paper_negative_v2` baselines in mean, but with
  only 3 seeds and high variance.

Conclusion: v45 is a valid reference/ablation anchor. It is not sufficient by
itself as the final main paper result unless the paper is reframed as an
analysis of when adaptation helps and fails.

## Recommended next analysis

1. Treat `v75r_c` as the current best Ant candidate, not v45 or v73.
2. Rerun `v75r_c` cleanly on trusted latest-code nodes for seeds 1-4.
3. Rerun the `v75r_c` configuration on HalfCheetah, Hopper, Walker2d before
   promoting it to a paper main version.
4. Fix or audit node007 code staging before using node007 for BAPR conclusions.
5. Keep bus-environment validation separate: no MuJoCo-only result is sufficient
   to claim BAPR preserves the required bus effectiveness.
