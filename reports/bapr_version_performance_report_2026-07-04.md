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
  result: `tail20 = 3001.6`, `best = 5592.3`, clean latest-controller logging,
  but the clean v76 validation rerun shows that this was not robust across
  seeds.
- `v76_clean_v75rc_wreg01_recent12_actor025` should not replace v45 as the main
  result. Ant 5-seed tail20 drops to `713.0 +- 1633.7`, and the single-seed
  HalfCheetah/Hopper/Walker2d validation runs are all below v45 seed0.
- `weight_reg=0.01` is promising only when paired with `recent12` and
  `actor_update_multiplier=0.25`, but the benefit is not enough by itself. The
  `actor_update_multiplier=0.0` variants show critic/pathology symptoms and
  lower final performance.
- The main v76 failure mode is not checkpointing or scheduler noise. On Ant, the
  recovery controller stays active for the whole tail window, actor updates stay
  throttled to `0.25`, controller regularization stays off, and the policy
  undertrains or drifts. This makes the recovery mode too permanent.
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
| `v75r_c` | Ant seed0 | `weight_reg=0.01`, recent12, actor `0.25` | Strong seed0 candidate before v76 validation | Valid run, but not robust across seeds |
| `v75r_d` | Ant seed0 | `weight_reg=0.01`, recent12, actor `0.0` | Actor freeze with `0.01` reg | Valid but unstable/low |
| `v76` | Ant seeds 1-4, HC/Hopper/Walker seed0 | Same as valid `v75r_c`: `weight_reg=0.01`, recent12, actor `0.25`, qstd-gated LCB controller | Clean validation of `v75r_c` beyond Ant seed0 | Valid latest-controller run; rejects promoting `v75r_c` |

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

## Clean v75r_c/v76 validation rerun

The clean validation rerun used the valid `v75r_c` configuration:
`weight_reg=0.01`, `beta_ood=0.003`, recent replay floor `0.03`, recent
multiplier `3.0`, recent add frac `0.03`, qstd-gated LCB actor objective,
controller recovery latch, `controller_actor_update_multiplier=0.25`, and
`controller_reg_multiplier=0.0`. Ant seed0 is the original valid `v75r_c`; Ant
seeds 1-4 and the other environments are the clean `v76` reruns.

| Environment | Seeds | tail20 mean +- sample std | Seed tail20 values | tail50 mean +- sample std | Interpretation |
|---|---:|---:|---|---:|---|
| Ant | 0-4 | `713.0 +- 1633.7` | 3001.6, 654.7, -1238.9, 1464.5, -316.7 | `925.5 +- 558.5` | Ant seed0 was not representative; multi-seed result is below v45 Ant (`1776.5 +- 623.5`) |
| HalfCheetah | 0 | `15172.2` | 15172.2 | `15381.6` | Below v45 seed0 (`17672.1`) and v45 mean (`18124.2 +- 1189.6`) |
| Hopper | 0 | `1799.5` | 1799.5 | `2405.5` | Below v45 seed0 (`2844.8`) |
| Walker2d | 0 | `1684.2` | 1684.2 | `2574.7` | Below v45 seed0 (`2889.4`) |

Ant per-seed diagnostics show why this setting is fragile:

| Seed | tail20 | tail50 | last | best | controller active tail20 | actor update multiplier tail20 | reg multiplier tail20 | recent replay tail20 | q_mean tail20 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3001.6 | 1767.9 | 3149.1 | 5592.3 | 1.00 | 0.25 | 0.00 | 0.124 | 195.4 |
| 1 | 654.7 | 646.3 | 516.1 | 4800.1 | 1.00 | 0.25 | 0.00 | 0.141 | 242.4 |
| 2 | -1238.9 | 367.9 | 570.0 | 4999.8 | 1.00 | 0.25 | 0.00 | 0.120 | 721.7 |
| 3 | 1464.5 | 1195.7 | 869.6 | 4677.4 | 1.00 | 0.25 | 0.00 | 0.137 | 227.6 |
| 4 | -316.7 | 649.5 | 76.6 | 5107.4 | 1.00 | 0.25 | 0.00 | 0.120 | 1567.3 |

Conclusion: v76 does not fail because the result is incomplete or because the
wrong code ran. It fails because the controller behavior is too blunt: once Ant
enters recovery, the latch keeps the policy in low-actor-update/no-reg mode for
the tail of training. That can rescue one seed, but it does not form a robust
algorithmic improvement.

## Mechanistic observations

1. Post-collapse regularization is harmful in v73.
   - `v73a/v73c` (`controller_reg_multiplier=0`) clearly beat `v73b/v73d`
     (`controller_reg_multiplier=0.1`) in final tail performance.
   - The earlier `reg01` names in v73 refer to controller reg multiplier `0.1`,
     not to base `weight_reg=0.01`.

2. `weight_reg=0.01` is not automatically good.
   - Valid `v75r_c` is the best Ant seed0 result in this group, but v76 shows
     that it does not generalize across Ant seeds.
   - Valid `v75r_d` is much weaker, suggesting actor update throttle `0.25`
     matters.
   - The best candidate is specifically:
     `weight_reg=0.01`, recent multiplier `3.0`, recent add frac `0.03`,
     actor update multiplier `0.25`; however, this is only a useful component
     hypothesis, not a promotable full algorithm.

3. Actor freezing is risky.
   - `v74b` and `v75r_d` have very large negative `q_mean` in the final window
     (around `-1.55e11`), consistent with critic/pathology rather than healthy
     convergence.
   - Prefer actor update scaling (`0.25`) over full actor freeze (`0.0`).

4. Permanent recovery latch is too strong in v76.
   - The clean v76 Ant seeds all have `controller_active` tail mean `1.0`.
   - Because actor updates stay at `0.25` and controller regularization stays
     at `0.0`, recovery becomes a long-term training regime rather than a short
     intervention.
   - Future variants should test release/decay behavior: either no permanent
     latch, a stricter latch release, or gradual restoration of actor updates
     and regularization after the immediate collapse window.

5. Old node007 staging runs must be treated as suspect.
   - `v74c2`, `v74d2`, `v75r_a`, and `v75r_b` logged environment variables for
     the intended controller, but their printed `BAPR config` lacks the latest
     controller fields (`controller_mode`, `controller_actor_update_multiplier`,
     etc.).
   - Those runs also have scheduler status `failed` despite launcher return code
     `0`. The metric arrays are complete, but they are not comparable to local
     or jtl latest-controller runs.
   - Scheduler staging for node007 was later patched to use a compact code-tar
     launch staging path. That fix does not rehabilitate the old stale-code
     runs; future node007 evidence still needs a real latest-code smoke or a
     fresh clean training run.

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

Conclusion: v45 is still the best valid reference/ablation anchor. v75r_c/v76
does not supersede it. Even v45 is not sufficient by itself as the final main
paper result unless the paper is reframed as an analysis of when adaptation
helps and fails.

## Recommended next analysis

1. Do not promote `v75r_c` or `v76` as the main paper version.
2. Keep v45 as the strongest completed reference, while being explicit that it
   is not yet a top-paper-strength algorithmic result.
3. The next algorithmic direction should weaken or release the recovery
   controller: no permanent latch, stricter release, or gradual restoration of
   actor updates and regularization after collapse.
4. For future node007 use, confirm the patched code-tar staging with a real
   latest-code smoke or fresh clean run; do not use the old stale-code node007
   runs as evidence.
5. Keep bus-environment validation separate: no MuJoCo-only result is sufficient
   to claim BAPR preserves the required bus effectiveness.
