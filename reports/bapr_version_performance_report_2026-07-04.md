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
- `v78` is the first soft-release controller screen and is now complete
  (`36/36`). The best Ant variant remains `v78c_soft_actorlight` with 5-seed
  tail20 `2010.6 +- 605.9`, slightly above v45 Ant (`1776.5 +- 623.5`), but
  still below the available SAC/ESCP/RE-SAC Ant baselines. Its HalfCheetah
  canary is weak (`10010.0`) and below v45, ESCP, and RE-SAC, so v78 should not
  be promoted as a broad algorithmic win.
- `v79` tested explicit controller exit/cooldown. The best Ant variant is
  `v79b_exit120_regsafe` with 5-seed tail20 `2220.5 +- 887.5`, slightly above
  v78c and v45, but still below ESCP/RE-SAC/SAC Ant baselines. The best
  HalfCheetah canary is only `14994.3`, below v45 and below ESCP/RE-SAC.
  Controller exit works mechanistically in some seeds (`v79c`), but the
  performance tradeoff is not favorable enough to promote v79.
- `v80` tested a conservative residual-policy objective. All 24 runs completed.
  It should not replace v45/v78/v79: the best Ant variant is only
  `1760.4 +- 897.0`, the best HalfCheetah canary is `14732.6`, and the
  residual gate is almost constant rather than meaningfully adaptive.
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
| `v78` | 6 Ant variants x 5 seeds, plus HC/Hopper/Walker canaries for v78a/v78c | Soft recovery signal with decay; v45 base reg/recent settings | Replace permanent latch with softer controller multipliers | Valid complete screen; Ant improves modestly, cross-env story fails |
| `v79` | 3 variants x Ant 0-4 plus HC/Hopper/Walker canaries | v78-like soft controller plus explicit exit/cooldown conditions | Test whether recovery can deactivate instead of staying active in the tail | Valid complete screen after clean rerun of v79a Ant seed2; not promotable |

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

## v78 soft-release controller screen

The v78 screen changed the hard recovery latch into a soft recovery signal with
decay, while keeping the v45 base setting (`weight_reg=0.003`,
`beta_ood=0.003`, recent floor `0.03`, reg latch/emergency). The batch completed
all 36 scheduler tasks: 6 Ant variants x 5 seeds plus two single-seed canary
variants on HalfCheetah/Hopper/Walker2d.

Ant tail20 summary:

| Variant | Seeds done | Tail20 mean +- sample std | Seed tail20 values | Tail50 mean +- sample std | Mechanistic read |
|---|---:|---:|---|---:|---|
| `v78a_soft_mild` | 0-4 | `1671.5 +- 672.7` | 1241.8, 914.1, 2689.9, 1741.1, 1770.6 | `1747.5 +- 689.5` | Below v45 mean; controller tail-active in all seeds |
| `v78b_soft_regsafe` | 0-4 | `1614.3 +- 649.9` | 1185.0, 1071.9, 2110.8, 1199.4, 2504.5 | `1582.8 +- 531.0` | More post-recovery reg does not help |
| `v78c_soft_actorlight` | 0-4 | `2010.6 +- 605.9` | 1366.6, 1795.7, 1596.5, 2532.9, 2761.1 | `2106.8 +- 669.0` | Best Ant result so far; only modestly above v45, and seed0 has Q-std pathology |
| `v78d_soft_recent02` | 0-4 | `1471.3 +- 801.4` | 1765.0, 865.5, 402.7, 2061.6, 2262.0 | `1562.8 +- 806.2` | Recent add hurts robustness; seed2 collapses low |
| `v78e_soft_broad` | 0-4 | `1798.8 +- 764.8` | 2760.7, 1247.8, 1352.2, 2491.5, 1141.9 | `1743.1 +- 702.7` | Slightly above v45 mean but high variance and weaker than v78c |
| `v78f_soft_wreg001` | 0-4 | `1243.4 +- 978.3` | 1675.0, 1130.0, 1700.1, 2105.1, -393.0 | `1554.7 +- 430.0` | `weight_reg=0.01` again hurts robustness |

Canary tail20, single seed:

| Variant | HalfCheetah | Hopper | Walker2d | Read |
|---|---:|---:|---:|---|
| `v78a_soft_mild` | 12557.7 | 3975.1 | 1195.4 | Hopper improves over v45 seed0, but HalfCheetah and Walker are not competitive |
| `v78c_soft_actorlight` | 10010.0 | 2043.1 | 1967.1 | Ant-best variant does not transfer to HalfCheetah; Walker is above v45 mean but below v45 seed0; Hopper is not better than v45 seed0 |

Interpretation: v78 is a real improvement over the failed v76 recovery latch on
Ant: `v78c` gives the first clean 5-seed Ant result above v45. But the effect is
too narrow to justify promoting v78 or launching a second broad v78-style run.
The controller remains active for most Ant tail windows (`controller_active`
tail mean is usually `1.0`), so the "soft release" mostly reduces throttle
severity rather than cleanly returning to normal training. The seed0 `v78c` Ant
run also has a very high tail Q-std (`~171`) and negative final return, meaning
the mean improvement is not a clean stability story.

Against the available paper baselines, v78 is not enough for a main algorithm
claim. On Ant, `v78c` (`2010.6 +- 605.9`) is below both gravity-fixed ESCP
(`2771.2 +- 899.8`) and RE-SAC (`2818.4 +- 1093.0`), and far below SAC
(`8917.1 +- 4687.8`). On HalfCheetah, the best v78 canary is `v78a`
(`12557.7`), below v45 (`18124.2 +- 1189.6`) and below the gravity-fixed
ESCP/RE-SAC means (`16242.5 +- 5869.9` and `16167.8 +- 5300.4`). Hopper/Walker
canaries can beat the weak negative-v2 baselines, but they do not fix the main
Ant/HalfCheetah story. The next useful direction is exactly what v79 starts to
test: explicit controller exit/cooldown rather than only softer multipliers.

## v79 explicit exit/cooldown controller screen

The v79 screen added explicit controller exit and cooldown fields on top of the
v78-style soft controller. It tested 3 variants: 5 Ant seeds plus
single-seed HalfCheetah/Hopper/Walker2d canaries. The original `v79a` Ant seed2
run reached only iter 400, then repeated node007 retries failed because the
checkpoint was cross-version fragile and its `replay_buffer.npz` became
unreadable. The table below uses the clean non-node007 rerun
`v79a_exit80_bapr_Ant_dw60_s2_cleanrerun` for that seed.

Ant tail20 summary:

| Variant | Seeds | Tail20 mean +- sample std | Seed tail20 values | Tail50 mean +- sample std | Mechanistic read |
|---|---:|---:|---|---:|---|
| `v79a_exit80` | 0-4 | `1462.1 +- 454.8` | 948.1, 1184.2, 2138.4, 1624.4, 1415.2 | `1467.7 +- 372.0` | Exit threshold is too short/strict; controller stays active in all tail windows |
| `v79b_exit120_regsafe` | 0-4 | `2220.5 +- 887.5` | 2484.3, 1475.7, 2882.6, 1112.2, 3147.7 | `1931.4 +- 739.1` | Best v79 Ant; modestly above v45/v78c but high variance and still not baseline-competitive |
| `v79c_improve_exit` | 0-4 | `1765.3 +- 1210.4` | 3704.4, 1239.0, 474.2, 1431.9, 1976.8 | `1828.6 +- 1108.8` | Exit/cooldown works in 4/5 seeds, but seed2 collapses and variance is too high |

Ant tail controller diagnostics:

| Variant | `controller_active` tail20 | `actor_update_multiplier` tail20 | `reg_multiplier` tail20 | `cooldown_iters` tail20 | `q_std_mean` tail20 |
|---|---|---|---|---|---|
| `v79a_exit80` | 1.00, 1.00, 1.00, 1.00, 1.00 | 0.92, 0.95, 0.93, 0.89, 0.91 | 0.84, 0.90, 0.85, 0.77, 0.82 | 0.0, 0.0, 0.0, 0.0, 0.0 | 22.4, 3.7, 3.8, 4.1, 4.2 |
| `v79b_exit120_regsafe` | 1.00, 0.20, 1.00, 1.00, 1.00 | 0.98, 1.00, 0.96, 0.92, 0.95 | 0.96, 0.99, 0.93, 0.88, 0.92 | 0.0, 58.0, 0.0, 0.0, 0.0 | 4.0, 3.5, 4.8, 4.5, 5.4 |
| `v79c_improve_exit` | 0.00, 0.20, 1.00, 0.00, 0.00 | 1.00, 1.00, 0.97, 1.00, 1.00 | 1.00, 0.99, 0.91, 1.00, 1.00 | 43.5, 58.0, 0.0, 44.5, 31.5 | 4.3, 3.7, 18.7, 4.8, 4.3 |

Canary tail20, single seed:

| Variant | HalfCheetah | Hopper | Walker2d | Read |
|---|---:|---:|---:|---|
| `v79a_exit80` | 9919.5 | 1452.5 | 1617.9 | Weak across all canaries except Walker roughly matches v45 mean |
| `v79b_exit120_regsafe` | 14994.3 | 2271.6 | 660.2 | Best HalfCheetah among v79, but still below v45/ESCP/RE-SAC; Walker collapses |
| `v79c_improve_exit` | 13190.0 | 1682.9 | 2211.1 | Best Walker canary, but HalfCheetah/Hopper are not enough |

Interpretation: v79 confirms the mechanism diagnosis but not the algorithmic
claim. Explicit exit/cooldown can turn recovery off (`v79c` exits in most Ant
seeds), so the implementation is now able to express the desired behavior. The
problem is that the performance optimum is still not clean: the best return
variant (`v79b`) mostly keeps the controller active, while the variant that exits
most cleanly (`v79c`) has a low Ant seed2 and very high variance. Against the
available paper baselines, `v79b` Ant (`2220.5 +- 887.5`) remains below
gravity-fixed ESCP (`2771.2 +- 899.8`) and RE-SAC (`2818.4 +- 1093.0`) and far
below SAC. On HalfCheetah, v79 remains below v45 (`18124.2 +- 1189.6`) and
below ESCP/RE-SAC. Do not promote v79 as the main paper version.

## v80 residual-policy redesign (launched 2026-07-05)

v80 changes the objective rather than only tuning the recovery controller. The
new `conservative_residual` / `v80_residual` actor objective treats the
zero-context EMA policy as a stable base policy, clips the online policy to a
small residual around that base action, and gates the residual by a conservative
ensemble-advantage estimate:

- base action: EMA policy with zero environment context;
- proposal: current context-conditioned BAPR policy;
- residual: clipped proposal minus base action;
- gate: sigmoid of `(proposal safe-Q - base safe-Q - margin) / temp`, where
  safe-Q is `Q_mean - qstd_scale * Q_std`;
- actor loss: safe-Q improvement plus a small behavior penalty toward the base
  policy and optional residual-size penalty.

Rationale: v78/v79 showed that exit/release mechanics can be expressed, but the
controller still fails to produce a robust final-return optimum. v80 makes the
adaptive part earn the right to move away from a stable base policy, instead of
letting the context policy freely drift and then trying to rescue collapse
afterward. This is intentionally conservative and should first be judged by Ant
stability plus HalfCheetah/Hopper/Walker2d canaries.

Submitted first-round scheduler-only batch:

| Variant | Main test | Canary tests | Key residual settings |
|---|---|---|---|
| `v80a_residual_safe` | Ant seeds 0-4 | HalfCheetah/Hopper/Walker2d seed0 | delta 0.20, qstd 0.50, margin 0, behavior 0.08 |
| `v80b_residual_wide` | Ant seeds 0-4 | HalfCheetah/Hopper/Walker2d seed0 | delta 0.35, qstd 0.35, margin -25, behavior 0.03 |
| `v80c_residual_strict` | Ant seeds 0-4 | HalfCheetah/Hopper/Walker2d seed0 | delta 0.25, qstd 1.00, margin 50, behavior 0.12, action penalty 0.01 |

The launch uses scheduler direct only, with checkpoint/resume enabled. No Slurm
or auto-adopt path is involved.

Final status at 2026-07-05: all 24 scheduler-only v80 runs completed and synced
local logs.

Ant tail20 summary:

| Variant | Seeds available | Tail20 mean +- sample std | Seed tail20 values | Mechanistic read |
|---|---:|---:|---|---|
| `v80a_residual_safe` | 0-4 | `1674.6 +- 751.7` | 1334.4, 1613.1, 1126.3, 2982.9, 1316.3 | Below v45 mean and below v78c/v79b; conservative residual is stable but weak |
| `v80b_residual_wide` | 0-4 | `1620.9 +- 575.7` | 1381.6, 1129.9, 1685.9, 1320.2, 2586.6 | Wider residual helps Hopper/Walker canaries but weakens Ant further |
| `v80c_residual_strict` | 0-4 | `1760.4 +- 897.0` | 1985.1, 819.9, 1512.7, 1304.5, 3179.7 | Roughly v45-level mean, but high variance and below v78c/v79b |

Canary tail20, single seed:

| Variant | HalfCheetah | Hopper | Walker2d | Read |
|---|---:|---:|---:|---|
| `v80a_residual_safe` | 13719.8 | 1256.9 | 1692.1 | HalfCheetah/Hopper are weak; Walker roughly v45 mean but below v45 seed0 |
| `v80b_residual_wide` | 14732.6 | 2795.5 | 2281.7 | Best v80 canary set, but HalfCheetah is below v45/v79b and Ant is weak |
| `v80c_residual_strict` | 11457.2 | 1551.1 | 1322.1 | Weak canaries; strict residual penalty is too restrictive |

Final v80 interpretation: v80 does not rescue the algorithm. On Ant, the best
5-seed v80 result is `1760.4 +- 897.0`, essentially v45-level and below v78c
(`2010.6 +- 605.9`) and v79b (`2220.5 +- 887.5`), which were themselves below
ESCP/RE-SAC/SAC. On HalfCheetah, the best v80 canary (`14732.6`) is below v45
(`18124.2 +- 1189.6`) and below the best v79 canary (`14994.3`). Hopper and
Walker2d are the only encouraging single-seed signals for `v80b`, but they do
not compensate for the Ant/HalfCheetah weakness.

The most important mechanism finding is that the residual gate is almost
constant rather than meaningfully adaptive. Tail20 `v80_residual_gate` is around
`0.500` for v80a, `0.522` for v80b, and `0.475` for v80c across Ant; tail
conservative advantages are tiny (`~0.5-1.5`). In practice, v80 mostly acts like
a fixed residual behavior constraint, not like a learned "only adapt when the
residual is safely better" controller. Another small residual hyperparameter
sweep is not recommended; the next redesign needs a sharper adaptive signal or
a different mechanism for deciding when the context policy should override the
stable base policy.

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
   - v78 soft release and v79 exit/cooldown show the next layer of the problem:
     softer multipliers can improve Ant a little, and explicit exit can turn
     recovery off, but neither gives a robust cross-environment improvement.
     The issue is no longer just implementation expressivity; the recovery
     trigger/objective itself is not aligned with broad final-return gains.

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

Conclusion: v45 is still the best valid multi-environment reference/ablation
anchor. v75r_c/v76, v78, v79, and v80 do not supersede it as a paper main result.
Even v45 is not sufficient by itself as the final main paper result unless the
paper is reframed as an analysis of when adaptation helps and fails.

## Recommended next analysis

1. Do not promote `v75r_c`, `v76`, `v78`, `v79`, or `v80` as the main paper version.
2. Keep v45 as the strongest completed reference, while being explicit that it
   is not yet a top-paper-strength algorithmic result.
3. The next algorithmic direction should not keep tuning only the recovery
   release knob or the v80 residual-gate temperature/margin. v79 already shows
   release can work mechanically, and v80 shows the conservative residual gate is
   nearly constant. The next change needs a genuinely sharper adaptive signal,
   or a reframed analysis route that explains why these adaptation controllers
   help Ant modestly but fail to beat strong SAC/ESCP/RE-SAC baselines across
   environments.
4. For future node007 use, confirm the patched code-tar staging with a real
   latest-code smoke or fresh clean run; do not use the old stale-code node007
   runs as evidence.
5. Keep bus-environment validation separate: no MuJoCo-only result is sufficient
   to claim BAPR preserves the required bus effectiveness.
