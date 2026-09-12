# BAPR four-environment evidence

This report consolidates the latest valid MuJoCo evidence. The rows are not all
algorithm evaluations: HalfCheetah is a frozen BAPR confirmation, Ant is a
privileged true-mode controller-development result, and Hopper/Walker2d are
oracle headroom screens. Keeping these tiers separate prevents an invalid
four-environment BAPR claim.

## Consolidated table

| Environment | Protocol and evidence tier | Adaptive / privileged switching return | Matched robust return | Relative gain | Termination | Registered decision |
|---|---|---:|---:|---:|---:|---|
| HalfCheetah-v2 | V21 frozen BAPR v5, 5 policy seeds x 3 event seeds | 3197.3 | 1448.8 | +120.7% | 0.0% / 0.0% | **PASS**: direct BAPR evidence |
| Ant-v2 | V26 equal-budget safe true-mode joint controller, 3 x 3 seeds | 3117.2 | 2598.1 | +19.5% | 6.7% / 13.3% | **FAIL**: only 1/3 seed gates passed |
| Hopper-v2 | V27 equal-budget true-mode oracle headroom, 3 x 3 seeds | 2463.9 | 2424.0 | +1.6% | 100% / 100% | **FAIL**: no survival-valid adaptation window |
| Walker2d-v2 | V27 equal-budget true-mode oracle headroom, 3 x 3 seeds | 2141.0 | 2186.7 | -2.1% | 100% / 100% | **FAIL**: no headroom and no survival-valid window |

Termination is shown as adaptive-or-privileged / robust. V21 reports the learned
BAPR policy itself. V26 and V27 deliberately expose true mode and therefore
measure controller or environment capacity, not deployable BAPR performance.

## HalfCheetah confirmation

The frozen V21 BAPR policy beats every matched comparator on every policy seed
and every switching event. Its mean switching return is 3197.3, compared with
1362.9 for recurrent ESCP, 1691.0 for RE-SAC, 1448.8 for robust SAC, and 2397.9
for the causal five-replica SAC selector.

| Comparator | Paired BAPR difference | 95% CI | Seed wins | Event wins |
|---|---:|---:|---:|---:|
| Robust SAC | +1748.5 | [+1368.7, +2128.2] | 5/5 | 15/15 |
| Recurrent ESCP | +1834.4 | [+1011.4, +2657.4] | 5/5 | 15/15 |
| RE-SAC | +1506.3 | [+880.6, +2132.0] | 5/5 | 15/15 |
| Causal SAC5 selector | +799.4 | [+199.3, +1399.5] | 5/5 | 15/15 |

V21 also passes oracle recovery and stationary retention on 5/5 policy seeds,
with no termination in any comparator arm. This is the only current MuJoCo row
that supports a direct algorithm claim for BAPR.

## Negative and diagnostic evidence

Ant still has adaptation headroom: the safest equal-budget privileged controller
improves mean switching return by 19.5%. It is not reliable enough for adoption,
however: only one of three policy seeds passes the complete gate, and the V26
analysis closes further Ant controller tuning on those development seeds.

Hopper and Walker2d fail earlier than estimator design. Both robust and oracle
controllers terminate in every stationary and switching audit. All 45 switching
episodes per role and environment terminate before the first scheduled switch
at step 250, so those protocols cannot identify causal online adaptation.

## Claim boundary

The defensible result is one strong positive environment plus three informative
limits, not universal MuJoCo superiority. HalfCheetah supports BAPR under the
persistent actuator-polarity protocol. Ant shows privileged headroom without a
stable deployable controller. Hopper and Walker2d show that the tested structured
channel is too failure-prone to evaluate adaptation. Bus remains a separate
positive application domain and is not pooled into this MuJoCo table.

## Sources of record

- `results_regime_polarity_full_state_final_confirmation_analysis_v21/analysis.json`
- `results_regime_polarity_ant_joint_mode_risk_analysis_v26/analysis.json`
- `results_regime_channel_survival_headroom_analysis_v27/analysis.json`
- `reports/regime_polarity_full_state_final_confirmation_v21_2026-09-09.md`
- `reports/regime_channel_survival_headroom_v27_2026-09-12.md`
