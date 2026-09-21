# BAPR four-environment evidence

This report consolidates the latest valid MuJoCo evidence. The rows are not all
algorithm evaluations: HalfCheetah has a causal learned-estimator result, Ant is
a privileged true-mode mechanism-development result, and Hopper/Walker2d are
oracle headroom screens. Keeping these tiers separate prevents an invalid
four-environment BAPR claim.

## Consolidated table

| Environment | Protocol and evidence tier | Adaptive / privileged switching return | Matched robust return | Relative gain | Termination | Registered decision |
|---|---|---:|---:|---:|---:|---|
| HalfCheetah-v2 | V32 prospective fixed-reference + causal v5 compensation, 10 x 3 seeds | 3297.1 | 2287.1 | +44.2% | 0.0% / 0.0% | **FAIL**: mean/CI passes, but SAC consistency is 7/10 seeds and 23/30 events |
| Ant-v2 | V29 frozen reference + true-mode action compensation, 3 x 3 seeds | 4510.7 | 2588.6 | +74.3% | 4.4% / 11.1% | **FAIL**: return headroom passes, absolute safety gate fails |
| Hopper-v2 | V27 equal-budget true-mode oracle headroom, 3 x 3 seeds | 2463.9 | 2424.0 | +1.6% | 100% / 100% | **FAIL**: no survival-valid adaptation window |
| Walker2d-v2 | V27 equal-budget true-mode oracle headroom, 3 x 3 seeds | 2141.0 | 2186.7 | -2.1% | 100% / 100% | **FAIL**: no headroom and no survival-valid window |

Termination is shown as adaptive-or-privileged / robust. V28 uses a frozen
causal learned estimator and is deployable under the audited information flow.
V29 and V27 expose true mode and therefore measure mechanism or environment
capacity, not deployable BAPR performance.

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
that supports the original policy-bank BAPR claim.

V28 isolates a stronger and simpler development mechanism. One calibration-selected frozen
reference policy plus causal action-sign compensation reaches 4076.2 switching
return, versus 3246.0 for the V21 BAPR bank, 2301.4 for the causal SAC5 selector,
and 1424.8 for robust SAC. It wins all five policy seeds and all 15 switching
events against each comparator. Its causal estimator has 98.84% mode accuracy
and a three-step median switch delay. Thus the current HalfCheetah evidence
supports causal coordinate compensation, not a need for multiple policy heads.

V31 then tests that mechanism on five fresh policy seeds with a reference mode
fixed before training and equal 8.4M interaction budgets. Causal compensation
reaches 3400.6, versus 2594.3 for SAC, 1901.8 for ESCP, and 1916.1 for RE-SAC.
It passes the preregistered ESCP comparison, exact oracle equivalence, 5/5
oracle-recovery, and 4/5 stationary-retention gates. It nevertheless fails the
overall confirmation because the two-sided paired CIs versus SAC and RE-SAC
barely include zero: [-30.2, 1642.8] and [-23.8, 2992.8]. Both still win 4/5
policy seeds and 12/15 events. Seed 87021 has only 2.1% true-mode headroom over
SAC, so this is reference-controller seed variance rather than estimator
failure; v5 accuracy remains 98.26% with a three-step median delay.

V32 keeps the V31 algorithm fixed and uses ten additional policy seeds selected
before training. Causal compensation reaches 3297.1, versus 2287.1 for SAC,
2046.0 for ESCP, and 2360.2 for RE-SAC. All three paired mean differences now
have positive two-sided 95% CI lower bounds: +1009.9 [391.7, 1628.2] against
SAC, +1251.1 [553.5, 1948.7] against ESCP, and +936.9 [337.1, 1536.7] against
RE-SAC. ESCP and RE-SAC pass all registered comparison gates. SAC does not:
7/10 seed wins and 23/30 event wins miss the frozen 8/10 and 24/30 thresholds.
The overall registered result is therefore still FAIL despite a statistically
resolved average advantage.

The V32 failure is again controller-side. True-mode oracle headroom over SAC is
only +2.3%, -2.3%, and +4.1% for the three SAC non-winning seeds, whereas causal
oracle recovery exceeds 95% in each, and both recovery and stationary-retention
gates pass 10/10. Frozen-v5 mode accuracy is 98.55% with a three-step median
delay. Further estimator or gate tuning is not supported by these results.

## Negative and diagnostic evidence

Ant has much larger action-compensation headroom than V26 suggested. V29's exact
true-mode transform reaches 4510.7 versus 2588.6 for robust SAC, wins 3/3 policy
seeds and 9/9 event cells, and is exactly trajectory-equivalent to the selected
reference policy's native mode. Its 4.4% switching termination is also below the
dynamic specialist bank's 28.9% and robust SAC's 11.1%. It still fails the
preregistered absolute safety gate: two reference policies terminate on new
stationary or switching streams after appearing safe during calibration. This
is reference-policy stochastic stability, not mode headroom or transform error.

Hopper and Walker2d fail earlier than estimator design. Both robust and oracle
controllers terminate in every stationary and switching audit. All 45 switching
episodes per role and environment terminate before the first scheduled switch
at step 250, so those protocols cannot identify causal online adaptation.

## Claim boundary

The defensible result is one confirmed causal policy-bank result (V21), one
strong action-compensation development result (V28), and two independent
fixed-reference cohorts (V31/V32) with large average gains but failed registered
consistency decisions, plus one strong privileged
mechanism result that misses its safety gate, and two invalid adaptation windows,
not universal MuJoCo superiority. HalfCheetah supports learned causal action
adaptation under V21, while the simpler compensation claim remains promising but
not uniformly confirmed across policy seeds. Ant shows the same
coordinate mechanism has large return headroom but lacks a reliably safe frozen
reference policy. Hopper and Walker2d show that the tested structured channel is
too failure-prone to evaluate adaptation. Bus remains a separate positive
application domain and is not pooled into this MuJoCo table.

V30 additionally rules out the existing robust policy as an Ant safety shield.
On states visited by the V29 compensated reference, its full 250-step
continuation terminates 9.9% of paired branches versus 0.3% for the candidate,
is worse in every seed and actuator mode, and loses about 582.5 return. All nine
source trajectories survive 1,000 steps and only two unique candidate branches
fail, so the registered finite-horizon risk-model gate has neither enough
positive examples nor fallback headroom. This closes the Ant shielding branch;
it does not alter V29's privileged return-headroom result.

## Sources of record

- `results_regime_polarity_full_state_final_confirmation_analysis_v21/analysis.json`
- `results_regime_polarity_action_compensation_analysis_v28/analysis.json`
- `results_regime_polarity_action_compensation_confirmation_analysis_v31/analysis.json`
- `results_regime_polarity_action_compensation_power_analysis_v32/analysis.json`
- `results_regime_polarity_ant_action_compensation_analysis_v29/analysis.json`
- `results_regime_polarity_ant_branch_risk_analysis_v30/analysis.json`
- `results_regime_channel_survival_headroom_analysis_v27/analysis.json`
- `reports/regime_polarity_full_state_final_confirmation_v21_2026-09-09.md`
- `reports/regime_polarity_action_compensation_v28_2026-09-12.md`
- `reports/regime_polarity_action_compensation_confirmation_v31_2026-09-12.md`
- `reports/regime_polarity_action_compensation_power_v32_2026-09-13.md`
- `reports/regime_polarity_ant_action_compensation_v29_2026-09-12.md`
- `reports/regime_polarity_ant_branch_risk_v30_2026-09-12.md`
- `reports/regime_channel_survival_headroom_v27_2026-09-12.md`
