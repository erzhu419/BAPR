# V18 frozen-v5 final comparison preregistration

## Question

Does the frozen v5 posterior-MAP BAPR controller, using the independently
confirmed v17 actor-only policy banks, improve causal switching control beyond
both standard single-policy baselines and a control with the same total number
of trained policies?

No estimator, gate, environment, policy-bank, threshold, or event stream is
tuned in this experiment.

## Frozen candidate and data splits

- Candidate: v17 actor-only robust-warm-start policy bank plus frozen v5
  executed-action posterior, routed by posterior MAP.
- Outer policy seeds: 81003, 81021, 81039, 81057, 81079.
- Calibration streams: 182001, 182017, 182033.
- Stationary holdout streams: 182101, 182117, 182133.
- Switching holdout streams: 182201, 182217, 182233.
- Switching uses explicit balanced four-mode schedules with 250-step dwell,
  five 1000-step episodes, deterministic policy means, and strict horizons.

## Comparators

- Matched robust SAC from v17.
- Fresh recurrent ESCP and released-B0 RE-SAC, each trained for 5.6M steps and
  350k updates under the same stochastic actuator-polarity environment.
- SAC5 equal-total-policy-budget control: the v17 robust SAC plus four new
  switching-trained SAC replicas. Calibration selects a best static policy and
  a best policy per physical mode. The causal SAC5 arm uses the same frozen v5
  posterior as BAPR to route among those policies.
- True-mode BAPR and true-mode SAC5 are diagnostic upper bounds, not deployable
  methods.

Only evaluation parameters, protocol signatures, manifests, and JSON audit
results are synchronized. Full checkpoints and replay buffers remain on their
training nodes for resume and are not copied back.

## Fixed decision rule

The primary arm is `bapr_v5_posterior_map`. A strong final algorithm claim is
accepted only if all of the following hold:

1. Against robust SAC, recurrent ESCP, released-B0 RE-SAC, and causal SAC5, the
   paired switching difference has positive mean and positive 95% t interval,
   wins at least 4/5 outer seeds, and wins at least 12/15 held-out events.
2. It has no higher termination rate than any of those comparators.
3. True-mode BAPR has at least 10% headroom over robust SAC, and causal BAPR
   recovers at least 70% of that headroom on at least 4/5 seeds.
4. On stationary holdout, causal BAPR retains at least 95% of the strongest
   single-policy baseline on at least 4/5 seeds without higher termination.

If BAPR beats the three standard single-policy baselines but not causal SAC5,
the supported conclusion is limited to specialization/adaptation being useful;
there is no evidence of an advantage at equal total policy-training budget. If
it loses a standard baseline, the frozen candidate is not promoted.
