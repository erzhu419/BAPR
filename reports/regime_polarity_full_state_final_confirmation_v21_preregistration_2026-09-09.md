# V21 full-state-final equal-policy-budget confirmation preregistration

## Selection basis and frozen boundary

V19 and V20 were development experiments. Across six fresh development policy
seeds, full-controller specialist initialization followed by the final
fixed-mode policy beat robust SAC on every seed and every switching event. In
V20 alone it passed all three bank cells, all 12 stationary seed-mode cells,
and all nine switching events, with mean safe switching return 3182.3 versus
1510.8 for robust SAC.

V20's two validation-selected arms are not used here. A post-run audit found
that their selector received a four-mode average rather than the registered
matching-mode validation return. The frozen V20 aggregate remains unchanged
and selected no registered candidate. V21 is a new confirmation protocol based
only on the valid `full_state_final` development arm.

V21 freezes the following recipe before any new result is generated:

- Train a switching robust SAC source for 1400 iterations, 5.6M transitions,
  and 350,000 updates.
- For each of four physical modes, copy actor, critic, target critic, and alpha;
  reset replay and all optimizer states; then train on that fixed mode for 700
  iterations, 2.8M additional transitions, and 175,000 updates.
- Publish the final policy. No validation checkpoint selection or actor-update
  thinning is used.
- Route the calibrated safe policy bank with the already frozen v5
  executed-action posterior MAP. The actuator-polarity environment, stochastic
  process, posterior, router, utility rule, and all thresholds are unchanged.

## Independent data

Confirmation policy seeds are `84003,84021,84039,84057,84079`. Calibration
event streams are `186001,186017,186033`; stationary holdout streams are
`186101,186117,186133`; switching holdout streams are
`186201,186217,186233`. None appeared in V18-V20. Switching uses balanced
four-mode schedules, 250-step dwell, five deterministic 1000-step episodes,
and strict horizons.

## Comparators and budgets

- Matched robust SAC is the source policy in the BAPR bank.
- Recurrent ESCP and released-B0 RE-SAC each train from scratch for 5.6M
  transitions and 350,000 updates under the same switching environment.
- Causal SAC5 uses the robust source plus four independently initialized SAC
  replicas, each trained for 5.6M transitions and 350,000 updates. Calibration
  selects both its best static controller and its best controller per mode; the
  same frozen v5 posterior routes causal SAC5 and BAPR.
- True-mode BAPR and true-mode SAC5 are diagnostic upper bounds only.

The principal comparison is deliberately conservative: BAPR uses five trained
policies in total, as does SAC5. BAPR specialists receive more total interaction
than a single SAC policy because each starts from the robust source, but the
comparison retains the exact budget convention registered for V18 so that the
only changed mechanism is specialist stability.

## Fixed decision rule

The primary arm is `bapr_v5_posterior_map`. A strong algorithm claim passes only
if all conditions hold:

1. Against robust SAC, recurrent ESCP, released-B0 RE-SAC, and causal SAC5, the
   paired switching difference has positive mean and positive 95% t interval,
   wins at least four of five policy seeds, and wins at least 12 of 15 events.
2. BAPR has no higher termination rate than any comparator.
3. True-mode BAPR has at least 10% headroom over robust SAC, and causal BAPR
   recovers at least 70% of that headroom on at least four seeds.
4. On stationary holdout, causal BAPR retains at least 95% of the strongest
   single-policy comparator on at least four seeds without higher termination.

If BAPR passes the three standard single-policy comparisons but not causal
SAC5, only the limited adaptation claim is retained. No event, seed, threshold,
environment, estimator, specialist, or checkpoint may be changed after
registration.

## Execution and artifacts

All jobs use scheduleurm directly. GPU jobs exclude `local`; no Slurm or
auto-adopt path is used. Producer retries are checkpoint-safe and may migrate
through scheduler-managed staging. Only compact policy/controller parameters,
protocol signatures, manifests, and JSON audits are synchronized. Replay
buffers and full checkpoints remain on their training nodes.
