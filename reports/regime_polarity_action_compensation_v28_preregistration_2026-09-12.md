# V28 actuator-polarity action-compensation preregistration

## Question

V21 established a five-seed HalfCheetah result for a frozen v5 posterior and a
bank of full-state-initialized specialists. V28 tests whether that bank is
needed in the current `actuator_polarity` benchmark, whose four regimes differ
only by diagonal action-sign matrices. It is a mechanism audit, not a new
training result or a claim about arbitrary dynamics changes.

## Frozen inputs

- Reuse the five V21 policy seeds `84003, 84021, 84039, 84057, 84079`.
- Reuse each seed's robust source, four final specialists, four independent
  SAC replicas, and the frozen HalfCheetah v5 executed-action estimator.
- Do not train, fine-tune, select a checkpoint, or alter the environment.
- Select one reference specialist per policy seed using calibration only.
- Report BAPR and SAC5 policy counts, original training interactions, and
  estimator cost separately. "Equal-policy-count" replaces "equal budget".

## New event splits

- Calibration: `194001, 194017, 194033`.
- Stationary holdout: `194101, 194117, 194133`.
- Switching holdout: `194201, 194217, 194233`.
- Switching uses five deterministic 1000-step episodes, 250-step dwell, and a
  balanced four-mode schedule. No holdout event participates in selection.

For each reference candidate `r`, calibration evaluates its unmodified policy
only in native mode `r`. The candidate with the highest mean calibration return
is selected; ties resolve to the lower mode id. V21's utility map and SAC5's
mode map are independently rebuilt from the same calibration split by their
already frozen V21 rules.

## Arms

1. `robust_sac`: the V21 robust source with no mode input.
2. `reference_no_compensation`: the selected reference specialist everywhere.
3. `reference_true_mode_compensation`: privileged diagnostic using
   `a_cmd = D_true D_ref pi_ref(s)`.
4. `reference_v5_map_compensation`: causal arm using the v5 posterior available
   before the current transition.
5. `bapr_v21_posterior_map`: the existing V21 specialist bank and frozen v5.
6. `sac5_v21_posterior_map`: the existing five-policy SAC control and frozen v5.

The estimator observes the command actually sent to the environment only after
the resulting transition. The oracle arm alone may read the hidden current
mode. HalfCheetah has no health termination, so return and switch-local control
loss are primary; zero termination is not safety evidence.

## Structural prerequisite

Before holdout evaluation, the implementation must pass:

- algebraic equivalence for action dimensions 3, 6, and 8 under every
  reference/target mode pair and shared actuator-noise draws;
- coupled HalfCheetah and Ant trajectory equivalence under identical initial
  states and random streams;
- end-to-end equality, within `1e-3` return, between the selected policy in its
  native mode and true-mode-compensated switching execution.

Failure is an implementation/protocol failure and blocks scientific adoption.

## Fixed decisions

A causal compensation claim requires all of the following:

- structural and end-to-end oracle equivalence pass;
- causal compensation has positive paired mean and positive 95% t interval
  against both no compensation and robust SAC;
- at least 4/5 policy-seed wins and 12/15 event wins against both controls;
- at least 70% recovery of oracle-over-no-compensation headroom on 4/5 seeds.

Bank necessity is decided separately. If V21 minus causal compensation has a
positive paired mean and interval with at least 4/5 seed wins, the bank adds
confirmed value. If the reverse comparison passes those conditions, structured
single-policy compensation is superior. Otherwise that comparison is
unresolved. SAC5 remains a strong contextual comparator, not an additional
route-selection target.

No Ant estimator is inferred from the HalfCheetah v5 model. Ant receives only
the structural/oracle check in this stage. No follow-on training is authorized
by this registration.
