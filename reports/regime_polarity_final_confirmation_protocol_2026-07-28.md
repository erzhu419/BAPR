# Frozen BAPR final confirmation

## Frozen method

The expected-action v4 development ablation passes every inference and
frozen-control gate without realized simulator action labels. Before any
final controller is trained, the estimator is frozen byte-for-byte:

- model manifest SHA-256:
  `fc4b28662ac1f38081aab02723ebcbea2fa47fbe37eea27bb4e1a564b8d571d0`;
- parameter archive SHA-256:
  `c1ff9021a3d3684116464c421c852eb16cd7ab286b2fd53bcd764d9a50393af1`;
- filter: hazard `0.002`, evidence scale `1.0`, posterior decay `0.98`;
- environment: HalfCheetah `actuator_polarity`, fixed 250-action dwell,
  Gaussian actuator noise `std=0.02`;
- online inputs: consecutive observations and commanded action only.

Any estimator, filter, threshold, environment, or protocol change after this
registration invalidates the confirmation.

## Independent split

- New policy-training seeds: `607,719,823,929,1031`.
- New event seeds: `95001,95002,95003`.
- Roles: equal-budget robust zero-context controller and true-context
  conditioned controller.
- Every controller starts from scratch and trains for 1400 iterations,
  5.6M transitions, and the same update budget used by the earlier
  confirmation.

The frozen estimator was fit only on policy seeds `8,16`, with filter
selection on seed `24`. Development policy seeds
`101,211,307,419,523` are not reused.

## Final audit

Each newly trained seed is evaluated in one paired task with four arms:

- robust controller;
- true-context oracle controller;
- frozen v4 soft posterior driving the conditioned controller;
- frozen v4 MAP posterior driving the conditioned controller.

Every arm shares event streams, strict stationary modes, switching sequences,
episode horizon, and action-level causal timing. Learned actions at time `t`
use only transitions before `t`.

The frozen gates are:

- mode accuracy at least `0.85`;
- median/P90 switch delay at most `25/50` actions;
- Brier score at most `0.25`;
- mean oracle-headroom recovery at least 50%;
- at least 4/5 policy-seed wins over robust;
- switching termination gap no more than five percentage points.

The primary final arm is soft posterior. MAP is a registered secondary
deployment ablation. Passing supports the BAPR mechanism on this benchmark;
failure is reported without further tuning on these seeds.

## Scheduler execution

The complete file-gated graph was submitted at high priority on 2026-07-28:

- robust training: `t58889-t58893`;
- oracle training: `t58894-t58898`;
- paired CPU audits: `t58899-t58903`;
- immutable JSON aggregate: `t58904`.

Training uses scheduler-managed checkpoint resume and node-down rerouting on
`local`, `jtl110gpu`, `jtl110gpu2`, and `node007`; `jtl311linux` is excluded.
Audits may run only on `node001-node006` after both seed-matched controller
bundles and the frozen estimator files exist. The aggregate is similarly
blocked on all five validated audit manifests.

## Final result (2026-07-29)

All 16 tasks completed. Every one of the ten controller bundles validates at
iteration `1399`, 5.6M transitions, and 350k updates; all five audit manifests
and their payload hashes validate.

| Arm | Switching return | Delta vs robust | Paired 95% CI | Wins |
|---|---:|---:|---:|---:|
| robust | 1340.1 | - | - | - |
| true-mode oracle | 2283.5 | +943.3 | [-221.4, 2108.1] | 4/5 |
| frozen soft posterior | 2161.5 | +821.4 | [-287.1, 1929.8] | 3/5 |
| frozen MAP posterior | 2156.1 | +816.0 | [-267.6, 1899.6] | 3/5 |

The inference gate passes again on the independent split: soft mode accuracy
is `0.995`, Brier score is `0.008`, and median/P90 switch delays are `4/10`
actions. Mean stationary return is `2303.0` for soft, `2330.9` for oracle,
and `1344.2` for robust, with no termination gap.

The preregistered final control gate nevertheless **fails**. Seed `719` has
only `+61.4` true-mode oracle headroom, so inference delay changes the soft
delta to `-45.7`. On seed `1031`, even the privileged oracle is `-180.6`
below robust; oracle-headroom recovery is therefore undefined rather than an
infinite negative ratio. These seeds remain in every paired statistic. The
soft estimator is close to oracle on all five seeds (`-122.0` mean gap), so
the failure is controller specialization variance, not mode inference.

This sealed split is not used for further tuning. The frozen v4 estimator is
retained as a successful causal system-identification component, but the
conditioned controller is not a confirmed final BAPR method. A subsequent
development branch must anchor adaptation to a paired robust actor: initialize
a zero residual from the trained robust policy, retain the exact robust action
under high posterior entropy or non-positive validated advantage, and train
only the mode-conditioned residual. It requires a separate development split
and another untouched confirmation split before any positive main claim.
