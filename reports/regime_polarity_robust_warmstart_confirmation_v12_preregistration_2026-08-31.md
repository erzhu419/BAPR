# Actor-only robust-warm-start policy-bank confirmation protocol

## Frozen question

V11 found that both robust-warm-start variants passed on the two development
seeds whose earlier fixed-mode banks had no headroom. This confirmation tests
whether the simpler `actor_only` initialization reproduces that result on five
entirely new policy seeds. It does not train or evaluate a learned estimator.

The frozen variant copies only the matched robust SAC actor. Its critic, target
critic, entropy temperature, optimizer states, and replay are initialized
freshly. No environment parameter, regularizer, gate, residual policy, or
posterior is changed after observing V11.

## Training and data split

- New policy seeds: `71003, 71021, 71039, 71057, 71079`.
- Each matched robust SAC source trains for 1400 iterations, 5.6M transitions.
- Four fixed-mode specialists per seed then train for 700 iterations, 2.8M
  additional transitions, from the robust actor only.
- Stationary calibration events: `167001, 167017, 167033`.
- Stationary holdout events: `167101, 167117, 167133`.
- Switching holdout events: `167201, 167217, 167233`.

The switching streams use explicit four-mode cycles rather than treating
different rollout-noise seeds as different switch schedules:

- `167201`: `0,1,2,3`;
- `167217`: `1,3,0,2`;
- `167233`: `2,0,3,1`.

Each episode rotates its frozen cycle by the episode index. Every arm in one
event must see an identical trace, and the three event traces must be distinct.

## Frozen gate

A seed passes only when at least three of four matching specialists beat its
matched robust policy by at least 5% on every stationary holdout event, with no
termination, and the calibration-selected true-mode safe oracle beats robust
by at least 10% on every one of the three distinct switching schedules, also
with no termination.

The policy-bank mechanism is confirmed only if at least four of five new seeds
pass. A failure does not authorize selecting `full_state` post hoc. It closes
the actor-only specialist bank as confirmed evidence and requires a separately
declared development experiment.

Training checkpoints and replay remain on their execution nodes. Scheduler
result synchronization contains only policy parameters and JSON provenance.
