# Polarity controller-variance diagnostic

## Question

The true-context conditioned policy establishes large actuator-polarity
headroom, and the frozen expected-action estimator recovers most of that
headroom. The independent final confirmation nevertheless loses on two of
five controller seeds. Stable-LCB bounded residuals do not repair this rare
controller-seed failure. This diagnostic asks whether independently trained
conditioned controllers can be combined into a stable teacher without any new
RL training.

## Sealed checkpoint-only protocol

- Environment: `HalfCheetah-v2`, `actuator_polarity`, fixed 250-action dwell.
- Controller groups are analyzed separately:
  - development: seeds `101,211,307,419,523`;
  - final: seeds `607,719,823,929,1031`.
- Event seeds `97001,97002,97003` are fresh relative to all earlier polarity
  audits.
- Each event uses strict 1000-action stationary and switching horizons with
  five episodes, matching the confirmation protocol.
- Arms include every individual robust/oracle controller, deterministic
  coordinate-wise mean and median robust/oracle ensembles, and mean/median
  oracle ensembles driven causally by the frozen expected-action v4 posterior.
- Learned arms may observe only `(observation, commanded action,
  next observation)`. True mode is used only by the explicit oracle arms.
- All arms on an event seed recreate the same environment stream. No
  checkpoint is updated and no replay data is generated.

This is retrospective controller diagnosis, not a new confirmatory result.
The development and final controller groups must not be pooled to claim a new
ten-seed confirmation.

## Decision rule

For a reduction to be a viable teacher, both controller groups must satisfy:

1. oracle ensemble beats its matched robust ensemble by at least 10% on every
   fresh event seed;
2. learned ensemble also beats robust by at least 10% on every event seed;
3. learned ensemble recovers at least 70% of ensemble oracle headroom.

Passing supports offline distillation into one posterior-conditioned student.
If oracle aggregation passes but learned aggregation fails, the remaining
problem is soft-posterior action geometry. If oracle aggregation itself fails,
naive action averaging is retired and no distillation or GPU run follows.

## Scheduler

Six CPU audit tasks and one file-gated aggregate were submitted through the
scheduler. GPU training, Slurm, and auto-adopt are excluded. CPU placement is
restricted to `node001-node006`.

The active graph is `t64515-t64521`: `t64515-t64520` are the six independent
event audits and `t64521` is the file-gated aggregate. An earlier prelaunch
graph, `t64508-t64514`, was cancelled without executing after the launch
staging gate incorrectly combined `require_node=local` with an explicit
remote-only allowlist. The scheduler gate now preserves explicit remote-only
placement and fails clearly if an oversized working tree cannot be staged;
the submitter also excludes all historical `results*`, `eval_bundles*`, and
`paper/` trees while staging the ten required checkpoint bundles explicitly.

## Result

All seven tasks completed and synced. Both reductions pass the original
matched-ensemble gate, but that relative percentage is misleading because
independently trained robust actions do not average into a valid robust policy.
The development robust mean/median ensembles score only `227.4/407.9`, versus
an individual-robust mean of `1037.8`; the final robust mean/median ensembles
score `-34.2/-93.1`, versus an individual-robust mean of `1338.4`.

The useful result is absolute teacher quality and reproducibility. Learned
median scores `2062.3` for the development group and `1970.9` for the final
group, with zero termination, and recovers `93.6%/94.6%` of the corresponding
oracle-median headroom. It is stronger than learned mean in both groups.
Individual oracle conditioning wins `5/5` development controllers and `4/5`
final controllers, confirming that controller-seed variance remains real.

The next registered screen therefore distills the frozen median teacher into
one posterior-conditioned policy. It compares the student against individual
robust controllers on new event seeds; the collapsed robust action ensemble is
not reused as a baseline. See
`reports/regime_polarity_policy_distillation_protocol_2026-08-01.md`.
