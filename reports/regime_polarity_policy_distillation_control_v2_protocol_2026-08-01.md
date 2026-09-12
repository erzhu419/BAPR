# Closed-loop policy-compression v2 protocol

## Motivation

The frozen v1 student confirmed a large gain over the ten-controller robust
population but failed against the preregistered strongest robust controller
`robust_final_seed_719`: `-33.6` return, `1/5` event wins. This is a compression
failure rather than missing adaptation headroom. On the same sealed split, the
causal learned-median teacher beats robust `719` by `+58.5` and the oracle
median teacher by `+111.9`; posterior mode accuracy is `0.9978`.

The v1 student was selected by offline action loss and saw only two fixed
DAgger event seeds. V2 therefore changes policy compression only. The
environment, frozen expected-action estimator, combined-ten median teacher,
strict horizon, comparator controllers, and causal online inputs remain fixed.

## Development variants

Three variants and three student initialization seeds (`1709,1811,1901`) form
nine GPU training tasks:

1. `wide_dagger`: a three-layer, width-512 posterior-conditioned MLP. This
   controls for capacity and broader DAgger coverage.
2. `mode_heads`: a shared observation trunk with four separate action heads,
   mixed by the causal soft posterior. This tests conditional-head
   interference without return weighting.
3. `mode_heads_return`: the same factorized model, with DAgger samples weighted
   by matched teacher-student return regret, post-switch age, and action
   disagreement. Weights are clipped and normalized to mean one.

All variants use four DAgger rounds. Each phase first minimizes weighted action
and pre-tanh imitation loss, then receives a closed-loop switching audit on an
independent control-validation split. The final checkpoint is the phase with
the highest mean switching return after a fixed `2000 * termination_rate`
penalty. Supervised loss is only a within-phase optimizer criterion and a tie
breaker; audit returns never select a model.

## Data separation

- initial collection: `98001,98002,98003`;
- DAgger collection: `101201,101219,101237,101261`;
- supervised validation: `98901,98902`;
- checkpoint-selection control validation: `101401,101417`;
- development control audit: `101603,101627,101653`.

The prior frozen confirmation events
`100019,100043,100069,100103,100151` remain sealed and are never loaded by this
development sweep. Passing v2 can authorize a separately registered
confirmation on another untouched split, but cannot revise the failed v1
confirmation.

## Development gates

Every student is compared on paired event streams with all ten individual
robust controllers, fixed robust `final_seed_719`, the causal learned-median
teacher, and the true-mode oracle median teacher. A student passes only if:

1. it beats the mean individual-robust population on all `3/3` audit events;
2. it beats fixed robust `719` on at least `2/3` audit events;
3. it recovers at least `80%` of learned-teacher headroom over the robust
   population;
4. its mean return is no more than `25` below the learned teacher;
5. switching termination rate is zero.

A variant passes only when at least two of three student initializations pass
and the student chosen solely by control-validation score also passes. Among
passing variants, that same pre-audit score selects the candidate for a future
confirmation.

## Execution and recovery

The scheduler graph contains nine GPU producers, 27 file-gated 32-core CPU
audits, and one file-gated aggregate. At most 36 tasks can run before the
aggregate is eligible. GPU work is allowed on `local`, `jtl110gpu`,
`jtl110gpu2`, `jtl311linux`, and `node007`, with an explicit `2400 MB`
estimate. CPU audits are restricted to `node001-node006`.

Each completed optimization phase writes immutable current parameters,
selected parameters, and only that phase's dataset increment. Scheduler
checkpoint migration can therefore resume after the latest verified phase.
The run uses scheduler submission only, with no Slurm or auto-adopt path.

The graph was submitted atomically at high priority on 2026-08-01:

- GPU training: `t64744-t64752`;
- file-gated CPU audits: `t64753-t64779`;
- file-gated aggregate: `t64780`.

The first producer (`t64744`) launched on `local:GPU0`. The other producers
remain scheduler-managed and may migrate among the allowed GPU nodes; no task
is hard-pinned. Audit and aggregate tasks remain gated on their declared model
or audit manifests.

After `jtl311linux` became available, its two GPUs were added to the queued
training whitelist and to the reusable submitter. This is an allowed-node
expansion, not a hard pin; checkpoint migration and fallback placement remain
enabled.

After this sweep completed, the reusable submitter was restricted to
`jtl311linux` for any future GPU retry or extension. The completed placements
above are historical; new BAPR GPU work must leave `local`, `jtl110gpu`,
`jtl110gpu2`, and `node007` available to other projects.
