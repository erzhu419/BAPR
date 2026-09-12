# Frozen-anchor controller development protocol

## Motivation

The first anchored residual development run completed with valid manifests and
equal budgets but failed. Its gate-zero actor was 39.5% and 82.0% below the
paired robust continuation on two of three seeds. Gradient isolation was not
enough: adaptive rollouts entered the shared replay and changed the robust
actor's closed-loop training distribution. True-mode residual control was
also negative on those two seeds.

This v2 protocol removes that ambiguity. It starts from the completed v1
`robust_continue` checkpoint and keeps the adaptive branch's robust actor
byte-for-byte frozen. Any final checkpoint whose base-policy hash differs from
the fork hash is invalid.

## Fixed benchmark and split

- Environment: `HalfCheetah-v2`.
- Family: persistent `actuator_polarity`.
- Dwell: fixed 250 actions.
- Per-transition actuator noise: Gaussian `std=0.02`.
- Development policy seeds: `1103,1213,1301`.
- Calibration event seeds: `96201,96202`.
- Strict audit event seeds: `96301,96302,96303`.
- Frozen expected-action estimator and filter: unchanged from the independent
  v4 confirmation.

These remain development seeds. A passing variant only authorizes a new
untouched five-policy-seed confirmation.

## Equal-budget fork

Each branch starts from its seed-matched v1 robust checkpoint at iteration
`2099`, 8.4M transitions, and 525k updates. Replay and optimizer states are
reset. Every branch then receives another 700 iterations and finishes at
iteration `2799`, 11.2M transitions, and 700k updates.

The common comparator is:

- `robust_long`: ordinary robust `RegimeSAC` for all additional interactions.

Adaptive branches use true-mode rollout context during development, paired
robust/oracle replay relabels, independent adaptive critics, SAC minimum
targets, per-context entropy temperatures, and no BOCD, Q-std gate, LCB,
IPM shift, or RE-SAC regularizer. Their robust base actor is never updated.

## Registered variants

Three variants isolate capacity from anchor preservation:

1. `shared_small`: shared mode-conditioned residual with pre-tanh cap `0.15`.
2. `shared_wide`: the same residual with cap `0.50`.
3. `mode_residual`: independent mode-specific additive `delta mean` and
   `delta log-std` networks with no amplitude cap.

All residual outputs are exactly zero at the fork. `mode_residual` keeps the
base actor on one common computation graph and zero-initializes only the mode
output heads; this avoids GPU rounding differences between copied batched
option networks.

## Calibration and fallback

For each variant and seed, held-out calibration compares:

- equal-budget `robust_long`;
- the adaptive checkpoint at gate zero, `anchored_base`;
- the same checkpoint with true mode, `oracle_residual`.

A mode is enabled only if oracle control exceeds the better of
`robust_long` and `anchored_base` by at least 2% and adds no more than two
percentage points of termination. `learned_safe` enables the frozen soft
posterior only when confidence is at least `0.85` and its MAP mode is enabled;
otherwise it executes the immutable base actor.

Strict audits retain the six registered arms:
`robust_continue` (the v2 `robust_long` alias), `anchored_base`,
`oracle_residual`, `oracle_safe`, `learned_raw`, and `learned_safe`.

## Promotion gates

A variant advances only if:

- its immutable base preserves at least 95% of equal-budget `robust_long` on
  every seed;
- true-mode residual control gains at least 5% on average and wins at least
  2/3 seeds;
- `learned_safe` gains at least 3% on average and wins at least 2/3 seeds;
- relevant termination gaps are at most two percentage points;
- the frozen estimator retains its registered accuracy, Brier, and delay
  thresholds;
- calibration enables at least one mode on every seed.

No threshold or development seed is changed after results are observed.

## Verification and resources

Bootstrap tests reproduce the source action and Q function for zero plus all
four one-hot contexts. Full-shape 250-update GPU tests keep the base-policy
hash unchanged while changing adaptive parameters. Real two-iteration
HalfCheetah resume tests reach `iter=2102` with oracle-only rollout source,
dual replay relabeling, and valid checkpoints.

With the local GPU's 574 MB desktop baseline, full-shape update totals were
956 MB for shared residual and 1212 MB for independent mode residual. The
scheduler claims are therefore measured at 2.8 GB and 4.2 GB respectively;
robust-long retains its historical 1.8 GB claim. GPU nodes are `local`,
`jtl110gpu`, `jtl110gpu2`, and `node007`; `jtl311linux` is excluded.

## Scheduler graph

The complete file-gated matrix contains 33 tasks:

- 3 shared `robust_long` GPU continuations;
- 9 adaptive GPU continuations;
- 9 CPU calibration tasks;
- 9 CPU strict audits;
- 3 CPU aggregates.

All tasks use scheduleurm directly with checkpoint-managed resume and
node-down rerouting. CPU work may use only `node001-node006`; no Slurm or
auto-adopt path is used.

The high-priority graph was submitted as `t63877-t63909`:

- `robust_long`: `t63877-t63879`;
- `shared_small`: `t63880-t63882`;
- `shared_wide`: `t63883-t63885`;
- `mode_residual`: `t63886-t63888`;
- calibrations: `t63889-t63897`;
- strict audits: `t63898-t63906`;
- aggregates: `t63907-t63909`.

At submission, every CPU task correctly reported its missing producer
artifacts. GPU producers entered input staging from the immutable v1 robust
bundles; no CPU evaluation launched early.
