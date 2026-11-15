# RE-SAC bus source-fidelity audit

Date: 2026-10-04

## Scope

This is a static source audit, not a new training result. It compares the
original RE-SAC bus implementation in
`/home/erzhu419/mine_code/RE-SAC/sac_ensemble_original_logging.py` and
`/home/erzhu419/mine_code/RE-SAC/env_original/sim.py` with the bus policy-bank
implementation in this repository.

## What matches

The BAPR bus controller preserves the main numerical conventions of the
original implementation: positive `weight_reg`, `beta=-2`, `beta_ood=0.01`,
`beta_bc=0.001`, `gamma=0.99`, `critic_actor_ratio=2`, batch size 2048,
ensemble size 10, hidden width 64, learning rate `1e-5`, reward
standardization to scale 10, and an entropy coefficient capped at 0.6. The
regularization term is added with the positive sign to both the target and
policy objectives in both implementations.

## What is not equivalent

1. The original environment is `env_original/sim.py` with no mode-profile
   engine. The BAPR environment adds persistent `MODE_PROFILES`, random initial
   modes, fixed modes, and scheduled mode switches. Its state/action bookkeeping
   also expands the nominal agent table for the switching workload.
2. The original RE-SAC experiment trains one controller for 500 episodes. The
   BAPR policy-bank protocol trains a robust source for 500 episodes and then
   five fixed-mode specialists for 200 additional episodes each, resetting
   optimizers and replay. This is a different controller and budget, not a
   reproduction of the original training run.
3. The BAPR policy-bank audit selects among robust and specialist controllers
   with privileged mode labels. That is an oracle/headroom screen for BAPR; it
   is not the original RE-SAC evaluation protocol.

## Decision

The BAPR bus policy-bank result cannot be used as a source-aligned RE-SAC
replication. Its registered headroom screen already failed: zero of three
policy seeds passed the required gate, with switching relative gain only
0%, 0.89%, and 0% for the three seeds. The result remains useful as a negative
headroom diagnostic, but it does not refute the original RE-SAC bus result.

For the BAPR paper, keep the bus result as contextual evidence only and keep
the current HalfCheetah/Reacher scope. A genuine RE-SAC source replication
would need a separate preregistered native `env_original` experiment with the
original single-controller protocol and must not be pooled with BAPR policy-bank
results. No new BAPR training is justified by this audit.
