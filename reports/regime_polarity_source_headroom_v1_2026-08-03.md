# Independent source-controller headroom protocol

## Question

The current deployable BAPR stack has a real causal inference/fallback
mechanism, but the final five-seed superiority gate failed because the shared
distilled controller stays near a 2k return plateau while independent SAC or
ESCP controllers can occasionally reach 2.4k-2.7k. Before training another
estimator, this protocol asks whether the actuator-polarity benchmark has a
large, reproducible policy-specialization upper bound at equal per-controller
budget.

## Frozen source ladder

Two fresh development seeds `4021,4049` independently train six controllers in
`HalfCheetah-v2` `actuator_polarity`:

- one robust SAC on the switching four-mode process;
- one ESCP controller on the same switching process;
- four independent SAC specialists, each fixed to exactly one physical mode.

Each controller receives 1,400 iterations, 4,000 transitions per iteration,
250 updates per iteration, and therefore exactly 5.6M transitions and 350k
updates. Modes persist for 250 actions. Every controller uses its own actor,
critic, target critic, optimizer, alpha, replay, and random initialization;
the specialists do not share a trunk or a teacher.

This is deliberately an optimistic upper bound: the dynamic oracle owns four
times the policy parameters and selects the matching independent specialist
from the true physical mode. It is not a deployable BAPR result.

## Cross-controller audit

Three fresh event streams `121001,121013,122003` produce:

- the full controller-by-stationary-mode return matrix;
- strict five-episode, 1,000-action switching returns for robust SAC, ESCP,
  every fixed specialist, and the privileged dynamic oracle;
- termination counts and exact source-bundle hashes;
- an assertion that every dynamic-oracle action was selected from the
  controller matching the physics mode used by that transition.

For each training seed, the upper-bound gate requires all of the following:

- dynamic-oracle stationary mean exceeds the strongest single controller by
  at least 10%;
- dynamic-oracle worst-mode return exceeds the strongest single controller's
  worst mode by at least 10%;
- dynamic-oracle switching return exceeds the strongest single controller by
  at least 10%;
- dynamic oracle beats every fixed specialist in switching return;
- the matching specialist is stationary-optimal in at least 3/4 modes;
- dynamic oracle has no more switching terminations than the strongest
  switching comparator.

Both fresh training seeds must pass. Only then may a posterior-conditioned
controller or learned estimator be trained. If either seed fails, estimator
training remains blocked: the benchmark/controller source pair lacks enough
reproducible headroom, regardless of how accurately a mode estimator could be
tuned.

## Scheduler graph

The graph contains 12 GPU source-controller tasks, two file-gated CPU audits,
and one file-gated aggregate. GPU tasks are restricted to `jtl311linux`; the
existing measured source-controller signature requests 2,300 MB. CPU tasks are
restricted to `node001-node006` and stage all six compact bundles for one seed.
The measured signature permits at most three-way packing on each 8 GB card;
the normal VRAM margin remains active. Checkpoint resume is command-managed.
No task uses Slurm, auto-adopt, or a learned estimator.

## Submitted graph

The scheduler graph is `t67077-t67091`: source-controller training is
`t67077-t67088`, per-seed CPU audit is `t67089-t67090`, and the aggregate is
`t67091`. The first dispatch launched `t67077-t67082` only on `jtl311linux`,
with three measured tasks per GPU. Every launched task passed CLI parsing,
built its CUDA rollout, completed iteration 0, and wrote a resumable
checkpoint. The remaining training tasks wait for those measured slots; the
audits and aggregate remain blocked on their exact prerequisite files.

## Result

All tasks `t67077-t67091` completed and the two audit manifests validate. Seed
4021 obtains dynamic-oracle gains of `+52.7%` stationary mean, `+7.2%`
stationary worst mode, and `+69.0%` switching over its strongest single
controller. Seed 4049 obtains `+49.2%`, `+40.0%`, and `+71.3%`, respectively.
Both seeds have `4/4` diagonal specialist optima and zero switching
terminations for the relevant comparator and oracle.

Seed 4021 fails only the preregistered `10%` worst-mode improvement floor;
seed 4049 passes every check. The strict two-seed headroom gate is therefore
`False`. The environment has substantial average and switching specialization
headroom, but this source-controller ladder does not establish the required
reproducible worst-mode margin. Learned-estimator training remains blocked.
