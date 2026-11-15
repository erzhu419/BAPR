# BAPR final validation execution record

## Frozen evidence

- Corrected five-seed switching mean: BAPR 2015.3, recurrent ESCP 1633.8,
  released-B0 RE-SAC 1568.4, and SAC 1222.5.
- Registered strongest-per-seed diagnostic: BAPR +4.86%, 4/5 wins, interval
  [-298.0, 485.0]; the primary superiority gate is false.
- Final checkpoint-only mechanism audit: `t79072-t79097`, pass.
- Deployment footprint: 883,266 float32 parameters (3.37 MiB).
- End-to-end BAPR construction: 147.059M interactions and 9.403M updates.

## Active registered graphs

### Fresh deployment confirmation

- Registration:
  `jax_experiments/deployments/regime_polarity_deployment_confirmation_v1/registration.json`.
- Tasks: `t79307-t79346` GPU training, `t79347-t79386` CPU audit, and
  `t79387` aggregate.
- Scope: ten untouched model seeds, five untouched event seeds, and four
  frozen methods.
- Claim gate: familywise corrected switching superiority plus stationary and
  termination safeguards.
- The first eight remote BAPR producers `t79307-t79314` stopped before
  training because the launch-stage list contained the mechanism registration
  but not three artifact directories needed to revalidate its source closure.
  The frozen files themselves were present and unchanged. Those directories
  were staged through scheduleurm to `jtl110gpu`, `node007`, and all six CPU
  audit nodes; retries `t79390-t79397` then passed validation and entered the
  student pipeline. The local producer `t79315` was already valid and was not
  interrupted.
- Until their staging closure is available, `jtl110gpu2` and the currently
  down `jtl311linux` are excluded only from the remaining queued confirmation
  tasks. This changes placement, not seeds, commands, inputs, or analysis.

### Persistent-damping oracle-headroom screen

- Registration:
  `jax_experiments/deployments/regime_damping_headroom_v2/registration.json`.
- Tasks: `t79247-t79270` GPU training, `t79271-t79294` CPU audit, and
  `t79295` aggregate.
- The v1 producers failed in argument parsing before training; obsolete v1
  audits and aggregate `t79127-t79151` were cancelled.
- Runtime validation confirms fixed gravity, mode-specific persistent joint
  damping, equal perturbation norm, and common per-step actuator noise.

## Decision boundary

Do not expand the BAPR claim to sample efficiency or universal MuJoCo
superiority. Do not train a learned estimator for the new damping benchmark
unless the preregistered dynamic-oracle headroom gate passes. Preserve all
registration hashes, checkpoint-safe resume behavior, and producer-to-audit
file dependencies.
