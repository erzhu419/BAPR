# Hopper and Walker2d structured-channel headroom screen

V27 tests whether Hopper and Walker2d provide survival-valid controller
adaptation headroom before applying the frozen V21 BAPR recipe. It reuses the
pre-existing `structured_channel` family without changing its parameters:
four persistent affected-channel patterns, positive impaired gain `0.45`,
per-step actuator noise standard deviation `0.04`, fixed 250-step dwell, and
unchanged robot physics.

Each environment uses robust and privileged true-mode oracle `RegimeSAC`
controllers with equal 5.6M-step and 350k-update budgets. Training seeds are
`86003, 86021, 86039`; strict evaluation uses disjoint event seeds
`193101, 193117, 193133`. This is an environment-capacity screen, not a BAPR
result.

An environment passes only if switching and worst-mode stationary gains are at
least 15%, all three policy seeds improve on both measures, at least three of
four stationary modes improve, the switching termination gap is at most five
percentage points, and absolute stationary and switching termination are at
most 10% for both roles. A passing environment alone is authorized for a
subsequent frozen-recipe BAPR transfer. A failed environment is retained
without tuning gain, noise, dwell, or BAPR on these seeds.

The registered scheduler DAG is `t92933-t92957`: 12 GPU training tasks, 12
dependency-gated CPU audits, and one aggregate. All 12 training tasks launched
on `jtl110gpu`, `jtl110gpu2`, `jtl311linux`, or `node007`; local is excluded.
The audits cannot launch before their corresponding compact policy bundle is
available, and the aggregate cannot launch before all audit manifests exist.
Only compact evaluation bundles and audit outputs are synchronized to the local
workspace; raw training checkpoints stay on their producer nodes.

## Result

Both environments fail the registered screen. Hopper obtains only `+1.6%`
oracle switching gain despite `+31.3%` worst-mode stationary gain, improves
only `2/4` mode means, and wins switching on `2/3` training seeds. Walker2d is
worse with oracle context: switching is `-2.1%`, worst-mode stationary return
is `-7.7%`, and only `2/4` mode means improve. Neither environment authorizes
a frozen BAPR transfer.

The decisive failure is survival, not context estimation. Both robust and
oracle terminate in 100% of stationary and switching evaluations. Across all
event streams, Hopper survives only 82.6 robust / 85.0 oracle stationary steps
on average and Walker2d 83.9 / 95.7. All 45 switching episodes per role and
environment terminate before the first scheduled switch at step 250. These
tasks therefore cannot support a causal adaptation claim.

The original audits were mistakenly placed on Windows `jtl110cpu` workers and
failed because their commands use POSIX environment-prefix syntax. Linux-only
retries `t92994-t93005` then exposed a validation-only omission: robust zero
context is recorded as action mode `-1`, but V27 had not declared that sentinel.
The amendment entry points set the sentinel at runtime without changing any
rollout, checkpoint, event seed, or registered source. Valid robust audits are
`t93018-t93023`; aggregate retry `t93031` completed with
`V27_HEADROOM=FAIL`.

Per the frozen stopping rule, gain, noise, dwell, and BAPR will not be tuned on
these seeds. Hopper and Walker2d remain survival-negative evidence under this
structured-channel protocol.

The four-environment evidence boundary, including the frozen HalfCheetah result
and Ant controller-development result, is consolidated in
`reports/BAPR_FOUR_ENVIRONMENT_EVIDENCE_2026-09-12.md`.
