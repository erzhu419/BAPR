# Fresh-policy-bank estimator confirmation preregistration

V17 is an independent confirmation of the frozen v16 switch-weighted inverse
estimator. It uses five policy seeds that were not used by v12-v16. For every
seed, a robust SAC source is trained from scratch and four fixed-mode
specialists are initialized from only the matched robust actor. Critic, target
critic, alpha, optimizers, and replay are reset exactly as in the confirmed v12
training protocol. Neither v5 nor v16 estimator parameters are updated.

Calibration, stationary holdout, and switching streams are new and mutually
disjoint. Calibration constructs the same robust-inclusive utility map used in
v12-v16. Switching compares robust SAC, true-mode safe utility, four-step
delayed oracle, frozen-v5 posterior MAP, and frozen-v16 posterior MAP under the
same explicit mode sequence and random stream. Deployed estimator arms consume
only observation, commanded action, and next observation.

The confirmation passes only if at least four of five fresh policy banks pass
the existing actor-only policy-bank gate, at least four have reproducible
safe-oracle headroom and four-step causal retention, and v16 meets the strict
robust-relative gain/recovery/all-event/zero-termination gate on at least four
seeds. V16 must beat v5 on at least four seeds, have a positive paired mean, and
have a strictly positive seed-level 95% confidence-interval lower bound.

Mechanism evidence is a co-primary gate: on at least four seeds, v16 must not
reduce switch-window routing accuracy and must not increase the
wrong-specialist action fraction relative to v5. Passing all gates promotes the
frozen v16 estimator for final baseline comparison. Failure closes the current
executed-action inverse-estimator family; it does not authorize another loss,
filter, fallback, or seed-selection sweep on these confirmation results.
