# Switch-weighted expected-action estimator development preregistration

V16 keeps the confirmed v12 policy banks, robust-inclusive utility maps,
environment, and plain posterior-MAP decision rule fixed. It initializes from
the frozen v5 inverse model and changes only the evidence model training loss
and trajectory distribution. Policy seeds 71003/71021/71039 are used for
training; 71057/71079 are held out from model updates. Training, validation,
and audit event seeds are disjoint from each other and all prior phases.

Each training trajectory is generated causally by the frozen v5 MAP router.
The first eight transitions of every regime are weighted by 8; all other
transitions have unit weight. The loss is weighted executed-action MSE plus
0.10 times weighted true-mode evidence cross-entropy. The v5 sticky-HMM filter
configuration is unchanged. True mode is available only offline for targets
and metrics; deployed inference still consumes observation, commanded action,
and next observation.

The development result passes only if fresh safe-oracle headroom and the
four-step causal margin each reproduce on at least four of five policy seeds,
v16 MAP meets the strict robust-relative gain/recovery/all-event/zero-
termination gate on at least four seeds, and v16 MAP beats frozen-v5 MAP on at
least four seeds with a positive paired mean difference. It must additionally
beat frozen-v5 MAP on both held-out policy seeds. Passing authorizes a new
policy-bank confirmation; failure closes further loss/filter tuning on this
inverse-dynamics estimator family.
