# Safe-utility causal-delay diagnostic preregistration

This checkpoint-only diagnostic uses the five frozen v9 policy banks, their
stationary-calibrated robust/specialist utility maps, and the frozen expected-
action estimator. It trains no controller, estimator, or selector.

The unused switching seeds are `156201,156217,156233`. Every arm receives the
same mode stream and five 1,000-step episodes. The fixed arms are robust SAC,
zero-delay true-mode safe utility, posterior-MAP safe utility, stale-policy
delays of `1/2/4/8` actions, and robust handoff delays of `1/2/4/8` actions.
The initial regime is treated as an onset; stale delay uses robust SAC until a
previous regime exists.

A policy bank has usable headroom when zero-delay safe utility exceeds matched
robust SAC by at least 10%. A delay-4 control retains causal margin when the
better of stale-policy and robust-handoff delay retains at least 70% of that
headroom. Estimator retraining is authorized only if at least four of five
banks have usable headroom, at least four retain delay-4 causal margin, and
posterior-MAP recovers at least 70% on fewer than four banks.

If fewer than four banks have headroom, controller-bank capacity is the primary
failure. If headroom is reproducible but delay-4 retention is not, causal
switch latency consumes the available margin. Only reproducible delayed-oracle
headroom combined with weak posterior recovery supports further estimator work.

Only compact JSON audit results and the aggregate Markdown/JSON are synchronized
to the local workspace. Existing inference bundles are staged; training
checkpoints, replay buffers, optimizer state, and CSV traces are not collected.
