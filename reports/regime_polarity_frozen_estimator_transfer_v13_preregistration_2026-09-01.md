# Frozen-estimator transfer preregistration

This checkpoint-only audit freezes the five v12 actor-only warm-start policy
banks, their robust-inclusive utility maps, and the v5 expected-action mode
estimator. It performs no policy, critic, estimator, or calibration update.

Each policy seed is evaluated on three unused event seeds with three explicit,
distinct four-mode schedules. Every arm receives the same mode trace within an
event. The compared arms are matched robust SAC, privileged true-mode safe
utility, a privileged four-step robust handoff followed by true-mode safe
utility, and the causal frozen-v5 posterior-MAP safe-utility router.

A seed has deployable causal margin only when the safe oracle gains at least
10% over robust on average, wins all three events, has zero termination, and
the four-step delayed oracle retains at least 70% of that margin while also
winning all three events. The frozen estimator passes a seed only when it gains
at least 10%, recovers at least 70% of safe-oracle headroom, reaches at least
95% mode accuracy, wins all three events, and has zero termination.

The phase is confirmed at four of five passing seeds. Estimator retraining is
authorized only if causal margin passes at four of five seeds but the frozen
estimator passes fewer than four. If four-step delayed oracle itself fails the
four-seed gate, estimator work stops because the policy bank lacks reproducible
causal margin under the tested switching protocol.
