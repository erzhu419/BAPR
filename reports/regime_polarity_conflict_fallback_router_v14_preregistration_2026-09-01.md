# Evidence-conflict fallback router preregistration

This development audit freezes the v12 controller banks, their utility maps,
and the v5 expected-action estimator. It performs no training. Three unused
event seeds provide explicit, distinct four-mode switching schedules.

The candidate router never receives mode ID, switch clock, dwell boundary, or
action gain. It starts with robust SAC. While an expert is active, it enters
robust fallback when the best alternative mode's one-step log likelihood
exceeds the active mode by at least 3.0. This threshold was frozen after one
local development replay: the stable-regime 99th percentile was 1.84, while
switch-local margins commonly reached 3-7. The router executes at least one
robust action after a conflict. It exits fallback only when raw evidence and
the filtered posterior agree, posterior confidence is at least 0.80, and that
agreement persists for 1, 2, or 3 consecutive transitions depending on the
registered candidate.

Matched robust SAC, privileged true-mode safe utility, privileged four-step
robust handoff, and unmodified frozen-v5 posterior MAP are controls. A candidate
passes a seed only if it gains at least 10% over robust, recovers at least 70%
of safe-oracle headroom, wins all three event streams, and has zero termination.
Candidates are ranked by seed-pass count, mean return, lower robust-fallback
fraction, then fewer confirmation steps. At least four of five seed passes are
required before a separately registered confirmation; otherwise switch-focused
estimator retraining is the next permitted step.
