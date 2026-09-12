# Persistent-damping v3 evaluation amendment

The v2 producer checkpoints and all oracle audits are frozen and reused. No
controller is retrained and no scientific threshold, seed, event stream,
environment, training budget, or evaluation budget changes.

All 12 v2 robust audits completed rollout and then failed the same trace check.
The robust controller correctly used the sealed all-zero context, represented
in `switching_trace.csv` as `action_task_id=-1`. The generic validator treated
that trace as a dynamic oracle trace because v2 did not declare the existing
robust sentinel.

The v3 amendment declares `ROBUST_TRACE_CONTEXT_MODE_ID=-1`, reruns only the 12
failed CPU robust audits into a new output namespace, and aggregates them with
the immutable v2 oracle audits. The superseded v2 aggregate `t79295` must not be
used for a scientific conclusion.
