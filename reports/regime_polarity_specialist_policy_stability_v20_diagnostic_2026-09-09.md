# V20 specialist policy-stability post-run diagnostic

All V20 scientific cells are complete. The scheduler has 49 completed records
and two failed specialist records whose unchanged signatures were completed by
replacement tasks. All 36 specialist bundles, nine audit manifests, and the
aggregate are present.

## Frozen aggregate result

The registered aggregate selected no candidate. Relative to
`full_state_final`, `full_state_best` lost 860.1 switching-return points on
average and `period2_best` lost 1168.9. Their minimum stationary retentions
were 33.9% and 29.2%, respectively. This frozen result is retained unchanged.

`full_state_final` itself is valid and strong development evidence. Across the
three new seeds it passed all three bank-level cells, all 12 stationary
seed-mode cells, and all nine switching events. Mean safe switching return was
3182.3 versus 1510.8 for robust SAC, a paired improvement of 1671.5. The
worst seed still improved by 96.1%, and the smallest stationary mode gain was
27.8%. Combined with the three V19 full-state development seeds, full-state
final training has now beaten robust SAC on all six policy seeds and all 18
switching-event comparisons; the six-seed mean paired improvement is 1437.8.

## Execution deviation in the two selection arms

The two `best` arms did not implement the preregistered fixed-mode validation
objective. The common trainer passed all four `test_tasks` to
`evaluate_stationary`. For a task list of length four, that function explicitly
rotates through every mode, calls `env.set_task` for each one, and reports their
pooled mean. `StochasticModeEnv.set_task` activates the requested mode even
when the training environment was constructed with a fixed mode ID. The scalar
fed to `SACPolicyStability.report_eval` was therefore cross-mode average return,
not return in the specialist's matching fixed mode.

The selection records show the practical consequence. All eight selected
policies for seed 83003, and most selected policies in the other two seeds,
came from the first post-fork evaluation at update 350250. Those snapshots are
essentially the generic robust controller before mode specialization. The
stationary audit then correctly rejected many of them or routed back to the
robust controller.

Accordingly:

- The negative `full_state_best` result does not test matching-mode checkpoint
  selection.
- `period2_best` confounds actor-period thinning with the same invalid selector,
  so it does not isolate actor-update frequency either.
- The unaffected `full_state_final` arm remains valid.
- The frozen V20 preregistered decision is still a failure; it is not rewritten
  after seeing the holdout. V21 instead treats V19 and V20 as development data,
  freezes full-state final training, and uses entirely new seeds and event
  streams for confirmation.

## Next experiment

Run one independent five-seed equal-policy-budget confirmation of the frozen
full-state-final policy bank with the unchanged frozen-v5 causal posterior.
Compare against matched robust SAC, recurrent ESCP, released-B0 RE-SAC, and a
causal SAC5 bank trained with the same number of policies. No checkpoint
selection, actor-period change, posterior tuning, environment change, or reuse
of V18-V20 holdouts is allowed.
