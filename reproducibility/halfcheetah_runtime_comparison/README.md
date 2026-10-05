# Runtime-matched HalfCheetah follow-up

This separate five-policy-seed cohort keeps the frozen compensation mechanism
and estimator while recording matching JAX 0.4.38, Brax 0.10.5, Flax 0.10.4 and
MuJoCo 3.1.6 versions for training and audits. It reuses the original event
streams and schedules, so it is not a new event holdout or a replacement for
the ten-seed preregistered composite decision.

- `preregistration.md`: the original frozen protocol, including its operator amendment.
- `analysis.json`: unchanged aggregate and primary/secondary contrast identities.
- `audits/seed_*/audit.json`: stationary and switching returns and coverage.
- `training/ARM/seed_*/complete.json`: all thirty training configurations,
  completed budgets, dependency versions, interpreter paths and hosts.

Final controllers receive 8.4M interactions and 525,000 updates. The robust
source receives 5.6M/350,000, followed by the reference fork's additional
2.8M/175,000. BAPR retains its separate historical estimator supervision; the
study matches controller budgets, not total end-to-end data or information.
No replay buffers, full checkpoints or new model downloads are included.

The primary contrast is BAPR minus the regularized JAX transfer recipe. That
recipe failed in this benchmark and is not a faithful replication of the
original RE-SAC implementation. Its five critics, network/optimization choices
and reward scale differ from that implementation. The historical RE-SAC-0 arm
also differs in more than the two regularization coefficients. The full and
zero recipes therefore cannot isolate a regularization effect. The secondary
SAC, recurrent-ESCP and RE-SAC-0 contrasts supply the useful runtime-matched
corroboration. All contrasts are retained, without merging historical cohorts
or interpreting a failed transfer recipe as evidence against the original paper.

From the repository root, `python scripts/check_manuscript_results.py` recomputes
the eight route means and four paired intervals from these audits and checks
the training/runtime records. The generation scripts are
`jax_experiments/analysis/regularized_comparison_protocol_v39.py` and
`jax_experiments/analysis/run_regularized_comparison_v39.py`.
