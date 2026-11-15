# Independent Reacher Confirmation

Five untouched policy seeds: 99007/99027/99047/99067/99087. This package contains
the preregistration, `analysis.json`, five per-seed `audits/seed_*/audit.json`,
and `frozen_estimator_metadata.json` recording the shared provider and fitting
cost. Earlier development and confirmation cohorts are not pooled here.
`previous_confirmation.json` preserves the earlier negative filter ablation;
`halfcheetah_information_diagnostic.json` preserves the separate inconclusive
same-information diagnostic. Neither contributes samples to the final result.

From the repository root, run `python paper/plot_reacher_confirmation.py` to
recompute the primary and secondary paired intervals and generate the manuscript
PDF/SVG/PNG and a small source-data CSV. The script reads only this JSON package
and requires matplotlib, NumPy and SciPy, not JAX or model checkpoints.
The means are recomputed from individual episode returns, not copied from cached
summaries. `python scripts/check_manuscript_results.py` additionally checks all
controller budgets, switching-window coverage and shared estimator accounting.

The primary endpoint is filtered compensation minus filtered belief-conditioned
SAC; instantaneous compensation minus instantaneous conditioned SAC is secondary.
Each seed averages 15 switching episodes. Statistical intervals use five policy
seeds, not 75 independent episodes. Stationary episodes are reported separately.

Each of four controllers per seed receives 1.6M interactions and 100,000 updates.
Compensation and conditioned SAC share the frozen inverse model and causal
belief definition. Both are charged the historical 1.6M provider interactions,
260,000 calibration transitions and 4500 fitting updates; memoryless robust
SAC is not. The conditioned actor has 70,916 parameters versus 69,892 for the
reference actor. Filtering and ensembling have no established separate return
gain in this cohort.

The matched runtime is JAX 0.4.38, Brax 0.10.5, Flax 0.10.4 and MuJoCo 3.1.6.
The generalized Reacher task uses persistent sign modes, noise standard
deviation 0.02, horizon 200 and dwell 50. These audits reproduce reported
statistics, not training or environment rollouts. Model parameters, raw
calibration arrays and full checkpoints are not included in this package.
