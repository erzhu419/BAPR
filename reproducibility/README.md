# Reproducing the manuscript results

The code and manuscript are on the `codex/code-paper` branch. Download the
[versioned results supplement](https://github.com/erzhu419/BAPR/raw/refs/tags/jmlr-results-v2/BAPR_JMLR_results.zip)
to the checkout root. Extract it without overwriting the checkout's
instructions. The following commands use Python 3.11 and the verified
audit dependencies:

```sh
unzip -nq BAPR_JMLR_results.zip
python3.11 -m venv .venv-audit
. .venv-audit/bin/activate
python -m pip install -r reproducibility/requirements-audit.txt
python scripts/check_manuscript_results.py
python scripts/check_jmlr_episode_results.py
```

The supplement includes the historical HalfCheetah and Reacher audit
records, the identification comparisons, the Brax mechanism audit, and
the Gymnasium MuJoCo suite. It preserves scientific and budget fields
without worker-specific paths. The MuJoCo `training/` manifests record
six arms per seed and morphology, runtime, and fresh-start training
budgets.

The V50 and V51 `episodes/` records come from new evaluations of the
frozen models, separate from the original aggregate and seed audits.
Tabulated replay diagnostics can be checked from episode returns,
termination and switch records. Replays on different hardware or
runtime builds need not produce identical trajectories. The
`jmlr_frozen_interface/` records contain the paired hard/soft/true-mode
and gain-mismatch experiments. No full state/action trajectories are
included. The episode checker recomputes the matched Walker2d switch
counts and paired gain-oracle intervals from these records.

To rerun an experiment, follow the corresponding protocol and runtime
instructions. See [V50](jmlr_brax_mechanism_v50/README.md) and
[V51](jmlr_mujoco_v51/README.md). Model weights, replay buffers and checkpoints
are not included. To check the formal results, run `lake build` from
`reproducibility/lean`.
