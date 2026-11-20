# Reproducing the manuscript results

The code and manuscript are on the `codex/code-paper` branch. Extract
`BAPR_JMLR_results.zip` at the checkout root to add the compact results
supplement. To recompute the reported seed-level results:

```sh
python -m pip install numpy scipy matplotlib
python scripts/check_manuscript_results.py
```

The supplement includes the historical HalfCheetah and Reacher audit
records, the identification comparisons, the Brax mechanism audit, and
the Gymnasium MuJoCo suite. The historical records retain their original
training and evaluation metadata. The MuJoCo `training/` manifests
record the six arms per seed and morphology, the runtime, and the
fresh-start training budgets. The V50 and V51 JSONs record seed
summaries, including termination and switch metrics. They do not include
the episode traces needed to independently recalculate each seed's
diagnostics.

To rerun an experiment, follow the corresponding protocol and runtime
instructions. See [V50](jmlr_brax_mechanism_v50/README.md) and
[V51](jmlr_mujoco_v51/README.md). Models, replay buffers and checkpoints
are not included. To check the formal results, run `lake build` from
`reproducibility/lean`.
