# BAPR Server Deployment

All-in-one deployment for the BAPR experiment suite.

## TL;DR one-liner (on server)

```bash
tar xzf bapr_server.tar.gz && cd bapr_server && bash setup_env.sh && nohup bash jax_experiments/run_server.sh > run.log 2>&1 &
```

Then `tail -f run.log` to watch progress. You can disconnect SSH safely.

## What it runs

Six phases in sequence, auto-resuming on interruption:

| Phase | Content | # runs | ~Wall time |
|-------|---------|--------|------------|
| 1 | Multi-seed main (4 algo × 4 env × 5 seeds) | 80 | ~8-12h |
| 2 | Ablation + Bad-BAPR (HalfCheetah) | 5 | ~1h |
| 3 | Sensitivity heatmap (HR × PS 5×5) | 25 | ~2h |
| 4 | Key ablations on Hopper/Walker/Ant | 6 | ~1h |
| 5 | ID/OOD evaluation (loads checkpoints) | — | ~30min |
| 6 | Generate all paper figures | — | ~5min |

Assuming ~6-8 parallel slots (2×12GB GPUs shared with others).

## Auto-resource detection

`run_server.sh` queries `nvidia-smi` at startup and computes:
- Free VRAM per GPU
- Max parallel slots = `(free_vram - 1500MB safety) / 1800MB_per_run`
- Jobs pinned to specific GPU via `CUDA_VISIBLE_DEVICES`
- Scheduler picks least-loaded GPU for each new job

Override defaults with env vars:
```bash
PER_RUN_MB=2000 bash jax_experiments/run_server.sh   # tighter per-run cap
MAX_PARALLEL=6 bash jax_experiments/run_server.sh    # absolute cap
SEEDS_OVERRIDE="0 1 2" bash jax_experiments/run_server.sh  # fewer seeds
PHASES="6" bash jax_experiments/run_server.sh        # just re-gen figures
```

## Results

- `jax_experiments/results/<algo>_<env>_<seed>/logs/*.npy` — training metrics
- `jax_experiments/results/<algo>_<env>_<seed>/checkpoints/` — resumable state
- `paper/fig_*.pdf` — publication figures
- `paper/results_table.txt` — LaTeX-format main results table

## Resume after interruption

Just re-run the same command. Each run checks for existing `eval_reward.npy`
and skips if ≥90% complete; otherwise resumes from the last checkpoint.

## Environment setup (if conda not installed)

```bash
# Install Miniconda first
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/miniconda3/etc/profile.d/conda.sh

# Then run the one-liner above
```

`setup_env.sh` auto-detects CUDA version from `nvidia-smi` and installs
`jax[cuda12]` or `jax[cuda11_pip]` accordingly. If CUDA is unavailable,
falls back to CPU-only (very slow, not recommended).
