#!/bin/bash
# Q2/Q3-Q4 ablation launcher (env-agnostic). Args:
#   ALGO SEED [MAX_ITERS] [ENV] [DWELL]
# ALGO ∈ {bapr_no_bocd, bapr_no_rmdm, bapr_no_adapt_beta,
#         bapr_fixed_decay, bad_bapr, bapr (full reference)}
set -e
for cdir in /home/erzhu419/miniconda3 /home/erzhu419/anaconda3 ~/.conda; do
    [ -f "$cdir/etc/profile.d/conda.sh" ] && { source "$cdir/etc/profile.d/conda.sh"; break; }
done
conda activate bapr-jax
NVIDIA_LIB_DIR=$(python -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))" 2>/dev/null || true)
if [ -n "$NVIDIA_LIB_DIR" ]; then
    for d in "$NVIDIA_LIB_DIR"/*/lib; do
        [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
    done
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40

ALGO=$1; SEED=${2:-0}; MAX_ITERS=${3:-1500}
ENV=${4:-HalfCheetah-v2}; DWELL=${5:-60}
RUN_NAME="ablation_${ALGO}_${ENV%-v2}_dw${DWELL}_s${SEED}"

cd /home/erzhu419/bapr_v15
mkdir -p jax_experiments/results_ablation_v2

exec python -u -m jax_experiments.train \
  --algo "$ALGO" --seed "$SEED" --max_iters "$MAX_ITERS" \
  --log_interval 5 --eval_episodes 5 --save_interval 100 \
  --save_root jax_experiments/results_ablation_v2 --backend spring \
  --env "$ENV" --env_type discrete_mode --mean_dwell_iters "$DWELL" \
  --penalty_scale 0.5 --critic_target_mode independent \
  --resume \
  --run_name "$RUN_NAME"
