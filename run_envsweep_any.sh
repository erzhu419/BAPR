#!/bin/bash
# Unified launcher: auto-detects local (anaconda3) vs server (miniconda3) conda.
# Args: ENV MODE_VARIANT KILL_ITER KILL_REWARD ENV_SHORT
set -e
if [ -d "$HOME/miniconda3/etc/profile.d" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -d "$HOME/anaconda3/etc/profile.d" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: no conda found"; exit 1
fi
conda activate bapr-jax

NVIDIA_LIB_DIR=$(python -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))" 2>/dev/null || true)
if [ -n "$NVIDIA_LIB_DIR" ]; then
    for d in "$NVIDIA_LIB_DIR"/*/lib; do
        [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
    done
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40

ENV=$1; VARIANT=$2; KILL_ITER=$3; KILL_REWARD=$4; ENV_SHORT=$5
HOST_TAG=$(hostname -s | tr -cd 'A-Za-z0-9' | head -c 12)
RUN_NAME="envsweep_${ENV_SHORT}_${VARIANT}_${HOST_TAG}"
cd /home/erzhu419/bapr_v15
mkdir -p jax_experiments/results_envsweep
rm -rf "jax_experiments/results_envsweep/${RUN_NAME}"
exec python -u -m jax_experiments.train \
  --algo bapr --seed 0 --max_iters 800 \
  --log_interval 5 --eval_episodes 5 --save_interval 100 \
  --save_root jax_experiments/results_envsweep --backend spring \
  --env "$ENV" --env_type discrete_mode --mean_dwell_iters 200 \
  --bapr_adaptation_mode gate --actor_objective mean \
  --penalty_scale 0.5 --critic_target_mode independent \
  --mode_variant "$VARIANT" \
  --early_kill_iter "$KILL_ITER" --early_kill_reward "$KILL_REWARD" \
  --run_name "$RUN_NAME"
