#!/bin/bash
# Unified launcher with 2-stage kill + resume.
# Args: ENV MODE_VARIANT S1_KILL_ITER S1_KILL_REWARD ENV_SHORT MEAN_DWELL \
#       [SEED] [MAX_ITERS] [S2_KILL_ITER] [S2_KILL_REWARD]
# Stage 1 (e.g. iter 278, R 383): peak_R must clear or kill (filter dead-on-arrival)
# Stage 2 (e.g. iter 1000, R 2000): peak_R must clear or kill (must show real learning)
# Pass both → train to MAX_ITERS. Resumes from latest ckpt if RUN_NAME re-launched.
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

ENV=$1; VARIANT=$2; S1_KI=$3; S1_KR=$4; ENV_SHORT=$5
DWELL=${6:-200}; SEED=${7:-0}; MAX_ITERS=${8:-2000}
S2_KI=${9:-1000}; S2_KR=${10:-2000}
RUN_NAME="envsweep2_${ENV_SHORT}_${VARIANT}_dw${DWELL}_max${MAX_ITERS}_s${SEED}"

cd /home/erzhu419/bapr_v15
mkdir -p jax_experiments/results_envsweep2
# NEVER rm — checkpoints are precious. --resume picks up if ckpt exists.

exec python -u -m jax_experiments.train \
  --algo bapr --seed "$SEED" --max_iters "$MAX_ITERS" \
  --log_interval 5 --eval_episodes 5 --save_interval 100 \
  --save_root jax_experiments/results_envsweep2 --backend spring \
  --env "$ENV" --env_type discrete_mode --mean_dwell_iters "$DWELL" \
  --bapr_adaptation_mode gate --actor_objective mean \
  --penalty_scale 0.5 --critic_target_mode independent \
  --mode_variant "$VARIANT" \
  --early_kill_iter "$S1_KI" --early_kill_reward "$S1_KR" \
  --stage2_kill_iter "$S2_KI" --stage2_kill_reward "$S2_KR" \
  --resume \
  --run_name "$RUN_NAME"
