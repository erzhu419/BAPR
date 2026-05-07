#!/bin/bash
# Unified per-seed launcher for paper experiments.
# Args: ALGO ENV SEED [MAX_ITERS] [DWELL] [TARGET_MODE] [TAG]
# ALGO ∈ {bapr, escp, resac, sac, bapr_no_bocd, bapr_no_rmdm,
#         bapr_no_adapt_beta, bapr_fixed_decay, bad_bapr}
# TAG (optional): extra label for run_name (e.g. "no2" / "main")
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

ALGO=$1; ENV=$2; SEED=$3
MAX_ITERS=${4:-1500}; DWELL=${5:-60}; TGT=${6:-min}; TAG=${7:-paper}
RUN_NAME="${TAG}_${ALGO}_${ENV%-v2}_dw${DWELL}_s${SEED}"

cd /home/erzhu419/bapr_v15
mkdir -p jax_experiments/results_paper

# Algo-conditional flags. BAPR family uses regime/no-per-trans/target_mode.
EXTRA=""
case "$ALGO" in
  bapr|bapr_no_bocd|bapr_no_rmdm|bapr_no_adapt_beta|bapr_fixed_decay|bad_bapr)
    EXTRA="--use_regime_belief --penalty_scale 0.5 --critic_target_mode ${TGT} --no_per_trans_belief"
    ;;
  escp|resac|sac)
    EXTRA=""  # default config; no BAPR-specific flags
    ;;
esac

exec python -u -m jax_experiments.train \
  --algo "$ALGO" --seed "$SEED" --max_iters "$MAX_ITERS" \
  --log_interval 5 --eval_episodes 5 --save_interval 100 \
  --save_root jax_experiments/results_paper --backend spring \
  --env "$ENV" --env_type discrete_mode --mean_dwell_iters "$DWELL" \
  $EXTRA \
  --resume \
  --run_name "$RUN_NAME"
