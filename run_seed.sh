#!/bin/bash
# Unified per-seed launcher for paper experiments.
# Args: ALGO ENV SEED [MAX_ITERS] [DWELL] [TARGET_MODE] [TAG]
# ALGO ∈ {bapr, escp, resac, sac, bapr_no_bocd, bapr_no_rmdm,
#         bapr_no_adapt_beta, bapr_fixed_decay, bad_bapr,
#         bapr_unsupervised}
# TAG (optional): extra label for run_name (e.g. "no2" / "main")
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d /cm/local/apps/cuda-driver/libs/535.261.03/lib64 ]; then
    export LD_LIBRARY_PATH="/cm/local/apps/cuda-driver/libs/535.261.03/lib64:${LD_LIBRARY_PATH:-}"
fi
if [ -d /cm/local/apps/cuda-driver/libs/535.261.03/bin ]; then
    export PATH="/cm/local/apps/cuda-driver/libs/535.261.03/bin:${PATH:-}"
fi
if [ -n "${BAPR_PYTHON:-}" ]; then
    PYTHON="$BAPR_PYTHON"
else
    for py in \
        /home/zhengliang01/scheduleurm_work/conda_envs/resac-jax-535-py310-final/bin/python \
        /home/erzhu419/anaconda3/envs/bapr-jax/bin/python \
        /home/erzhu419/miniconda3/envs/bapr-jax/bin/python \
        /home/erzhu419/.conda/envs/bapr-jax/bin/python \
        /home/erzhu419/resac_envs/resac-jax-535-py310-final/bin/python \
        /home/erzhu419/.conda/envs/resac-jax/bin/python \
        /home/erzhu419/miniconda3/envs/resac-jax/bin/python; do
        [ -x "$py" ] && { PYTHON="$py"; break; }
    done
fi
if [ -z "${PYTHON:-}" ]; then
    for cdir in /home/erzhu419/miniconda3 /home/erzhu419/anaconda3 ~/.conda; do
        [ -f "$cdir/etc/profile.d/conda.sh" ] && { source "$cdir/etc/profile.d/conda.sh"; break; }
    done
    conda activate bapr-jax
    PYTHON=python
fi
NVIDIA_LIB_DIR=$("$PYTHON" -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))" 2>/dev/null || true)
if [ -n "$NVIDIA_LIB_DIR" ]; then
    for d in "$NVIDIA_LIB_DIR"/*/lib; do
        [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
    done
fi
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.40}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

ALGO=$1; ENV=$2; SEED=$3
MAX_ITERS=${4:-1500}; DWELL=${5:-60}; TGT=${6:-independent}; TAG=${7:-paper}
EXTRA_ARGS=("${@:8}")
RUN_NAME="${TAG}_${ALGO}_${ENV%-v2}_dw${DWELL}_s${SEED}"
RUN_DIR="jax_experiments/results_paper/${RUN_NAME}"

cd "${BAPR_ROOT:-$SCRIPT_DIR}"
mkdir -p jax_experiments/results_paper

# Algo-conditional flags. BAPR family uses regime belief, per-transition
# replay belief, and independent critic targets by default.
EXTRA=""
BAPR_PENALTY_SCALE_VALUE="${BAPR_PENALTY_SCALE:-0.5}"
case "$ALGO" in
  bapr|bapr_no_bocd|bapr_no_rmdm|bapr_no_adapt_beta|bapr_fixed_decay|bad_bapr)
    EXTRA="--penalty_scale ${BAPR_PENALTY_SCALE_VALUE} --critic_target_mode ${TGT}"
    if [ "${BAPR_BELIEF_MODE:-joint}" = "joint" ]; then
      EXTRA="$EXTRA --use_regime_belief"
    fi
    if [ "${BAPR_PER_TRANS_BELIEF:-1}" = "0" ]; then
      EXTRA="$EXTRA --no_per_trans_belief"
    fi
    if [ -n "${BAPR_GRAD_CLIP_NORM:-}" ]; then
      EXTRA="$EXTRA --bapr_grad_clip_norm ${BAPR_GRAD_CLIP_NORM}"
    fi
    if [ -n "${BAPR_BETA:-}" ]; then
      EXTRA="$EXTRA --beta ${BAPR_BETA}"
    fi
    if [ -n "${BAPR_ACTOR_OBJECTIVE:-}" ]; then
      EXTRA="$EXTRA --actor_objective ${BAPR_ACTOR_OBJECTIVE}"
    fi
    if [ "${BAPR_BELIEF_CONDITIONED:-1}" = "0" ]; then
      EXTRA="$EXTRA --no_belief_conditioning"
    fi
    ;;
  bapr_unsupervised)
    EXTRA="--penalty_scale ${BAPR_PENALTY_SCALE_VALUE} --critic_target_mode ${TGT}"
    ;;
  escp|resac|sac)
    EXTRA=""  # default config; no BAPR-specific flags
    ;;
esac
if [ "${ORACLE_RESET:-0}" = "1" ]; then
  EXTRA="$EXTRA --oracle_reset_on_switch"
fi

"$PYTHON" -u -m jax_experiments.train \
  --algo "$ALGO" --seed "$SEED" --max_iters "$MAX_ITERS" \
  --log_interval 5 --eval_episodes 5 --save_interval 100 \
  --save_root jax_experiments/results_paper --backend spring \
  --env "$ENV" --env_type discrete_mode --mean_dwell_iters "$DWELL" \
  $EXTRA \
  "${EXTRA_ARGS[@]}" \
  --resume \
  --run_name "$RUN_NAME"
status=$?
if [ "$status" -eq 0 ]; then
  printf 'done run=%s\n' "$RUN_NAME" > "${RUN_DIR}/done.ok"
fi
exit "$status"
