#!/bin/bash
# Re-run Walker2d + Ant with tuned parameters:
#   - log_scale_limit=1.5 (gravity range: 0.22x ~ 4.5x, was 0.05x ~ 20x)
#   - penalty_scale=2.5 for BAPR (was 5.0)
#
# Run names use _v2 suffix → will NOT overwrite Hopper/HalfCheetah results.
#
# Usage: bash jax_experiments/run_walker_ant.sh
# Monitor: tail -f jax_experiments/logs/*_v2.log

set -e

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

SEED=8
MAX_ITERS=2000
SAVE_ROOT="jax_experiments/results"
BACKEND="spring"
LOG_SCALE=1.5       # was 3.0 → gravity now 0.22x–4.5x instead of 0.05x–20x
PENALTY_SCALE=2.5   # was 5.0 → BAPR β_eff less aggressive

ALGOS=("resac" "escp" "bapr")
ENVS=("Walker2d-v2" "Ant-v2")

TOTAL=$((${#ALGOS[@]}*${#ENVS[@]}))

echo "=============================================="
echo "  Walker2d + Ant Re-run: ${#ALGOS[@]} algos × ${#ENVS[@]} envs = $TOTAL runs"
echo "  Seed: $SEED, Max iters: $MAX_ITERS, Backend: $BACKEND"
echo "  log_scale_limit: $LOG_SCALE (was 3.0)"
echo "  penalty_scale: $PENALTY_SCALE (was 5.0, BAPR only)"
echo "  Mode: 3 PARALLEL WORKERS (one per algo)"
echo "  Run names use _v2 suffix → existing results safe"
echo "=============================================="

mkdir -p jax_experiments/logs

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate jax-rl

# Setup CUDA libs
NVIDIA_LIB_DIR=$(python -c "import nvidia; import os; print(os.path.dirname(nvidia.__file__))" 2>/dev/null || true)
if [ -n "$NVIDIA_LIB_DIR" ]; then
    for d in "$NVIDIA_LIB_DIR"/*/lib; do
        [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH}"
    done
fi

run_algo() {
    local algo=$1
    local worker_id=$2

    # Build extra args
    local extra_args="--log_scale_limit $LOG_SCALE"
    if [ "$algo" = "bapr" ]; then
        extra_args="$extra_args --penalty_scale $PENALTY_SCALE"
    fi

    for env in "${ENVS[@]}"; do
        run_name="${algo}_${env}_${SEED}_v2"
        log_file="jax_experiments/logs/${run_name}.log"

        echo "[Worker $worker_id/$algo] Starting: $env — $(date '+%H:%M:%S')"

        python -u -m jax_experiments.train \
            --algo "$algo" \
            --env "$env" \
            --seed $SEED \
            --max_iters $MAX_ITERS \
            --save_root "$SAVE_ROOT" \
            --run_name "$run_name" \
            --backend "$BACKEND" \
            $extra_args \
            > "$log_file" 2>&1

        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "[Worker $worker_id/$algo] ⚠️  FAILED on $env (exit $EXIT_CODE)"
        else
            echo "[Worker $worker_id/$algo] ✅ $env — $(date '+%H:%M:%S')"
        fi
    done
    echo "[Worker $worker_id/$algo] 🏁 ALL ENVS DONE — $(date '+%H:%M:%S')"
}

echo ""
echo "Starting workers with 30s stagger..."
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

run_algo "resac" 1 &
PID1=$!
echo "  Worker 1 (resac) launched: PID=$PID1"

sleep 30

run_algo "escp" 2 &
PID2=$!
echo "  Worker 2 (escp) launched: PID=$PID2"

sleep 30

run_algo "bapr" 3 &
PID3=$!
echo "  Worker 3 (bapr) launched: PID=$PID3"

echo ""
echo "All 3 workers running. Monitor with:"
echo "  tail -f jax_experiments/logs/*_v2.log"
echo "  nvidia-smi -l 5"
echo ""

wait $PID1; E1=$?
wait $PID2; E2=$?
wait $PID3; E3=$?

echo ""
echo "=============================================="
echo "  All $TOTAL experiments finished!"
echo "  Worker 1 (resac): exit=$E1"
echo "  Worker 2 (escp):  exit=$E2"
echo "  Worker 3 (bapr):  exit=$E3"
echo "  Results: $SAVE_ROOT/"
echo "  End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
