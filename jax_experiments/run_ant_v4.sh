#!/bin/bash
# Ant Re-run v4: gravity perturbation + baseline-subtracted λ_w
# 3 PARALLEL WORKERS — one per algorithm, Ant only
#
# KEY CHANGES (v4 vs v3):
#   - BAPR code: λ_w now uses baseline subtraction (already patched in bapr.py)
#   - penalty_scale = 5.0 (default, NOT v3's 2.5)
#   - Same gravity perturbation as v3 (log_scale=0.75)
#   - changing_period=40000 (10 iters/task, same as v3)
#
# Usage: bash jax_experiments/run_ant_v4.sh
# Monitor: tail -f jax_experiments/logs/*Ant*_v4.log

# set -e  # disabled: one worker crash should NOT kill others

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

SEED=8
MAX_ITERS=2000
SAVE_ROOT="jax_experiments/results"
BACKEND="spring"
LOG_SCALE=0.75            # gravity 0.48x~2.11x (same as v3)
VARYING="gravity"         # same as v3
CHANGE_PERIOD=40000       # 10 iters per task

ALGOS=("resac" "escp" "bapr")
ENV="Ant-v2"

echo "=============================================="
echo "  Ant Re-run v4 (baseline-subtracted λ_w)"
echo "  3 algos × 1 env = 3 workers"
echo "  Seed: $SEED | Max iters: $MAX_ITERS | Backend: $BACKEND"
echo "  varying_params: $VARYING"
echo "  log_scale_limit: $LOG_SCALE (gravity 0.48x~2.11x)"
echo "  changing_period: $CHANGE_PERIOD (10 iters/task)"
echo "  penalty_scale: 5.0 (default, BAPR only)"
echo "  BAPR fix: baseline-subtracted λ_w"
echo "  MEM_FRACTION: 0.12 per worker"
echo "  Run names use _v4 suffix"
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

run_single() {
    local algo=$1
    local worker_id=$2
    local run_name="${algo}_Ant-v2_8_v4"
    local log_file="jax_experiments/logs/${run_name}.log"

    echo "[W$worker_id] $algo/$ENV START — $(date '+%H:%M:%S')"

    nohup python -u -m jax_experiments.train \
        --algo "$algo" \
        --env "$ENV" \
        --seed "$SEED" \
        --max_iters "$MAX_ITERS" \
        --save_root "$SAVE_ROOT" \
        --run_name "$run_name" \
        --backend "$BACKEND" \
        --log_scale_limit "$LOG_SCALE" \
        --changing_period "$CHANGE_PERIOD" \
        --varying_params $VARYING \
        > "$log_file" 2>&1

    local EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[W$worker_id] ⚠️  $algo/$ENV FAILED (exit $EXIT_CODE)"
    else
        echo "[W$worker_id] ✅ $algo/$ENV DONE — $(date '+%H:%M:%S')"
    fi
}

echo ""
echo "Launching 3 workers with 15s stagger..."
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

PIDS=()
WORKER_ID=1
for algo in "${ALGOS[@]}"; do
    run_single "$algo" "$WORKER_ID" &
    PIDS+=($!)
    echo "  W$WORKER_ID ($algo/$ENV) launched: PID=${PIDS[-1]}"
    WORKER_ID=$((WORKER_ID + 1))
    sleep 15
done

echo ""
echo "All 3 workers running. Monitor with:"
echo "  tail -f jax_experiments/logs/*Ant*_v4.log"
echo "  watch -n5 'grep -h \"Iter\" jax_experiments/logs/*Ant*_v4.log | tail -6'"
echo ""

# Wait for all
FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"; CODE=$?
    [ $CODE -ne 0 ] && FAILED=$((FAILED + 1))
done

echo ""
echo "=============================================="
echo "  All 3 Ant experiments finished!"
echo "  Failed: $FAILED / 3"
echo "  End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
