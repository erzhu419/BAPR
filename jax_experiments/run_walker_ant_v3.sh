#!/bin/bash
# Walker2d + Ant Re-run v3: Fix gravity sampling + task switching
# 6 PARALLEL WORKERS — one per (algo, env) combination
#
# ROOT CAUSE FIX (v3):
#   - log_scale_limit=0.75 → gravity 0.48x~2.11x (0 instant-death tasks, was 15/40)
#   - changing_period=40000 → 10 iters/task (was 20000 = 5 iters/task)
#   - penalty_scale=2.5 for BAPR (unchanged)
#
# Run names use _v3 suffix → will NOT overwrite any existing results.
#
# Usage: bash jax_experiments/run_walker_ant_v3.sh
# Monitor: tail -f jax_experiments/logs/*_v3.log

set -e

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.12
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

SEED=8
MAX_ITERS=2000
SAVE_ROOT="jax_experiments/results"
BACKEND="spring"
LOG_SCALE=0.75      # gravity 0.48x~2.11x — no instant-death tasks
PENALTY_SCALE=2.5   # BAPR only
CHANGE_PERIOD=40000 # 10 iters per task (was 5)

ALGOS=("resac" "escp" "bapr")
ENVS=("Walker2d-v2" "Ant-v2")

TOTAL=$((${#ALGOS[@]}*${#ENVS[@]}))

echo "=============================================="
echo "  Walker2d + Ant Re-run v3 (6 parallel workers)"
echo "  ${#ALGOS[@]} algos × ${#ENVS[@]} envs = $TOTAL workers"
echo "  Seed: $SEED | Max iters: $MAX_ITERS | Backend: $BACKEND"
echo "  log_scale_limit: $LOG_SCALE (gravity 0.48x~2.11x)"
echo "  changing_period: $CHANGE_PERIOD (10 iters/task)"
echo "  penalty_scale: $PENALTY_SCALE (BAPR only)"
echo "  MEM_FRACTION: 0.12 per worker (6 × 0.12 = 0.72 GPU)"
echo "  Run names use _v3 suffix → existing results safe"
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
    local env=$2
    local worker_id=$3

    local extra_args="--log_scale_limit $LOG_SCALE --changing_period $CHANGE_PERIOD"
    if [ "$algo" = "bapr" ]; then
        extra_args="$extra_args --penalty_scale $PENALTY_SCALE"
    fi

    local run_name="${algo}_${env}_${SEED}_v3"
    local log_file="jax_experiments/logs/${run_name}.log"

    echo "[W$worker_id] $algo/$env START — $(date '+%H:%M:%S')"

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

    local EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[W$worker_id] ⚠️  $algo/$env FAILED (exit $EXIT_CODE)"
    else
        echo "[W$worker_id] ✅ $algo/$env DONE — $(date '+%H:%M:%S')"
    fi
}

echo ""
echo "Launching 6 workers with 15s stagger..."
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

PIDS=()
WORKER_ID=1
for algo in "${ALGOS[@]}"; do
    for env in "${ENVS[@]}"; do
        run_single "$algo" "$env" "$WORKER_ID" &
        PIDS+=($!)
        echo "  W$WORKER_ID ($algo/$env) launched: PID=${PIDS[-1]}"
        WORKER_ID=$((WORKER_ID + 1))
        sleep 15  # stagger: avoid GPU OOM during concurrent JIT compilation
    done
done

echo ""
echo "All $TOTAL workers running. Monitor with:"
echo "  tail -f jax_experiments/logs/*_v3.log"
echo "  watch -n5 'grep -h \"Iter\" jax_experiments/logs/*_v3.log | tail -12'"
echo "  nvidia-smi -l 5"
echo ""

# Wait for all
FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"; CODE=$?
    [ $CODE -ne 0 ] && FAILED=$((FAILED + 1))
done

echo ""
echo "=============================================="
echo "  All $TOTAL experiments finished!"
echo "  Failed: $FAILED / $TOTAL"
echo "  Results: $SAVE_ROOT/"
echo "  End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
