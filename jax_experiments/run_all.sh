#!/bin/bash
# Launch 3 algorithms × 4 environments = 12 experiments
# Strategy: 3 parallel workers (one per algorithm), each runs 4 envs sequentially
# GPU: 8GB, ~400MB/worker steady + JIT peaks → 3 workers safe with staggered start
#
# Usage: bash jax_experiments/run_all.sh
# Monitor: tail -f jax_experiments/logs/*.log
# Kill all: pkill -f "jax_experiments.train"

set -e

# Limit each JAX process to ~25% of GPU memory (8GB / ~3 workers + headroom)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

SEED=8
MAX_ITERS=2000
SAVE_ROOT="jax_experiments/results"
BACKEND="spring"

ALGOS=("resac" "escp" "bapr")
ENVS=("Hopper-v2" "HalfCheetah-v2" "Walker2d-v2" "Ant-v2")

TOTAL=$((${#ALGOS[@]}*${#ENVS[@]}))

echo "=============================================="
echo "  JAX RL Experiments: ${#ALGOS[@]} algos × ${#ENVS[@]} envs = $TOTAL runs"
echo "  Seed: $SEED, Max iters: $MAX_ITERS, Backend: $BACKEND"
echo "  Mode: 3 PARALLEL WORKERS (one per algo)"
echo "  Estimated time: ~1.6h per experiment, ~6.5h total"
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

# Worker function: runs all envs for a given algorithm
run_algo() {
    local algo=$1
    local worker_id=$2
    
    for env in "${ENVS[@]}"; do
        run_name="${algo}_${env}_${SEED}"
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

# Launch 3 workers with staggered start (30s gap to avoid JIT peak collision)
echo ""
echo "Starting workers with 30s stagger to avoid JIT peak memory collision..."
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

run_algo "resac" 1 &
PID1=$!
echo "  Worker 1 (resac) launched: PID=$PID1"

sleep 30  # Stagger: let worker 1 finish JIT before worker 2 starts

run_algo "escp" 2 &
PID2=$!
echo "  Worker 2 (escp) launched: PID=$PID2"

sleep 30

run_algo "bapr" 3 &
PID3=$!
echo "  Worker 3 (bapr) launched: PID=$PID3"

echo ""
echo "All 3 workers running. Monitor with:"
echo "  tail -f jax_experiments/logs/*.log"
echo "  nvidia-smi -l 5"
echo ""

# Wait for all workers
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
