#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  BAPR Priority Experiments — Minimum Viable Paper
#  Run this first to get all paper figures with 1 seed (seed=8).
#  Then run run_all_experiments.sh phase 2 for multi-seed statistics.
# ═══════════════════════════════════════════════════════════════════════════
set -e

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10

# CUDA libs from pip-installed nvidia packages (needed for JAX GPU)
NVIDIA_ROOT="/home/erzhu419/anaconda3/envs/jax-rl/lib/python3.11/site-packages/nvidia"
CUDA_LIB_DIRS=""
for subdir in "$NVIDIA_ROOT"/*/lib; do
    [ -d "$subdir" ] && CUDA_LIB_DIRS="${CUDA_LIB_DIRS:+$CUDA_LIB_DIRS:}$subdir"
done
export LD_LIBRARY_PATH="${CUDA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"

# RTX 4060 8GB: 4 parallel with reduced per-run workload
MAX_PARALLEL=4

PYTHON="conda run --no-capture-output -n jax-rl python -u"
MODULE="jax_experiments.train"
SAVE_ROOT="jax_experiments/results"
BACKEND="spring"
MAX_ITERS=1000
LOG_INTERVAL=10
EVAL_EPISODES=3
SAMPLES_PER_ITER=2000
UPDATES_PER_ITER=125
SEED=8

declare -A ENV_PARAMS
ENV_PARAMS["HalfCheetah-v2"]="gravity"
ENV_PARAMS["Hopper-v2"]="gravity"
ENV_PARAMS["Walker2d-v2"]="dof_damping body_mass"
ENV_PARAMS["Ant-v2"]="gravity"

declare -A ENV_SCALE
ENV_SCALE["HalfCheetah-v2"]=0.75
ENV_SCALE["Hopper-v2"]=0.75
ENV_SCALE["Walker2d-v2"]=1.5
ENV_SCALE["Ant-v2"]=0.75

mkdir -p jax_experiments/logs

run() {
    local ALGO=$1 ENV=$2 TAG=$3 EXTRA=$4
    local PARAMS="${ENV_PARAMS[$ENV]}"
    local SCALE="${ENV_SCALE[$ENV]}"
    local NAME="${ALGO}_${ENV}_${SEED}"
    [ -n "$TAG" ] && NAME="${NAME}_${TAG}"

    echo ">>> Starting: $NAME"
    $PYTHON -m $MODULE \
        --algo $ALGO --env $ENV --seed $SEED \
        --max_iters $MAX_ITERS \
        --log_interval $LOG_INTERVAL \
        --eval_episodes $EVAL_EPISODES \
        --samples_per_iter $SAMPLES_PER_ITER \
        --updates_per_iter $UPDATES_PER_ITER \
        --save_root $SAVE_ROOT --run_name $NAME \
        --backend $BACKEND \
        --log_scale_limit $SCALE \
        --changing_period 20000 \
        --varying_params $PARAMS \
        $EXTRA \
        > "jax_experiments/logs/${NAME}.log" 2>&1
    echo "<<< Finished: $NAME"
}

# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1: Main results — 3 algorithms × 4 environments (12 runs)
#          Paper: training curves (Q1) + main table
# ═══════════════════════════════════════════════════════════════════════════
step1_main() {
    echo "╔═══════════════════════════════════════════════╗"
    echo "║  STEP 1: Main Results (3 algo × 4 env)       ║"
    echo "╚═══════════════════════════════════════════════╝"

    for ENV in "HalfCheetah-v2" "Hopper-v2" "Walker2d-v2" "Ant-v2"; do
        for ALGO in "bapr" "escp" "resac"; do
            run $ALGO $ENV "" &
            sleep 3
            # Throttle to MAX_PARALLEL
            while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
                sleep 15
            done
        done
    done
    wait
    echo "Step 1 complete!"
}

# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2: Ablation + Bad-BAPR on HalfCheetah (6 runs)
#          Paper: ablation study (Q2) + Bad-BAPR divergence (Q4)
# ═══════════════════════════════════════════════════════════════════════════
step2_ablation() {
    echo "╔═══════════════════════════════════════════════╗"
    echo "║  STEP 2: Ablations + Bad-BAPR (HalfCheetah)  ║"
    echo "╚═══════════════════════════════════════════════╝"

    local ENV="HalfCheetah-v2"
    for ABL in "bapr_no_bocd" "bapr_no_rmdm" "bapr_no_adapt_beta" "bapr_fixed_decay" "bad_bapr"; do
        run $ABL $ENV "ablation" &
        sleep 3
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
            sleep 15
        done
    done
    wait

    echo "Step 2 complete!"
}

# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3: Sensitivity sweep on HalfCheetah (short runs, 500 iters)
#          Paper: sensitivity heatmaps (Q5)
# ═══════════════════════════════════════════════════════════════════════════
step3_sensitivity() {
    echo "╔═══════════════════════════════════════════════╗"
    echo "║  STEP 3: Sensitivity Sweep (HalfCheetah)     ║"
    echo "╚═══════════════════════════════════════════════╝"

    local ENV="HalfCheetah-v2"

    # Hazard rate × penalty scale grid (skip default 0.05/5.0)
    for HR in 0.01 0.03 0.1 0.2; do
        for PS in 1.0 2.5 5.0 7.5 10.0; do
            run "bapr" $ENV "sens_hr${HR}_ps${PS}" \
                "--max_iters 300 --hazard_rate $HR --penalty_scale $PS" &
            sleep 2

            while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
                sleep 15
            done
        done
    done

    # Default HR=0.05 row
    for PS in 1.0 2.5 7.5 10.0; do
        run "bapr" $ENV "sens_hr0.05_ps${PS}" \
            "--max_iters 300 --penalty_scale $PS" &
        sleep 3
    done

    wait
    echo "Step 3 complete!"
}

# ═══════════════════════════════════════════════════════════════════════════
#  STEP 4: Generate all paper figures
# ═══════════════════════════════════════════════════════════════════════════
step4_figures() {
    echo "╔═══════════════════════════════════════════════╗"
    echo "║  STEP 4: Generate Paper Figures               ║"
    echo "╚═══════════════════════════════════════════════╝"

    mkdir -p paper

    # BOCD money plot (needs bocd_belief, surprise_r etc.)
    for ENV in "HalfCheetah-v2" "Hopper-v2" "Ant-v2"; do
        LOG_DIR="${SAVE_ROOT}/bapr_${ENV}_${SEED}/logs"
        [ -d "$LOG_DIR" ] && \
        $PYTHON -m jax_experiments.analysis.plot_bocd_money \
            --log_dir "$LOG_DIR" \
            --output "paper/fig_bocd_money_${ENV}.pdf" || true
    done

    # Ablation curves
    $PYTHON -m jax_experiments.analysis.multiseed_plot \
        --results_root $SAVE_ROOT \
        --envs HalfCheetah-v2 \
        --algos bapr bapr_no_bocd bapr_no_rmdm bapr_no_adapt_beta bapr_fixed_decay \
        --output "paper/fig_ablation_curves.pdf" || true

    # Bad-BAPR divergence
    $PYTHON -m jax_experiments.analysis.multiseed_plot \
        --results_root $SAVE_ROOT \
        --envs HalfCheetah-v2 \
        --algos bapr bad_bapr \
        --output "paper/fig_bad_bapr.pdf" || true

    # Adaptation speed
    for ENV in "HalfCheetah-v2" "Hopper-v2" "Ant-v2"; do
        $PYTHON -m jax_experiments.analysis.adaptation_speed \
            --results_root $SAVE_ROOT --env $ENV \
            --output "paper/fig_adaptation_speed_${ENV}.pdf" || true
    done

    # Embedding t-SNE
    LOG_DIR="${SAVE_ROOT}/bapr_HalfCheetah-v2_${SEED}/logs"
    [ -d "$LOG_DIR" ] && \
    $PYTHON -m jax_experiments.analysis.embedding_viz \
        --log_dir "$LOG_DIR" \
        --output "paper/fig_embedding_tsne.pdf" || true

    echo "Step 4 complete! Figures in paper/"
}

# ═══════════════════════════════════════════════════════════════════════════
STEP="${1:-all}"
case "$STEP" in
    1|main)         step1_main ;;
    2|ablation)     step2_ablation ;;
    3|sensitivity)  step3_sensitivity ;;
    4|figures)      step4_figures ;;
    all)
        step1_main
        step2_ablation
        step3_sensitivity
        step4_figures
        ;;
    *)
        echo "Usage: $0 {1|2|3|4|all}"
        echo "  1/main        — 3 algo × 4 env main results"
        echo "  2/ablation    — Ablation + Bad-BAPR on HalfCheetah"
        echo "  3/sensitivity — Hazard rate × penalty scale sweep"
        echo "  4/figures     — Generate all paper figures"
        echo "  all           — Run everything"
        exit 1
        ;;
esac

echo "Done!"
